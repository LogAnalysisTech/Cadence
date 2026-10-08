from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

from monitor_runtime.runners.support import (
    api_secret,
    configure_runner_experiment,
    extract_monitor_artifacts,
    materialize_phase_policy_snapshot,
    modal_secret,
    redacted_config,
    runner_experiment_config,
    summarize_instance,
    validate_experiment_credentials,
    write_json,
)
from monitor_runtime.provider_error_detection import classify_provider_error


ROOT = Path(__file__).resolve().parents[2]
MINI_ROOT = ROOT / "coding_agent-baselines" / "mini-swe-agent"
VOLUME_NAME = "rafd-mini-swe-agent-artifacts"
DEFAULT_CASES = ("django__django-13158", "pytest-dev__pytest-7490")
ORIGINAL_STEP_LIMIT = 250


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "graphviz", "libgraphviz-dev", "pkg-config", "gcc")
    .pip_install(
        "bashlex",
        "datasets",
        "boto3",
        "jinja2",
        "litellm>=1.75.5",
        "modal",
        "networkx",
        "openai!=1.100.0,!=1.100.1",
        "platformdirs",
        "prompt_toolkit",
        "pydantic>=2",
        "python-dotenv",
        "pyyaml",
        "pygraphviz",
        "requests",
        "rich",
        "swe-rex>=1.4.0",
        "tenacity",
        "textual",
        "typer",
    )
    .add_local_dir(MINI_ROOT / "src", remote_path="/root/mini_src", copy=True)
    .add_local_dir(ROOT / "cadence", remote_path="/root/cadence", copy=True, ignore=["__pycache__/**", "tests/**"])
    .add_local_dir(
        ROOT / "monitor_runtime",
        remote_path="/root/monitor_runtime",
        copy=True,
        ignore=["runs/**", "__pycache__/**"],
    )
    .add_local_dir(
        ROOT / "monitor_baseline",
        remote_path="/root/monitor_baseline",
        copy=True,
        ignore=["__pycache__/**"],
    )
)


app = modal.App("rafd-mini-swe-agent-monitor")
artifact_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _patch_swerex_modal_cleanup() -> None:
    """Terminate live SWE-ReX sandboxes when their deployment is closed.

    SWE-ReX 1.4.x only terminates a sandbox after it has already exited, which
    leaks one Modal container per completed case. Keep this runner-local patch
    until the corrected lifecycle behavior is available in the dependency.
    """

    from swerex.deployment.modal import ModalDeployment

    if getattr(ModalDeployment, "_rafd_cleanup_patch", False):
        return

    async def stop(self) -> None:
        try:
            if self._runtime is not None:
                await self._runtime.close()
        finally:
            self._runtime = None
            sandbox = self._sandbox
            self._sandbox = None
            if sandbox is not None:
                try:
                    if await sandbox.poll.aio() is None:
                        await sandbox.terminate.aio()
                finally:
                    self._app = None

    ModalDeployment.stop = stop
    ModalDeployment._rafd_cleanup_patch = True


@app.function(
    image=image,
    secrets=[api_secret(), modal_secret()],
    volumes={"/artifacts": artifact_volume},
    timeout=6 * 60 * 60,
    retries=0,
)
def run_cases_remote(
    run_id: str,
    cases: list[str],
    agent_model: str,
    selector_model: str,
    reviewer_model: str,
    phase_policy_enabled: bool,
    phase_policy_startup_mode: str,
    enable_monitor: bool = True,
    finding_lifecycle_enabled: bool = True,
    experiment_payload: dict[str, object] | None = None,
    dataset: str = "verified",
    step_limit: int = ORIGINAL_STEP_LIMIT,
) -> dict:
    import sys
    import time
    import traceback
    from pathlib import Path

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/mini_src")
    _patch_swerex_modal_cleanup()
    artifact_volume.reload()

    experiment = configure_runner_experiment(
        experiment_payload=experiment_payload,
        agent_variant="mini-swe-agent",
        run_id=run_id,
        agent_model=agent_model,
        selector_model=selector_model,
        reviewer_model=reviewer_model,
        phase_policy_startup_mode=phase_policy_startup_mode,
        legacy_enable_monitor=enable_monitor,
        legacy_phase_policy_enabled=phase_policy_enabled,
        legacy_finding_lifecycle_enabled=finding_lifecycle_enabled,
    )
    # The serialized experiment payload is the single source of truth for
    # remote workers; positional model arguments remain only for compatibility
    # with older submitted Modal calls.
    agent_model = experiment.agent_model

    from datasets import load_dataset
    from minisweagent.agents import get_agent
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import (
        DATASET_MAPPING,
        get_sb_environment,
        update_preds_file,
    )
    from minisweagent.utils.serialize import recursive_merge

    out_dir = Path("/artifacts") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = {
        inst["instance_id"]: inst
        for inst in load_dataset(DATASET_MAPPING[dataset], split="test")
        if inst["instance_id"] in set(cases)
    }
    missing = sorted(set(cases) - set(instances))
    if missing:
        raise RuntimeError(f"Missing instances: {missing}")

    base_config = get_config_from_spec(
        builtin_config_dir / "benchmarks" / "swebench.yaml"
    )
    experiment_data = experiment.as_dict()
    phase_policy_active = (
        experiment.monitor_mode == "realtime"
        and "phase_policy" not in experiment.ablations
    )
    summary: dict[str, object] = {
        "run_id": run_id,
        "cases": list(cases),
        "started_at": time.time(),
        "execution_location": "modal",
        "dataset": dataset,
        "agent_model": agent_model,
        "selector_model": experiment.selector_model,
        "reviewer_model": experiment.reviewer_model,
        "experiment": experiment_data,
        "monitor_enabled": experiment.monitor_mode != "vanilla",
        "phase_policy": {
            "enabled": phase_policy_active,
            "startup_mode": experiment.phase_policy_startup_mode,
            "model": experiment.phase_policy_model or experiment.selector_model,
        },
        "step_limit": step_limit,
        "results": {},
    }
    write_json(out_dir / "summary.json", summary)

    for instance_id in cases:
        print(f"[{run_id}] starting {instance_id}", flush=True)
        instance_dir = out_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        traj_path = instance_dir / f"{instance_id}.traj.json"
        log_path = instance_dir / "runner.log"
        result = ""
        exit_status = ""
        agent = None
        env = None

        with log_path.open("w", encoding="utf-8") as log:
            try:
                config = recursive_merge(
                    base_config,
                    {
                        "agent": {
                            "step_limit": step_limit,
                            "cost_limit": 3.0,
                            "output_path": traj_path,
                        },
                        "environment": {
                            "environment_class": "swerex_modal",
                            "startup_timeout": 300.0,
                            "runtime_timeout": 3600.0,
                            "deployment_timeout": 3600.0,
                        },
                        "model": {
                            "model_name": agent_model,
                            "cost_tracking": "ignore_errors",
                            "model_kwargs": {
                                "temperature": 0,
                                "drop_params": True,
                                "parallel_tool_calls": True,
                            },
                        },
                    },
                )
                log.write(
                    json.dumps({"config": redacted_config(config)}, indent=2) + "\n"
                )
                env = get_sb_environment(config, instances[instance_id])
                agent = get_agent(
                    get_model(config=config["model"]),
                    env,
                    config["agent"],
                    default_type="default",
                )
                info = agent.run(instances[instance_id]["problem_statement"])
                exit_status = info.get("exit_status", "")
                result = info.get("submission", "")
                log.write(
                    f"Finished {instance_id}: {exit_status}, patch_chars={len(result)}\n"
                )
            except Exception as exc:
                exit_status = type(exc).__name__
                log.write(f"Exception for {instance_id}: {exc}\n")
                log.write(traceback.format_exc())
                provider_error = classify_provider_error(str(exc))
                if provider_error:
                    print(
                        f"[{run_id}] provider_error {instance_id}: {provider_error}",
                        flush=True,
                    )
                print(f"[{run_id}] exception {instance_id}: {exc}", flush=True)
            finally:
                if agent is not None:
                    agent.save(
                        traj_path,
                        {
                            "instance_id": instance_id,
                            "info": {
                                "exit_status": exit_status,
                                "submission": result,
                            },
                        },
                    )
                if env is not None:
                    try:
                        env.stop()
                    except Exception as exc:
                        log.write(f"Environment stop failed: {exc}\n")
                update_preds_file(
                    out_dir / "preds.json",
                    instance_id,
                    f"mini-swe-agent-{agent_model}-{experiment.monitor_mode}",
                    result,
                )

        extract_monitor_artifacts(
            instance_id,
            traj_path,
            instance_dir,
            include_monitor_views=experiment.monitor_mode != "vanilla",
        )
        (instance_dir / f"{instance_id}.patch").write_text(result, encoding="utf-8")
        summary["results"][instance_id] = summarize_instance(traj_path)
        write_json(out_dir / "summary.json", summary)
        artifact_volume.commit()
        print(f"[{run_id}] finished {instance_id}: {exit_status}", flush=True)

    summary["finished_at"] = time.time()
    summary["phase_policy"]["artifacts"] = materialize_phase_policy_snapshot(
        run_id=run_id,
        agent_variant="mini-swe-agent",
        startup_mode=experiment.phase_policy_startup_mode,
        enabled=phase_policy_active,
    )
    write_json(out_dir / "summary.json", summary)
    artifact_volume.commit()
    return summary


@app.function(
    image=image,
    secrets=[api_secret(), modal_secret()],
    volumes={"/artifacts": artifact_volume},
    timeout=6 * 60 * 60,
    retries=0,
)
def run_case_remote(
    run_id: str,
    instance_id: str,
    agent_model: str,
    selector_model: str,
    reviewer_model: str,
    phase_policy_enabled: bool,
    phase_policy_startup_mode: str,
    enable_monitor: bool = True,
    finding_lifecycle_enabled: bool = True,
    experiment_payload: dict[str, object] | None = None,
    dataset: str = "verified",
    step_limit: int = ORIGINAL_STEP_LIMIT,
) -> dict:
    import json
    import sys
    import traceback
    from pathlib import Path

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/mini_src")
    _patch_swerex_modal_cleanup()
    artifact_volume.reload()

    experiment = configure_runner_experiment(
        experiment_payload=experiment_payload,
        agent_variant="mini-swe-agent",
        run_id=run_id,
        agent_model=agent_model,
        selector_model=selector_model,
        reviewer_model=reviewer_model,
        phase_policy_startup_mode=phase_policy_startup_mode,
        legacy_enable_monitor=enable_monitor,
        legacy_phase_policy_enabled=phase_policy_enabled,
        legacy_finding_lifecycle_enabled=finding_lifecycle_enabled,
    )
    agent_model = experiment.agent_model

    from datasets import load_dataset
    from minisweagent.agents import get_agent
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import DATASET_MAPPING, get_sb_environment
    from minisweagent.utils.serialize import recursive_merge

    out_dir = Path("/artifacts") / run_id
    instance_dir = out_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prediction_parts").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary_parts").mkdir(parents=True, exist_ok=True)

    instances = {
        inst["instance_id"]: inst
        for inst in load_dataset(DATASET_MAPPING[dataset], split="test")
        if inst["instance_id"] == instance_id
    }
    if instance_id not in instances:
        raise RuntimeError(f"Missing instance: {instance_id}")

    base_config = get_config_from_spec(
        builtin_config_dir / "benchmarks" / "swebench.yaml"
    )
    traj_path = instance_dir / f"{instance_id}.traj.json"
    log_path = instance_dir / "runner.log"
    result = ""
    exit_status = ""
    agent = None
    env = None

    print(f"[{run_id}] starting {instance_id}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            config = recursive_merge(
                base_config,
                {
                    "agent": {
                        "step_limit": step_limit,
                        "cost_limit": 3.0,
                        "output_path": traj_path,
                    },
                    "environment": {
                        "environment_class": "swerex_modal",
                        "startup_timeout": 300.0,
                        "runtime_timeout": 3600.0,
                        "deployment_timeout": 3600.0,
                    },
                    "model": {
                        "model_name": agent_model,
                        "cost_tracking": "ignore_errors",
                        "model_kwargs": {
                            "temperature": 0,
                            "drop_params": True,
                            "parallel_tool_calls": True,
                        },
                    },
                },
            )
            log.write(json.dumps({"config": redacted_config(config)}, indent=2) + "\n")
            env = get_sb_environment(config, instances[instance_id])
            agent = get_agent(
                get_model(config=config["model"]),
                env,
                config["agent"],
                default_type="default",
            )
            info = agent.run(instances[instance_id]["problem_statement"])
            exit_status = info.get("exit_status", "")
            result = info.get("submission", "")
            log.write(
                f"Finished {instance_id}: {exit_status}, patch_chars={len(result)}\n"
            )
        except Exception as exc:
            exit_status = type(exc).__name__
            log.write(f"Exception for {instance_id}: {exc}\n")
            log.write(traceback.format_exc())
            provider_error = classify_provider_error(str(exc))
            if provider_error:
                print(
                    f"[{run_id}] provider_error {instance_id}: {provider_error}",
                    flush=True,
                )
            print(f"[{run_id}] exception {instance_id}: {exc}", flush=True)
        finally:
            if agent is not None:
                agent.save(
                    traj_path,
                    {
                        "instance_id": instance_id,
                        "info": {
                            "exit_status": exit_status,
                            "submission": result,
                        },
                    },
                )
            if env is not None:
                try:
                    env.stop()
                except Exception as exc:
                    log.write(f"Environment stop failed: {exc}\n")

    extract_monitor_artifacts(
        instance_id,
        traj_path,
        instance_dir,
        include_monitor_views=experiment.monitor_mode != "vanilla",
    )
    (instance_dir / f"{instance_id}.patch").write_text(result, encoding="utf-8")
    prediction = {
        "model_name_or_path": f"mini-swe-agent-{agent_model}-{experiment.monitor_mode}",
        "instance_id": instance_id,
        "model_patch": result,
    }
    write_json(out_dir / "prediction_parts" / f"{instance_id}.json", prediction)
    summary = summarize_instance(traj_path)
    summary["instance_id"] = instance_id
    write_json(out_dir / "summary_parts" / f"{instance_id}.json", summary)
    artifact_volume.commit()
    print(f"[{run_id}] finished {instance_id}: {exit_status}", flush=True)
    return summary


@app.function(
    image=image,
    volumes={"/artifacts": artifact_volume},
    timeout=30 * 60,
    retries=0,
)
def finalize_run_remote(
    run_id: str,
    cases: list[str],
    agent_model: str,
    selector_model: str,
    reviewer_model: str,
    started_at: float,
    phase_policy_enabled: bool,
    phase_policy_startup_mode: str,
    enable_monitor: bool = True,
    experiment_payload: dict[str, object] | None = None,
    dataset: str = "verified",
    step_limit: int = ORIGINAL_STEP_LIMIT,
) -> dict:
    import json
    import time
    from pathlib import Path

    artifact_volume.reload()
    out_dir = Path("/artifacts") / run_id
    if not cases:
        cases = sorted(
            path.stem for path in (out_dir / "summary_parts").glob("*.json")
        )
    results = {}
    predictions = []
    for instance_id in cases:
        summary_path = out_dir / "summary_parts" / f"{instance_id}.json"
        prediction_path = out_dir / "prediction_parts" / f"{instance_id}.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.pop("instance_id", None)
            results[instance_id] = summary
        else:
            results[instance_id] = {
                "trajectory_exists": False,
                "missing_summary_part": True,
            }
        if prediction_path.exists():
            predictions.append(json.loads(prediction_path.read_text(encoding="utf-8")))
    experiment = (
        experiment_payload
        or runner_experiment_config(
            monitor_mode="",
            agent_model=agent_model,
            selector_model=selector_model,
            reviewer_model=reviewer_model,
            phase_policy_startup_mode=phase_policy_startup_mode,
            legacy_enable_monitor=enable_monitor,
            legacy_phase_policy_enabled=phase_policy_enabled,
        ).as_dict()
    )
    phase_policy_active = (
        experiment["monitor_mode"] == "realtime"
        and "phase_policy" not in experiment["ablations"]
    )
    summary = {
        "run_id": run_id,
        "cases": list(cases),
        "started_at": started_at,
        "finished_at": time.time(),
        "execution_location": "modal",
        "dataset": dataset,
        "agent_model": agent_model,
        "selector_model": selector_model,
        "reviewer_model": reviewer_model,
        "experiment": experiment,
        "monitor_enabled": experiment["monitor_mode"] != "vanilla",
        "phase_policy": {
            "enabled": phase_policy_active,
            "startup_mode": experiment["phase_policy_startup_mode"],
            "model": experiment.get("phase_policy_model") or experiment["selector_model"],
            "artifacts": materialize_phase_policy_snapshot(
                run_id=run_id,
                agent_variant="mini-swe-agent",
                startup_mode=experiment["phase_policy_startup_mode"],
                enabled=phase_policy_active,
            ),
        },
        "step_limit": step_limit,
        "results": results,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "preds.json", predictions)
    artifact_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    run_id: str = "",
    cases: str = ",".join(DEFAULT_CASES),
    agent_model: str = "gpt-5-mini",
    selector_model: str = "gpt-5-mini",
    reviewer_model: str = "gpt-5.2",
    parallelism: int = 1,
    enable_monitor: bool = True,
    phase_policy_enabled: bool = True,
    phase_policy_startup_mode: str = "cold",
    finding_lifecycle_enabled: bool = True,
    monitor_mode: str = "",
    phase_policy_model: str = "",
    liveplan_variant: str = "monitor",
    liveplan_refiner_model: str = "",
    liveplan_refiner_config: str = "",
    adaptive_interval_fixed: int = 0,
    adaptive_interval_base: int = 5,
    adaptive_interval_growth: int = 2,
    adaptive_interval_execution: str = "severity",
    ablations: str = "",
    dataset: str = "verified",
    summary_cases: str = "",
    continuous_map: bool = False,
    step_limit: int = ORIGINAL_STEP_LIMIT,
) -> None:
    import time

    resolved_run_id = run_id or f"mini_modal_monitor_{time.strftime('%Y%m%d_%H%M%S')}"
    case_list = [case.strip() for case in cases.split(",") if case.strip()]
    summary_case_list = [
        case.strip() for case in summary_cases.split(",") if case.strip()
    ] or case_list
    if dataset not in {"lite", "verified"}:
        raise ValueError("dataset must be lite or verified")
    if phase_policy_startup_mode not in {"cold", "warm"}:
        raise ValueError("phase_policy_startup_mode must be cold or warm")
    experiment = runner_experiment_config(
        monitor_mode=monitor_mode,
        agent_model=agent_model,
        selector_model=selector_model,
        reviewer_model=reviewer_model,
        phase_policy_model=phase_policy_model,
        liveplan_variant=liveplan_variant,
        liveplan_refiner_model=liveplan_refiner_model,
        liveplan_refiner_config=liveplan_refiner_config,
        adaptive_interval_fixed=adaptive_interval_fixed,
        adaptive_interval_base=adaptive_interval_base,
        adaptive_interval_growth=adaptive_interval_growth,
        adaptive_interval_execution=adaptive_interval_execution,
        ablations=ablations,
        phase_policy_startup_mode=phase_policy_startup_mode,
        legacy_enable_monitor=enable_monitor,
        legacy_phase_policy_enabled=phase_policy_enabled,
        legacy_finding_lifecycle_enabled=finding_lifecycle_enabled,
    )
    experiment_payload = experiment.as_dict()
    validate_experiment_credentials(experiment)
    if parallelism <= 1:
        summary = run_cases_remote.remote(
            resolved_run_id,
            case_list,
            agent_model,
            selector_model,
            reviewer_model,
            phase_policy_enabled,
            phase_policy_startup_mode,
            enable_monitor,
            finding_lifecycle_enabled,
            experiment_payload,
            dataset,
            step_limit,
        )
    elif continuous_map:
        started_at = time.time()
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = [
                executor.submit(
                    run_case_remote.remote,
                    resolved_run_id,
                    instance_id,
                    agent_model,
                    selector_model,
                    reviewer_model,
                    phase_policy_enabled,
                    phase_policy_startup_mode,
                    enable_monitor,
                    finding_lifecycle_enabled,
                    experiment_payload,
                    dataset,
                    step_limit,
                )
                for instance_id in case_list
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"Remote case failed: {type(exc).__name__}: {exc}")
        summary = finalize_run_remote.remote(
            resolved_run_id,
            summary_case_list,
            agent_model,
            selector_model,
            reviewer_model,
            started_at,
            phase_policy_enabled,
            phase_policy_startup_mode,
            enable_monitor,
            experiment_payload,
            dataset,
            step_limit,
        )
    else:
        started_at = time.time()
        for start in range(0, len(case_list), parallelism):
            batch = case_list[start : start + parallelism]
            print(f"Running batch {start // parallelism + 1}: {batch}")
            list(
                run_case_remote.map(
                    [resolved_run_id] * len(batch),
                    batch,
                    [agent_model] * len(batch),
                    [selector_model] * len(batch),
                    [reviewer_model] * len(batch),
                    [phase_policy_enabled] * len(batch),
                    [phase_policy_startup_mode] * len(batch),
                    [enable_monitor] * len(batch),
                    [finding_lifecycle_enabled] * len(batch),
                    [experiment_payload] * len(batch),
                    [dataset] * len(batch),
                    [step_limit] * len(batch),
                )
            )
        summary = finalize_run_remote.remote(
            resolved_run_id,
            summary_case_list,
            agent_model,
            selector_model,
            reviewer_model,
            started_at,
            phase_policy_enabled,
            phase_policy_startup_mode,
            enable_monitor,
            experiment_payload,
            dataset,
            step_limit,
        )
    print(json.dumps(summary, indent=2))
    print(f"Artifacts volume: {VOLUME_NAME}")
    print(f"Artifacts path: /{resolved_run_id}")

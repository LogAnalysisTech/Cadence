from __future__ import annotations

import base64
import json
import os
import shlex
import time
from pathlib import Path

import modal
from monitor_runtime.runners.support import modal_secret as _modal_secret


ROOT = Path(__file__).resolve().parents[2]
MINI_ROOT = ROOT / "coding_agent-baselines" / "mini-swe-agent"
VOLUME_NAME = "rafd-mini-swe-agent-artifacts"
MAX_EVAL_CONTAINERS = int(
    os.getenv("REALTIME_MONITOR_EVAL_MAX_CONTAINERS", "20")
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "boto3",
        "datasets",
        "jinja2",
        "modal",
        "platformdirs",
        "prompt_toolkit",
        "pydantic>=2",
        "python-dotenv",
        "pyyaml",
        "rich",
        "swebench",
        "swe-rex>=1.4.0",
        "tenacity",
        "textual",
        "typer",
    )
    .add_local_dir(MINI_ROOT / "src", remote_path="/root/mini_src", copy=True)
)

app = modal.App("rafd-swebench-evaluator")
artifact_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def _patch_swerex_modal_cleanup() -> None:
    """Prevent SWE-ReX 1.4.x from leaking completed Modal sandboxes."""

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
    secrets=[_modal_secret()],
    volumes={"/artifacts": artifact_volume},
    timeout=60 * 60,
    retries=0,
    max_containers=MAX_EVAL_CONTAINERS,
)
def evaluate_case(
    run_id: str,
    instance_id: str,
    artifact_scope: str = "shared",
    dataset: str = "verified",
) -> dict[str, object]:
    import sys
    import time
    import traceback

    from modal.exception import SandboxTimeoutError

    sys.path.insert(0, "/root/mini_src")
    _patch_swerex_modal_cleanup()

    from datasets import load_dataset
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.run.benchmarks.swebench import DATASET_MAPPING, get_sb_environment
    from minisweagent.utils.serialize import recursive_merge
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    artifact_root = _artifact_root(artifact_scope)
    out_dir = artifact_root / run_id / "evaluation_modal" / instance_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = (
        artifact_root / run_id / "prediction_parts" / f"{instance_id}.json"
    )
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    instance = next(
        item
        for item in load_dataset(DATASET_MAPPING[dataset], split="test")
        if item["instance_id"] == instance_id
    )
    test_spec = make_test_spec(instance)
    base_config = get_config_from_spec(
        builtin_config_dir / "benchmarks" / "swebench.yaml"
    )
    config = recursive_merge(
        base_config,
        {
            "environment": {
                "environment_class": "swerex_modal",
                "startup_timeout": 900.0,
                "runtime_timeout": 3600.0,
                "deployment_timeout": 3600.0,
            }
        },
    )

    result: dict[str, object] = {
        "instance_id": instance_id,
        "resolved": False,
        "evaluation_completed": False,
        "patch_chars": len(prediction.get("model_patch") or ""),
    }
    env = None
    try:
        for attempt in range(3):
            try:
                env = get_sb_environment(config, instance)
                break
            except SandboxTimeoutError:
                if attempt == 2:
                    raise
                time.sleep(30 * (attempt + 1))
        patch = prediction.get("model_patch") or ""
        if not patch:
            result["failure"] = "empty_patch"
        else:
            _write_remote_bytes(
                env,
                remote_path="/tmp/model.patch",
                data=patch.encode(),
            )
            apply_result = env.execute(
                {"command": "git apply -v /tmp/model.patch"},
                cwd="/testbed",
                timeout=300,
            )
            result["patch_apply_returncode"] = apply_result["returncode"]
            (out_dir / "patch_apply.log").write_text(
                apply_result["output"] or "", encoding="utf-8"
            )
            if apply_result["returncode"] != 0:
                result["failure"] = "patch_apply_failed"
            else:
                _write_remote_bytes(
                    env,
                    remote_path="/tmp/eval.sh",
                    data=test_spec.eval_script.encode(),
                )
                eval_result = _run_evaluation_script(env)
                test_output_path = out_dir / "test_output.log"
                test_output_path.write_text(
                    eval_result["output"] or "", encoding="utf-8"
                )
                result["eval_returncode"] = eval_result["returncode"]
                failure = _evaluation_execution_failure(eval_result)
                if failure:
                    result["failure"] = failure
                    result["error"] = eval_result.get("exception_info") or "Test execution did not reach the end marker"
                    return result
                report = get_eval_report(
                    test_spec=test_spec,
                    prediction=prediction,
                    test_log_path=test_output_path,
                    include_tests_status=True,
                )
                (out_dir / "report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
                result["resolved"] = bool(report[instance_id]["resolved"])
                result["tests_status"] = report[instance_id].get("tests_status", {})
                result["evaluation_completed"] = bool(result["tests_status"])
                if not result["evaluation_completed"]:
                    result["failure"] = "missing_test_results"
    except Exception as exc:
        result["failure"] = type(exc).__name__
        result["error"] = str(exc)
        (out_dir / "traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if env is not None:
            env.stop()
        (out_dir / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        artifact_volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/artifacts": artifact_volume},
    timeout=10 * 60,
)
def finalize_evaluation(
    run_id: str,
    cases: list[str],
    artifact_scope: str = "shared",
    dataset: str = "verified",
) -> dict[str, object]:
    eval_dir = _artifact_root(artifact_scope) / run_id / "evaluation_modal"
    results = {}
    for instance_id in cases:
        path = eval_dir / instance_id / "result.json"
        results[instance_id] = (
            json.loads(path.read_text()) if path.exists() else {"missing": True}
        )
    summary = {
        "run_id": run_id,
        "dataset": dataset,
        "cases": cases,
        "resolved": [
            case for case, result in results.items() if result.get("resolved")
        ],
        "results": results,
    }
    (eval_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    artifact_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    run_id: str,
    cases: str,
    artifact_scope: str = "shared",
    parallelism: int = 6,
    summary_cases: str = "",
    dataset: str = "verified",
    continuous_map: bool = False,
) -> None:
    case_list = [case.strip() for case in cases.split(",") if case.strip()]
    summary_case_list = [
        case.strip() for case in summary_cases.split(",") if case.strip()
    ] or case_list
    if artifact_scope != "shared":
        raise ValueError("artifact_scope must be shared")
    if dataset not in {"lite", "verified"}:
        raise ValueError("dataset must be lite or verified")
    if parallelism < 1:
        raise ValueError("parallelism must be at least 1")
    if continuous_map:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            future_cases = {
                executor.submit(
                    evaluate_case.remote,
                    run_id,
                    instance_id,
                    artifact_scope,
                    dataset,
                ): instance_id
                for instance_id in case_list
            }
            for future in as_completed(future_cases):
                try:
                    future.result()
                except Exception as exc:
                    instance_id = future_cases[future]
                    print(
                        f"Remote evaluation failed for {instance_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
    else:
        batches = [
            case_list[start : start + parallelism]
            for start in range(0, len(case_list), parallelism)
        ]
        for batch in batches:
            list(
                evaluate_case.map(
                    [run_id] * len(batch),
                    batch,
                    [artifact_scope] * len(batch),
                    [dataset] * len(batch),
                )
            )
    print(
        json.dumps(
            finalize_evaluation.remote(
                run_id, summary_case_list, artifact_scope, dataset
            ),
            indent=2,
        )
    )


def _artifact_root(artifact_scope: str) -> Path:
    if artifact_scope == "shared":
        return Path("/artifacts")
    raise ValueError("artifact_scope must be shared")


def _evaluation_execution_failure(result: dict) -> str | None:
    if result["returncode"] < 0 or result.get("exception_info"):
        return "evaluation_execution_error"
    if ">>>>> End Test Output" not in result.get("output", ""):
        return "incomplete_test_output"
    return None


def _run_evaluation_script(
    env, *, script_path: str = "/tmp/eval.sh", timeout: float = 1800,
    poll_interval: float = 5,
) -> dict:
    """Poll a detached process to avoid SWE-ReX's 300-second HTTP timeout."""
    output_path = script_path + ".output"
    status_path = script_path + ".status"
    wrapper = (
        f"/bin/bash {shlex.quote(script_path)}; code=$?; "
        f"printf '%s' \"$code\" > {shlex.quote(status_path)}; exit \"$code\""
    )
    launcher = (
        "import pathlib, subprocess; "
        f"pathlib.Path({status_path!r}).unlink(missing_ok=True); "
        f"output=open({output_path!r}, 'w'); "
        f"p=subprocess.Popen(['/bin/bash', '-c', {wrapper!r}], "
        "stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True); "
        "print(p.pid)"
    )
    started = env.execute({"command": "python -c " + shlex.quote(launcher)}, timeout=30)
    if started["returncode"] != 0:
        raise RuntimeError(f"Could not start evaluation: {started['output']}")
    pid = int(started["output"].strip())
    deadline = time.monotonic() + timeout
    while True:
        status = env.execute(
            {"command": f"if test -f {shlex.quote(status_path)}; then cat {shlex.quote(status_path)}; fi"},
            timeout=30,
        )
        if status["returncode"] != 0:
            raise RuntimeError(f"Could not poll evaluation: {status['output']}")
        if status["output"].strip():
            output = env.execute({"command": "cat " + shlex.quote(output_path)}, timeout=60)
            if output["returncode"] != 0:
                raise RuntimeError(f"Could not read evaluation output: {output['output']}")
            return {"returncode": int(status["output"].strip()), "output": output["output"]}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            env.execute({"command": f"kill -KILL -- -{pid}"}, timeout=30)
            output = env.execute({"command": "cat " + shlex.quote(output_path)}, timeout=60)
            return {
                "returncode": -1, "output": output["output"],
                "exception_info": f"Evaluation exceeded {timeout:g} seconds",
            }
        time.sleep(min(poll_interval, remaining))


def _write_remote_bytes(env, *, remote_path: str, data: bytes) -> None:
    """Transfer large evaluator inputs without exceeding shell command limits."""

    encoded = base64.b64encode(data).decode("ascii")
    encoded_path = f"{remote_path}.b64"
    commands = [f": > {encoded_path}"]
    commands.extend(
        f"printf '%s' '{encoded[start : start + 24_000]}' >> {encoded_path}"
        for start in range(0, len(encoded), 24_000)
    )
    commands.append(
        'python -c "import base64; '
        f"open('{remote_path}','wb').write(base64.b64decode(open('{encoded_path}').read()))\""
    )
    for command in commands:
        transfer = env.execute({"command": command}, timeout=60)
        if transfer["returncode"] != 0:
            raise RuntimeError(f"Could not transfer {remote_path}: {transfer['output']}")

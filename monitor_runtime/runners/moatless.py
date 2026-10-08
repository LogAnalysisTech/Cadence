from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

import modal

from monitor_runtime.runners.support import (
    api_secret as _api_secret,
    runner_experiment_config,
    validate_experiment_credentials,
)


ROOT = Path(__file__).resolve().parents[2]
MOATLESS_ROOT = ROOT / "coding_agent-baselines" / "moatless"
VOLUME_NAME = "rafd-mini-swe-agent-artifacts"
INDEX_VOLUME_NAME = "rafd-moatless-code-indexes"
REPO_VOLUME_NAME = "rafd-moatless-repo-cache"
FLOW_PATH = ROOT / "configs" / "moatless-swebench.json"
MAX_ITERATIONS = 50
INDEX_ROOT = Path("/index_store")
INDEX_SCHEMA_VERSION = "voyage-code-3-1024-v1"
INDEX_MANIFEST_NAME = ".rafd-index.json"
REPO_ROOT = Path("/repo_store")
REPO_CACHE_VERSION = "git-mirror-v1"
REPO_MANIFEST_NAME = ".rafd-repo-cache.json"
DATASET_MAPPING = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "graphviz", "libgraphviz-dev", "pkg-config", "gcc")
    .pip_install_from_pyproject(MOATLESS_ROOT / "pyproject.toml")
    .pip_install(
        "bashlex",
        "litellm==1.76.0",
        "networkx",
        "openai==1.101.0",
        "pygraphviz",
        "requests",
        "tree-sitter-python==0.23.6",
    )
    .add_local_dir(
        MOATLESS_ROOT / "moatless", remote_path="/root/moatless_src/moatless", copy=True
    )
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

app = modal.App("rafd-moatless-monitor")
artifact_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
index_volume = modal.Volume.from_name(INDEX_VOLUME_NAME, create_if_missing=True)
repo_volume = modal.Volume.from_name(REPO_VOLUME_NAME, create_if_missing=True)


def _index_metadata(instance: dict) -> dict[str, object]:
    """Describe the immutable repository index used by one benchmark case."""

    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "embedding_model": "voyage-code-3",
        "dimensions": 1024,
    }


def _index_dir(instance: dict) -> Path:
    """Return a versioned cache path that cannot cross repository revisions."""

    return (
        INDEX_ROOT
        / INDEX_SCHEMA_VERSION
        / instance["instance_id"]
        / instance["base_commit"]
    )


def _index_is_ready(index_dir: Path, instance: dict) -> bool:
    manifest_path = index_dir / INDEX_MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest == _index_metadata(instance) and any(
        path.name != INDEX_MANIFEST_NAME for path in index_dir.iterdir()
    )


def _repo_cache_key(repo: str) -> str:
    return repo.replace("/", "__")


def _repo_mirror_dir(repo: str) -> Path:
    return REPO_ROOT / REPO_CACHE_VERSION / f"{_repo_cache_key(repo)}.git"


def _repo_manifest_path(repo: str) -> Path:
    return _repo_mirror_dir(repo).parent / f"{_repo_cache_key(repo)}.manifest.json"


def _git_commit_exists(repo_dir: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _repo_cache_is_ready(repo: str, required_commits: list[str]) -> bool:
    mirror_dir = _repo_mirror_dir(repo)
    manifest_path = _repo_manifest_path(repo)
    if not mirror_dir.is_dir() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        manifest.get("cache_version") != REPO_CACHE_VERSION
        or manifest.get("repo") != repo
    ):
        return False
    verified = set(manifest.get("verified_commits", []))
    return set(required_commits).issubset(verified) and all(
        _git_commit_exists(mirror_dir, commit) for commit in required_commits
    )


def _clone_instance_repository(instance: dict, repo_path: Path) -> None:
    mirror_dir = _repo_mirror_dir(instance["repo"])
    if not _repo_cache_is_ready(instance["repo"], [instance["base_commit"]]):
        raise FileNotFoundError(
            "Repository cache is missing or incomplete for "
            f"{instance['repo']}@{instance['base_commit']}. "
            "Run this Modal entrypoint with --prepare-repositories first."
        )
    shutil.rmtree(repo_path, ignore_errors=True)
    _run(
        [
            "git",
            "clone",
            "--shared",
            "--no-checkout",
            str(mirror_dir),
            str(repo_path),
        ]
    )
    _run(["git", "checkout", instance["base_commit"]], cwd=repo_path)
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, text=True
    ).strip()
    if actual_commit != instance["base_commit"]:
        raise RuntimeError(
            f"Repository cache checkout mismatch: expected {instance['base_commit']}, "
            f"got {actual_commit}"
        )


@app.function(
    image=image,
    volumes={"/repo_store": repo_volume},
    timeout=6 * 60 * 60,
    retries=0,
)
def prepare_repository_remote(
    repo: str, required_commits: list[str]
) -> dict[str, object]:
    """Build one immutable full Git mirror covering every requested benchmark revision."""

    repo_volume.reload()
    required_commits = sorted(set(required_commits))
    if _repo_cache_is_ready(repo, required_commits):
        return {
            "repo": repo,
            "status": "cached",
            "commits": len(required_commits),
            "mirror_dir": str(_repo_mirror_dir(repo)),
        }

    started = time.monotonic()
    staging_root = Path("/tmp/repo_cache_build")
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / f"{_repo_cache_key(repo)}.git"
    shutil.rmtree(staging_dir, ignore_errors=True)
    _run(
        [
            "git",
            "clone",
            "--mirror",
            f"https://github.com/{repo}.git",
            str(staging_dir),
        ]
    )
    missing = [
        commit
        for commit in required_commits
        if not _git_commit_exists(staging_dir, commit)
    ]
    if missing:
        # A benchmark revision can occasionally be outside the currently advertised
        # branch tips. Ask for those immutable objects explicitly while this is still
        # the cache-building phase; benchmark workers themselves never use the network.
        _run(["git", "fetch", "origin", *missing], cwd=staging_dir)
        missing = [
            commit
            for commit in required_commits
            if not _git_commit_exists(staging_dir, commit)
        ]
        if missing:
            raise RuntimeError(
                f"Mirror for {repo} is missing {len(missing)} required commit(s): "
                f"{missing[:5]}"
            )
    _run(["git", "repack", "-a", "-d"], cwd=staging_dir)

    mirror_dir = _repo_mirror_dir(repo)
    manifest_path = _repo_manifest_path(repo)
    mirror_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(mirror_dir, ignore_errors=True)
    shutil.copytree(staging_dir, mirror_dir, symlinks=True)
    manifest_path.write_text(
        json.dumps(
            {
                "cache_version": REPO_CACHE_VERSION,
                "repo": repo,
                "remote": f"https://github.com/{repo}.git",
                "verified_commits": required_commits,
                "prepared_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    repo_volume.commit()
    return {
        "repo": repo,
        "status": "built",
        "commits": len(required_commits),
        "mirror_dir": str(mirror_dir),
        "build_seconds": time.monotonic() - started,
    }


@app.function(
    image=image,
    secrets=[_api_secret()],
    volumes={"/index_store": index_volume, "/repo_store": repo_volume},
    timeout=6 * 60 * 60,
    retries=0,
)
def prepare_index_remote(
    instance_id: str, dataset: str = "verified"
) -> dict[str, object]:
    """Build a missing code index once, outside the measured agent run."""

    import sys

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/moatless_src")
    index_volume.reload()
    repo_volume.reload()

    from datasets import load_dataset
    from moatless.index.code_index import CodeIndex
    from moatless.index.settings import IndexSettings
    from moatless.repository.git import GitRepository

    dataset_name = DATASET_MAPPING.get(dataset)
    if dataset_name is None:
        raise ValueError("dataset must be lite or verified")
    instance = next(
        item
        for item in load_dataset(dataset_name, split="test")
        if item["instance_id"] == instance_id
    )
    index_dir = _index_dir(instance)
    if _index_is_ready(index_dir, instance):
        return {
            "instance_id": instance_id,
            "status": "cached",
            "index_dir": str(index_dir),
        }

    started = time.monotonic()
    repo_path = Path("/tmp/index_repos") / instance_id
    _clone_instance_repository(instance, repo_path)
    repository = GitRepository(repo_path=str(repo_path), shadow_mode=True)
    code_index = CodeIndex(
        file_repo=repository,
        settings=IndexSettings(embed_model="voyage-code-3", dimensions=1024),
    )
    # A single embedding worker stays below the shared Voyage 3M TPM limit.
    code_index.run_ingestion(num_workers=1)

    staging_dir = Path("/tmp/index_build") / instance_id
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    code_index.persist(persist_dir=str(staging_dir))
    (staging_dir / INDEX_MANIFEST_NAME).write_text(
        json.dumps(_index_metadata(instance), indent=2), encoding="utf-8"
    )

    shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging_dir, index_dir)
    index_volume.commit()
    return {
        "instance_id": instance_id,
        "status": "built",
        "index_dir": str(index_dir),
        "build_seconds": time.monotonic() - started,
    }


@app.function(
    image=image,
    secrets=[_api_secret()],
    volumes={
        "/artifacts": artifact_volume,
        "/index_store": index_volume,
        "/repo_store": repo_volume,
    },
    timeout=6 * 60 * 60,
    retries=0,
    # Moatless/LiteLLM can leave asynchronous logging file handles alive after
    # a case returns. Reusing that container makes the next Volume.reload()
    # fail before the agent starts, so isolate each benchmark case in a fresh
    # worker lifecycle.
    single_use_containers=True,
)
def run_case_remote(
    run_id: str,
    instance_id: str,
    flow_config: dict,
    agent_model: str,
    selector_model: str,
    reviewer_model: str,
    phase_policy_enabled: bool,
    phase_policy_startup_mode: str,
    enable_monitor: bool = True,
    experiment_payload: dict[str, object] | None = None,
    dataset: str = "verified",
) -> dict:
    import asyncio
    import sys
    import traceback

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/moatless_src")
    artifact_volume.reload()
    index_volume.reload()
    repo_volume.reload()
    import os

    from monitor_runtime.runners.support import configure_runner_experiment

    os.environ["MOATLESS_DIR"] = "/root/moatless_src"
    experiment = configure_runner_experiment(
        experiment_payload=experiment_payload,
        agent_variant="moatless",
        run_id=run_id,
        agent_model=agent_model,
        selector_model=selector_model,
        reviewer_model=reviewer_model,
        phase_policy_startup_mode=phase_policy_startup_mode,
        legacy_enable_monitor=enable_monitor,
        legacy_phase_policy_enabled=phase_policy_enabled,
    )
    # Prefer the serialized experiment contract over legacy positional values.
    agent_model = experiment.agent_model

    from datasets import load_dataset

    out_dir = Path("/artifacts") / run_id
    case_dir = out_dir / "moatless" / instance_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prediction_parts").mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "instance_id": instance_id,
        "agent": "moatless",
        "status": "running",
        "experiment": experiment.as_dict(),
        "monitor_enabled": experiment.monitor_mode != "vanilla",
        "phase_policy": {
            "enabled": experiment.monitor_mode == "realtime"
            and "phase_policy" not in experiment.ablations,
            "startup_mode": experiment.phase_policy_startup_mode,
            "model": experiment.phase_policy_model or experiment.selector_model,
        },
    }
    try:
        dataset_name = DATASET_MAPPING.get(dataset)
        if dataset_name is None:
            raise ValueError("dataset must be lite or verified")
        instance = next(
            item
            for item in load_dataset(dataset_name, split="test")
            if item["instance_id"] == instance_id
        )
        repo_path = Path("/tmp/repos") / instance_id
        _clone_instance_repository(instance, repo_path)

        index_dir = _index_dir(instance)
        if not _index_is_ready(index_dir, instance):
            raise FileNotFoundError(
                f"Prebuilt index is missing for {instance_id}. "
                "Run this Modal entrypoint with --prepare-indices first."
            )

        effective_flow = _patch_flow(
            flow_config,
            agent_model,
            monitor_mode=experiment.monitor_mode,
            liveplan_variant=experiment.liveplan_variant,
        )
        (case_dir / "effective_flow.json").write_text(
            json.dumps(effective_flow, indent=2), encoding="utf-8"
        )
        run_data = asyncio.run(
            _run_flow(
                instance,
                repo_path,
                index_dir,
                effective_flow,
                sanitize_transient_files=experiment.monitor_mode == "realtime",
            )
        )
        trajectory = run_data["trajectory"]
        patch = run_data["patch"]
        (case_dir / "trajectory.json").write_text(
            json.dumps(trajectory, indent=2, default=str), encoding="utf-8"
        )
        (case_dir / "patch.diff").write_text(patch, encoding="utf-8")
        decisions, interventions, guidance = _monitor_records(trajectory)
        if experiment.monitor_mode != "vanilla":
            (case_dir / "monitor_decisions.json").write_text(
                json.dumps(decisions, indent=2), encoding="utf-8"
            )
            (case_dir / "monitor_interventions.json").write_text(
                json.dumps(interventions, indent=2), encoding="utf-8"
            )
            (case_dir / "monitor_guidance.json").write_text(
                json.dumps(guidance, indent=2), encoding="utf-8"
            )
        overhead = run_data["overhead"]
        phase_reviews_executed = _operation_event_count(
            overhead, "phase_reviewer"
        )
        phase_policy_resolutions = _operation_event_count(
            overhead, "phase_policy_resolver"
        )
        # `checks` (below) only ever counted `phase_reviewer` -- the
        # monitor_runtime family's own reviewer operation -- so every
        # monitor_baseline (LivePlan/Fixed-Interval/Adaptive Interval) run
        # always reported `checks: 0` regardless of how many refiner calls
        # actually happened (confirmed empirically, Codex review,
        # 2026-09-06). These three are separate, monitor_baseline-specific
        # counters so that gap doesn't require guessing from `checks` again.
        baseline_rule_checks = _operation_event_count(overhead, "rule_check")
        baseline_refiner_calls = _operation_event_count(overhead, "refiner_call")
        baseline_refiner_cooldowns = _operation_event_count(overhead, "refiner_cooldown")
        if experiment.monitor_mode != "vanilla":
            (case_dir / "overhead.json").write_text(
                json.dumps(overhead, indent=2), encoding="utf-8"
            )
        prediction = {
            "model_name_or_path": f"moatless-{agent_model}-{experiment.monitor_mode}",
            "instance_id": instance_id,
            "model_patch": patch,
        }
        (out_dir / "prediction_parts" / f"{instance_id}.json").write_text(
            json.dumps(prediction, indent=2), encoding="utf-8"
        )
        result.update(
            {
                "status": "completed",
                "nodes": run_data["nodes"],
                # A check is a paid reviewer invocation. Decisions produced by
                # a local phase-policy skip remain visible in actions_inspected,
                # but must not inflate reviewer-call frequency or cost metrics.
                "actions_inspected": len(decisions) + len(interventions),
                "checks": phase_reviews_executed,
                "phase_reviews_executed": phase_reviews_executed,
                "phase_policy_resolutions": phase_policy_resolutions,
                "interventions": len(interventions),
                "monitor_guidance": len(guidance),
                "patch_chars": len(patch),
                "patch_nonempty": bool(patch.strip()),
                "finish_reason": run_data["finish_reason"],
                "flow_error": run_data["flow_error"],
                # `status`/`flow_error`/`finish_reason` can all still read as
                # a clean, uneventful run even when the flow actually ended
                # on an empty completion response or ran out of budget
                # without the agent ever calling Finish/Reject -- see
                # `_terminal_status`. Check these, not the fields above, to
                # tell whether a run actually finished on its own terms.
                "agent_exit": run_data["agent_exit"],
                "terminal_node_error": run_data["terminal_node_error"],
                "baseline_rule_checks": baseline_rule_checks,
                "baseline_refiner_calls": baseline_refiner_calls,
                "baseline_refiner_cooldowns": baseline_refiner_cooldowns,
                "removed_transient_paths": run_data["removed_transient_paths"],
                "index": {
                    "source": "persistent_volume",
                    "cache_key": str(index_dir.relative_to(INDEX_ROOT)),
                    "load_seconds": run_data["index_load_seconds"],
                },
                "overhead_summary": overhead.get("summary", {}),
            }
        )
    except Exception as exc:
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        (case_dir / "traceback.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
    (case_dir / "run_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    artifact_volume.commit()
    return result


async def _run_flow(
    instance: dict,
    repo_path: Path,
    index_dir: Path,
    config: dict,
    *,
    sanitize_transient_files: bool,
) -> dict:
    import os

    from moatless.context_data import current_project_id, current_trajectory_id
    from moatless.flow.search_tree import SearchTree
    from moatless.index.code_index import CodeIndex
    from moatless.node import Node
    from moatless.repository.git import GitRepository
    from moatless.runtime.local import SweBenchLocalEnvironment
    from moatless.storage.file_storage import FileStorage
    from moatless.workspace import Workspace

    repository = GitRepository(repo_path=str(repo_path), shadow_mode=True)
    if not _index_is_ready(index_dir, instance):
        raise FileNotFoundError(f"Prebuilt index is missing or invalid: {index_dir}")
    index_started = time.monotonic()
    code_index = CodeIndex.from_persist_dir(
        persist_dir=str(index_dir), file_repo=repository
    )
    index_load_seconds = time.monotonic() - index_started
    storage = FileStorage(
        base_dir=Path("/tmp/moatless_storage") / instance["instance_id"]
    )
    os.environ["SKIP_CONDA_ACTIVATE"] = "true"
    runtime = SweBenchLocalEnvironment(
        repo_path=repo_path,
        swebench_instance=instance,
        storage=storage,
    )
    workspace = Workspace(
        repository=repository,
        code_index=code_index,
        runtime=runtime,
        environment=runtime,
        storage=storage,
    )
    flow = SearchTree.from_dict(config)
    flow.project_id = "rafd_modal"
    flow.trajectory_id = instance["instance_id"]
    current_project_id.set(flow.project_id)
    current_trajectory_id.set(flow.trajectory_id)
    flow._root = Node.create_root(user_message=instance["problem_statement"])
    flow_error = None
    try:
        final_node = await flow.run(workspace=workspace)
    except Exception as exc:
        flow_error = f"{type(exc).__name__}: {exc}"
        final_node = flow.root.get_last_node()
    agent_exit, terminal_node_error = _terminal_status(final_node)
    raw_patch = _extract_patch(flow, final_node)
    if sanitize_transient_files:
        patch, removed_transient_paths = _sanitize_transient_patch(
            raw_patch,
            repo_path=repo_path,
            problem_statement=instance["problem_statement"],
        )
    else:
        patch, removed_transient_paths = raw_patch, []
    trajectory = flow.get_trajectory_data()
    monitor = getattr(flow.agent, "_realtime_phase_monitor", None)
    overhead = monitor.overhead_payload() if monitor is not None else {}
    if monitor is not None:
        trajectory["overhead"] = overhead
    return {
        "trajectory": trajectory,
        "overhead": overhead,
        "patch": patch,
        "removed_transient_paths": removed_transient_paths,
        "nodes": len(flow.root.get_all_nodes()),
        "finish_reason": getattr(flow, "_finish_reason", None),
        "flow_error": flow_error,
        "agent_exit": agent_exit,
        "terminal_node_error": terminal_node_error,
        "index_load_seconds": index_load_seconds,
    }


def _terminal_status(final_node: object) -> tuple[str, str | None]:
    """Classify how the run actually ended, independent of `status`/
    `flow_error`/`finish_reason`/`checks` -- confirmed (Codex review,
    2026-09-06) that those can all still read as a clean "completed" run
    even when the final node's own completion call came back empty (a real
    `CompletionRejectError`, `node.error` set, zero action_steps) or the run
    simply exhausted its iteration/depth budget without the agent ever
    calling Finish/Reject. `node.error` and the terminal action's own
    declared name are the ground truth for what actually happened, so this
    reads those directly instead of trusting the wrapper-level summary
    fields.
    """

    error = getattr(final_node, "error", None)
    action_steps = getattr(final_node, "action_steps", None) or []
    terminal_title = None
    if action_steps:
        last_action = action_steps[-1].action
        model_config = getattr(last_action, "model_config", None) or {}
        terminal_title = model_config.get("title") or type(last_action).__name__
    if terminal_title == "Finish":
        agent_exit = "finish"
    elif terminal_title == "Reject":
        agent_exit = "reject"
    elif error:
        agent_exit = "completion_error"
    else:
        agent_exit = "budget_exceeded_or_unknown"
    return agent_exit, error


def _patch_flow(
    config: dict,
    model: str,
    *,
    monitor_mode: str = "vanilla",
    liveplan_variant: str = "monitor",
) -> dict:
    from monitor_runtime.experiment_config import normalize_monitor_mode

    monitor_mode = normalize_monitor_mode(monitor_mode)
    config = json.loads(json.dumps(config))
    completion = config["agent"]["completion_model"]
    completion.update(
        {
            "model_id": model,
            "model": model,
            "temperature": 1,
            "max_tokens": 8192,
            "message_cache": False,
            "disable_thoughts": True,
        }
    )
    completion.pop("headers", None)
    completion.pop("params", None)
    if monitor_mode == "liveplan" and liveplan_variant in {"monitor", "refiner"}:
        config["agent"]["system_prompt"] = (
            config["agent"]["system_prompt"]
            .replace(
                "Make multiple tool calls in one response:",
                "Make exactly one tool call in each response:",
            )
            .replace(
                "Make all necessary changes in one response:",
                "Make one focused change per response:",
            )
        )
    config["max_iterations"] = MAX_ITERATIONS
    config["max_depth"] = MAX_ITERATIONS
    config["max_cost"] = 2.0
    return config


def _extract_patch(flow, final_node) -> str:
    for node in reversed(flow.root.get_leaf_nodes()):
        if node.file_context:
            patch = node.file_context.generate_git_patch(ignore_tests=True)
            if patch:
                return patch
    if final_node and final_node.file_context:
        return final_node.file_context.generate_git_patch(ignore_tests=True) or ""
    return ""


_TRANSIENT_ROOT_FILE = re.compile(
    r"^(?:repro(?:duce)?|probe|evidence|debug|tmp|temp|check|rafd|run)"
    r"(?:[_-].*)?\.(?:py|sh|txt|log)$",
    re.IGNORECASE,
)
_TRANSIENT_TOOL_FILE = re.compile(
    r"^(?:fix|patch|probe|repro(?:duce)?|run)(?:[_-].*)?\.(?:py|sh|txt|log)$",
    re.IGNORECASE,
)
_TRANSIENT_SUFFIXES = {".bak", ".log", ".orig", ".pickle", ".pkl", ".rej"}
_TRANSIENT_DIRS = {
    "rafd-evidence",
    "tmp",
    ".tmp",
    "build",
    "dist",
    "design_notes",
}


def _sanitize_transient_patch(
    patch: str,
    *,
    repo_path: Path,
    problem_statement: str,
) -> tuple[str, list[str]]:
    """Drop clearly diagnostic, untracked files from Moatless' final patch.

    Moatless keeps a cached ``file_context`` even after Cleanup removes a probe,
    so prompt-only cleanup can still serialize that probe into the prediction.
    This zero-step filter is deliberately narrow: tracked files and paths
    explicitly named by the task are always retained.
    """
    if not patch:
        return patch, []

    starts = list(
        re.finditer(r"(?m)^--- (?:a/)?([^\n]+)\n\+\+\+ (?:b/)?([^\n]+)\n", patch)
    )
    if not starts:
        return patch, []

    task_text = problem_statement.lower()
    kept: list[str] = [patch[: starts[0].start()]]
    removed: list[str] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(patch)
        section = patch[match.start() : end]
        new_path = match.group(2)
        if new_path == "/dev/null":
            new_path = match.group(1)
        normalized = new_path.removeprefix("a/").removeprefix("b/")
        path = Path(normalized)
        if _is_transient_untracked_path(path, repo_path, task_text):
            removed.append(normalized)
        else:
            kept.append(section)
    return "".join(kept), removed


def _is_transient_untracked_path(path: Path, repo_path: Path, task_text: str) -> bool:
    normalized = path.as_posix().lower()
    if normalized in task_text or path.name.lower() in task_text:
        return False
    tracked = (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_path),
                "ls-files",
                "--error-unmatch",
                "--",
                path.as_posix(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    if tracked:
        return False
    lowered_parts = {part.lower() for part in path.parts}
    return (
        bool(lowered_parts & _TRANSIENT_DIRS)
        or path.suffix.lower() in _TRANSIENT_SUFFIXES
        or bool(_TRANSIENT_ROOT_FILE.fullmatch(path.name))
        or (
            bool(path.parts)
            and path.parts[0].lower() == "tools"
            and bool(_TRANSIENT_TOOL_FILE.fullmatch(path.name))
        )
        or path.name.lower() in {"submission_artifact.txt", "submission-artifact.txt"}
    )


def _monitor_records(
    trajectory: object,
) -> tuple[list[dict], list[dict], list[dict]]:
    decisions: list[dict] = []
    interventions: list[dict] = []
    guidance: list[dict] = []
    recorded_action_groups: set[str] = set()

    def walk(value: object, path: str = "root") -> None:
        if isinstance(value, dict):
            monitor = value.get("phase_monitor")
            if isinstance(monitor, dict):
                action_group_step_id = monitor.get("action_group_step_id")
                if action_group_step_id is None:
                    decisions.append({"path": path, **monitor})
                else:
                    group_key = str(action_group_step_id)
                    if group_key not in recorded_action_groups:
                        recorded_action_groups.add(group_key)
                        decisions.append({"path": path, **monitor})
            events = value.get("monitor_events")
            if isinstance(events, list):
                for i, event in enumerate(events):
                    if not isinstance(event, dict):
                        continue
                    record = {"path": f"{path}.monitor_events[{i}]", **event}
                    if event.get("event_type") in {
                        "monitor_advisory",
                        "monitor_guidance",
                    }:
                        guidance.append(record)
                    else:
                        interventions.append(record)
            for key, child in value.items():
                if key not in {"phase_monitor", "monitor_events"}:
                    walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(trajectory)
    # Compatibility for trajectories written before monitor fields were marked
    # as explicitly set on Pydantic models. Their intervention prompt remains
    # durable in user_message even though monitor_events was omitted.
    if not interventions:

        def recover(value: object, path: str = "root") -> None:
            if isinstance(value, dict):
                message = value.get("user_message")
                marker = "A realtime monitor reviewed your previous phase: `"
                if isinstance(message, str) and message.startswith(marker):
                    phase_match = re.search(r"previous phase: `([^`]+)`", message)
                    capabilities = re.findall(
                        r"^- (CAP-\d+)\s*$", message, flags=re.MULTILINE
                    )
                    interventions.append(
                        {
                            "path": f"{path}.user_message",
                            "event_type": "monitor_intervention",
                            "intervention_for_phase": phase_match.group(1)
                            if phase_match
                            else None,
                            "selected_capabilities": capabilities,
                            "message": message,
                            "recovered_from_prompt": True,
                        }
                    )
                for key, child in value.items():
                    recover(child, f"{path}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    recover(child, f"{path}[{index}]")

        recover(trajectory)
    return decisions, interventions, guidance


def _operation_event_count(overhead: object, operation: str) -> int:
    """Return actual recorded calls for one overhead-ledger operation."""
    if not isinstance(overhead, dict):
        return 0
    summary = overhead.get("summary")
    if not isinstance(summary, dict):
        return 0
    by_operation = summary.get("by_operation")
    if not isinstance(by_operation, dict):
        return 0
    operation_summary = by_operation.get(operation)
    if not isinstance(operation_summary, dict):
        return 0
    return int(operation_summary.get("event_count") or 0)


def _run(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def _load_flow() -> dict:
    """Load the checked-in public flow; no Git history is needed to launch."""
    return json.loads(FLOW_PATH.read_text(encoding="utf-8"))


@app.function(
    image=image,
    volumes={"/artifacts": artifact_volume},
    timeout=10 * 60,
    retries=0,
)
def finalize_phase_policy_remote(
    run_id: str,
    phase_policy_enabled: bool,
    phase_policy_startup_mode: str,
    experiment_payload: dict[str, object] | None = None,
) -> dict[str, object] | None:
    """Merge phase entries written by parallel Moatless workers."""

    import sys

    sys.path.insert(0, "/root")
    artifact_volume.reload()
    from monitor_runtime.runners.support import (
        materialize_phase_policy_snapshot,
    )

    enabled = phase_policy_enabled
    startup_mode = phase_policy_startup_mode
    if experiment_payload is not None:
        enabled = (
            experiment_payload["monitor_mode"] == "realtime"
            and "phase_policy" not in experiment_payload["ablations"]
        )
        startup_mode = str(experiment_payload["phase_policy_startup_mode"])

    result = materialize_phase_policy_snapshot(
        run_id=run_id,
        agent_variant="moatless",
        startup_mode=startup_mode,
        enabled=enabled,
    )
    artifact_volume.commit()
    return result


def _rate_limit_failure(result: object) -> bool:
    """Return whether a remote result represents a retryable API rate limit."""

    if not isinstance(result, dict):
        return False
    text = " ".join(
        str(result.get(key) or "") for key in ("error", "flow_error")
    ).lower()
    return any(marker in text for marker in ("429", "rate limit", "too many requests"))


def _run_cases_continuously(
    *,
    run_id: str,
    case_list: list[str],
    flow: dict,
    agent_model: str,
    selector_model: str,
    reviewer_model: str,
    phase_policy_enabled: bool,
    phase_policy_startup_mode: str,
    enable_monitor: bool,
    experiment_payload: dict[str, object],
    dataset: str,
    parallelism: int,
    rate_limit_parallelism: int,
    spawn_interval_seconds: float,
) -> list[dict]:
    """Run without batch barriers, optionally spacing out worker starts."""

    pending = deque(case_list)
    active: dict[str, tuple[str, object]] = {}
    attempts: dict[str, int] = {}
    results: dict[str, dict] = {}
    active_parallelism = parallelism
    rate_limit_parallelism = max(1, min(rate_limit_parallelism, parallelism))
    spawn_interval_seconds = max(0.0, spawn_interval_seconds)
    next_spawn_at = 0.0

    def spawn(instance_id: str) -> None:
        call = run_case_remote.spawn(
            run_id,
            instance_id,
            flow,
            agent_model,
            selector_model,
            reviewer_model,
            phase_policy_enabled,
            phase_policy_startup_mode,
            enable_monitor,
            experiment_payload,
            dataset,
        )
        active[call.object_id] = (instance_id, call)

    while pending or active:
        while pending and len(active) < active_parallelism:
            now = time.monotonic()
            if now < next_spawn_at:
                break
            spawn(pending.popleft())
            next_spawn_at = now + spawn_interval_seconds
            if spawn_interval_seconds:
                break

        completed_any = False
        for call_id, (instance_id, call) in list(active.items()):
            try:
                item = call.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:
                item = {
                    "instance_id": instance_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            completed_any = True
            del active[call_id]
            if _rate_limit_failure(item):
                attempts[instance_id] = attempts.get(instance_id, 0) + 1
                if active_parallelism > rate_limit_parallelism:
                    active_parallelism = rate_limit_parallelism
                    print(
                        json.dumps(
                            {
                                "status": "parallelism_reduced",
                                "parallelism": active_parallelism,
                                "reason": "rate_limit",
                                "instance_id": instance_id,
                            }
                        ),
                        flush=True,
                    )
                if attempts[instance_id] < 3:
                    pending.append(instance_id)
                    print(
                        json.dumps(
                            {
                                "status": "retrying_rate_limited_case",
                                "instance_id": instance_id,
                                "attempt": attempts[instance_id] + 1,
                            }
                        ),
                        flush=True,
                    )
                    continue
            results[instance_id] = item
            print(json.dumps(item), flush=True)

        if not completed_any and (active or pending):
            wait_seconds = 2.0
            if pending and len(active) < active_parallelism:
                wait_seconds = min(
                    wait_seconds, max(0.05, next_spawn_at - time.monotonic())
                )
            time.sleep(wait_seconds)

    return [results[instance_id] for instance_id in case_list]


@app.local_entrypoint()
def main(
    run_id: str = "",
    cases: str = "psf__requests-2317,sympy__sympy-17630",
    agent_model: str = "gpt-5-mini",
    selector_model: str = "gpt-5-mini",
    reviewer_model: str = "gpt-5.2",
    parallelism: int = 6,
    continuous_map: bool = False,
    rate_limit_parallelism: int = 8,
    spawn_interval_seconds: float = 0.0,
    enable_monitor: bool = True,
    phase_policy_enabled: bool = True,
    phase_policy_startup_mode: str = "cold",
    prepare_repositories: bool = False,
    prepare_indices: bool = False,
    repository_retry_delay_seconds: float = 180.0,
    monitor_mode: str = "",
    phase_policy_model: str = "",
    liveplan_variant: str = "monitor",
    liveplan_refiner_model: str = "",
    liveplan_refiner_config: str = "",
    ablations: str = "",
    dataset: str = "verified",
) -> None:
    resolved_run_id = run_id or f"cross_agent_{time.strftime('%Y%m%d_%H%M%S')}"
    case_list = [case.strip() for case in cases.split(",") if case.strip()]
    if parallelism < 1:
        raise ValueError("parallelism must be at least 1")
    if phase_policy_startup_mode not in {"cold", "warm"}:
        raise ValueError("phase_policy_startup_mode must be cold or warm")
    if dataset not in DATASET_MAPPING:
        raise ValueError("dataset must be lite or verified")
    if repository_retry_delay_seconds < 0:
        raise ValueError("repository_retry_delay_seconds must be non-negative")
    if prepare_repositories:
        from datasets import load_dataset

        rows = list(load_dataset(DATASET_MAPPING[dataset], split="test"))
        commits_by_repo: dict[str, set[str]] = {}
        for row in rows:
            commits_by_repo.setdefault(row["repo"], set()).add(row["base_commit"])

        prepared_repositories: list[dict[str, object]] = []
        for repo in sorted(commits_by_repo):
            required_commits = sorted(commits_by_repo[repo])
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    item = prepare_repository_remote.remote(repo, required_commits)
                except Exception as exc:
                    last_error = exc
                    print(
                        json.dumps(
                            {
                                "repo": repo,
                                "status": "retrying",
                                "attempt": attempt,
                                "error": f"{type(exc).__name__}: {exc}",
                                "retry_delay_seconds": repository_retry_delay_seconds,
                            }
                        ),
                        flush=True,
                    )
                    if attempt < 3:
                        time.sleep(repository_retry_delay_seconds)
                    continue
                prepared_repositories.append(item)
                print(json.dumps(item), flush=True)
                break
            else:
                assert last_error is not None
                raise RuntimeError(
                    f"Failed to cache {repo} after 3 attempts"
                ) from last_error

        print(
            json.dumps(
                {
                    "mode": "prepare_repositories",
                    "dataset": dataset,
                    "repo_volume": REPO_VOLUME_NAME,
                    "instances": len(rows),
                    "repositories": len(commits_by_repo),
                    "required_commits": sum(
                        len(commits) for commits in commits_by_repo.values()
                    ),
                    "results": prepared_repositories,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    flow = _load_flow()
    experiment = runner_experiment_config(
        monitor_mode=monitor_mode,
        agent_model=agent_model,
        selector_model=selector_model,
        reviewer_model=reviewer_model,
        phase_policy_model=phase_policy_model,
        liveplan_variant=liveplan_variant,
        liveplan_refiner_model=liveplan_refiner_model,
        liveplan_refiner_config=liveplan_refiner_config,
        ablations=ablations,
        phase_policy_startup_mode=phase_policy_startup_mode,
        legacy_enable_monitor=enable_monitor,
        legacy_phase_policy_enabled=phase_policy_enabled,
    )
    experiment_payload = experiment.as_dict()
    if prepare_indices:
        from monitor_runtime.settings import get_settings

        if not get_settings().values.get("VOYAGE_API_KEY"):
            raise RuntimeError("VOYAGE_API_KEY is required to prepare Moatless indices")
        prepared: list[dict[str, object]] = []
        pending = list(case_list)
        attempts: dict[str, int] = {}
        active_parallelism = parallelism
        while pending:
            batch = pending[:active_parallelism]
            del pending[:active_parallelism]
            batch_results = list(
                prepare_index_remote.map(
                    batch,
                    [dataset] * len(batch),
                    return_exceptions=True,
                )
            )
            for instance_id, item in zip(batch, batch_results):
                if not isinstance(item, Exception):
                    prepared.append(item)
                    print(json.dumps(item), flush=True)
                    continue

                attempts[instance_id] = attempts.get(instance_id, 0) + 1
                error = f"{type(item).__name__}: {item}"
                lowered = error.lower()
                is_rate_limit = any(
                    marker in lowered
                    for marker in ("429", "rate limit", "too many requests")
                )
                is_terminal_credential_error = any(
                    marker in lowered
                    for marker in (
                        "authentication",
                        "api key",
                        "insufficient credit",
                        "insufficient quota",
                        "quota exhausted",
                        "payment required",
                        "402",
                    )
                )
                if is_terminal_credential_error:
                    raise RuntimeError(
                        f"Stopping index preparation after credential/quota failure: {error}"
                    ) from item
                if is_rate_limit and active_parallelism > 1:
                    active_parallelism -= 1
                    print(
                        json.dumps(
                            {
                                "status": "parallelism_reduced",
                                "parallelism": active_parallelism,
                                "reason": error,
                            }
                        ),
                        flush=True,
                    )
                if attempts[instance_id] < 3:
                    pending.append(instance_id)
                    continue
                failure = {
                    "instance_id": instance_id,
                    "status": "error",
                    "error": error,
                    "attempts": attempts[instance_id],
                }
                prepared.append(failure)
                print(json.dumps(failure), flush=True)
        print(
            json.dumps(
                {
                    "mode": "prepare_indices",
                    "index_volume": INDEX_VOLUME_NAME,
                    "results": prepared,
                },
                indent=2,
            )
        )
        return
    validate_experiment_credentials(experiment, require_voyage=True)
    results = []
    if continuous_map:
        results = _run_cases_continuously(
            run_id=resolved_run_id,
            case_list=case_list,
            flow=flow,
            agent_model=agent_model,
            selector_model=selector_model,
            reviewer_model=reviewer_model,
            phase_policy_enabled=phase_policy_enabled,
            phase_policy_startup_mode=phase_policy_startup_mode,
            enable_monitor=enable_monitor,
            experiment_payload=experiment_payload,
            dataset=dataset,
            parallelism=parallelism,
            rate_limit_parallelism=rate_limit_parallelism,
            spawn_interval_seconds=spawn_interval_seconds,
        )
    else:
        for start in range(0, len(case_list), parallelism):
            batch = case_list[start : start + parallelism]
            results.extend(
                run_case_remote.map(
                    [resolved_run_id] * len(batch),
                    batch,
                    [flow] * len(batch),
                    [agent_model] * len(batch),
                    [selector_model] * len(batch),
                    [reviewer_model] * len(batch),
                    [phase_policy_enabled] * len(batch),
                    [phase_policy_startup_mode] * len(batch),
                    [enable_monitor] * len(batch),
                    [experiment_payload] * len(batch),
                    [dataset] * len(batch),
                )
            )
    phase_policy_artifacts = finalize_phase_policy_remote.remote(
        resolved_run_id,
        phase_policy_enabled and enable_monitor,
        phase_policy_startup_mode,
        experiment_payload,
    )
    print(
        json.dumps(
            {
                "run_id": resolved_run_id,
                "experiment": experiment_payload,
                "phase_policy": phase_policy_artifacts,
                "results": results,
            },
            indent=2,
        )
    )

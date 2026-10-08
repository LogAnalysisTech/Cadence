#!/usr/bin/env python3
"""
Run moatless agent on a single SWE-bench instance using a pre-built FAISS index.
Saves trajectory.json, patch.diff, and eval.json to runs/<instance_id>_<timestamp>/.

Usage:
    python run_single_instance.py [--instance INSTANCE_ID]
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
MODEL    = "gpt-5-mini"
MAX_COST = 2.0        # USD hard cap
MAX_ITER = 50

SCRIPT_DIR = Path(__file__).parent.resolve()
FLOW_CFG   = SCRIPT_DIR / ".moatless/flows/swebench_claude.json"
REPO_BASE  = Path("/private/tmp/repos")

# API keys – read from config.py two levels up
_cfg_path = SCRIPT_DIR.parent.parent / "config.py"
_cfg      = {}
exec(_cfg_path.read_text(), _cfg)
os.environ.setdefault("OPENAI_API_KEY", _cfg["API_KEY"])
os.environ.setdefault("VOYAGE_API_KEY", _cfg["VOYAGE_API_KEY"])
os.environ.setdefault("MOATLESS_DIR",   str(SCRIPT_DIR))

import litellm
litellm.drop_params = True   # silently drop unsupported params (e.g. message_cache)
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_single")


def _patch_flow_config(cfg: dict, model: str) -> dict:
    """Replace Claude model with an OpenAI model; strip runtime-dependent actions."""
    cfg = copy.deepcopy(cfg)
    cm = cfg["agent"]["completion_model"]
    cm["model_id"] = model
    cm["model"]    = model
    cm.pop("params",  None)
    cm.pop("headers", None)
    cm["temperature"]      = 1    # gpt-5 models require temperature=1
    cm["max_tokens"]       = 4096
    cm["message_cache"]    = False
    cm["disable_thoughts"] = True
    cfg["max_cost"]       = MAX_COST
    cfg["max_iterations"] = MAX_ITER
    _remove = {
        "moatless.actions.run_tests.RunTests",
        "moatless.actions.run_python_script.RunPythonScript",
        "moatless.actions.create_python_file.CreatePythonFile",
    }
    cfg["agent"]["actions"] = [
        a for a in cfg["agent"]["actions"]
        if a.get("action_class") not in _remove
    ]
    return cfg


def _extract_patch(flow, final_node) -> str:
    """Extract patch using file_context (correct approach for MCTS shadow mode)."""
    # Try best leaf node first (official moatless approach)
    try:
        leaf_nodes = flow.root.get_leaf_nodes()
        for node in leaf_nodes:
            if node.file_context:
                patch = node.file_context.generate_git_patch(ignore_tests=True)
                if patch:
                    return patch
    except Exception as e:
        logger.warning(f"file_context patch extraction failed: {e}")

    # Fallback: collect artifact_changes from trajectory along best path
    try:
        traj = flow.get_trajectory_data()
        nodes_by_id = {n["node_id"]: n for n in traj.get("nodes", [])}

        # Walk from final_node up to root to find the path
        path_ids = []
        nid = final_node.node_id
        while nid is not None:
            path_ids.append(nid)
            parent_id = nodes_by_id.get(nid, {}).get("parent_id")
            nid = parent_id
        path_ids.reverse()

        # Collect artifact diffs in order, keep last diff per file
        file_diffs: dict[str, str] = {}
        for nid in path_ids:
            node_data = nodes_by_id.get(nid, {})
            for step in node_data.get("action_steps", []):
                for ac in step.get("observation", {}).get("artifact_changes", []):
                    fpath = ac.get("artifact_id", "")
                    diff  = ac.get("properties", {}).get("diff", "")
                    if diff:
                        file_diffs[fpath] = diff

        if file_diffs:
            return "\n".join(file_diffs.values())
    except Exception as e:
        logger.warning(f"trajectory patch extraction failed: {e}")

    return ""


async def main(instance_id: str):
    output_dir = SCRIPT_DIR / "runs" / f"{instance_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    index_dir  = SCRIPT_DIR / "index_store" / instance_id

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Instance: {instance_id}  Output: {output_dir}")

    # ── 1. Load instance metadata ──────────────────────────────────────────
    from moatless.evaluation.utils import get_moatless_instance
    instance    = get_moatless_instance(instance_id)
    problem     = instance["problem_statement"]
    repo_name   = instance["repo"]
    base_commit = instance["base_commit"]
    logger.info(f"Problem (first 120): {problem[:120]}")

    # ── 2. Clone repo (shared base clone, per-instance worktree) ─────────
    import subprocess
    import time

    # Share one base clone per upstream repo to avoid re-cloning for each instance
    repo_slug = repo_name.replace("/", "__")
    base_clone = REPO_BASE / f"base_{repo_slug}"
    repo_path  = REPO_BASE / f"swe-bench_{instance_id}"

    def _run_git(args, **kwargs):
        return subprocess.run(args, check=True, **kwargs)

    def _clone_base():
        """Clone base repo (blobless) with retry."""
        for attempt in range(1, 5):
            try:
                shutil.rmtree(base_clone, ignore_errors=True)
                base_clone.mkdir(parents=True, exist_ok=True)
                _run_git(["git", "clone", "--filter=blob:none", "--no-checkout",
                          f"https://github.com/{repo_name}.git", str(base_clone)])
                return
            except subprocess.CalledProcessError:
                if attempt == 4:
                    raise
                wait = 30 * attempt
                logger.warning(f"Clone failed (attempt {attempt}/4), retrying in {wait}s …")
                time.sleep(wait)

    if not (base_clone / ".git").exists():
        logger.info(f"Base clone {repo_name} → {base_clone}")
        _clone_base()

    # Fetch the specific commit (no --depth=1: conflicts with blobless promisor remote)
    for attempt in range(1, 5):
        try:
            _run_git(["git", "fetch", "origin", base_commit],
                     cwd=str(base_clone))
            break
        except subprocess.CalledProcessError:
            if attempt == 4:
                raise
            wait = 30 * attempt
            logger.warning(f"Fetch failed (attempt {attempt}/4), retrying in {wait}s …")
            if attempt == 3:
                logger.warning("Re-cloning base after repeated fetch failures")
                _clone_base()
            time.sleep(wait)

    # Create a fresh per-instance copy from the base clone for shadow-mode isolation
    if repo_path.exists():
        shutil.rmtree(repo_path)
    shutil.copytree(str(base_clone), str(repo_path), symlinks=True)
    _run_git(["git", "checkout", base_commit], cwd=str(repo_path))
    logger.info(f"Repo ready: {repo_path}")

    # ── 3. GitRepository + index + workspace ──────────────────────────────
    from moatless.repository.git import GitRepository
    from moatless.index.code_index import CodeIndex
    from moatless.environment.local import LocalBashEnvironment
    from moatless.workspace import Workspace

    repository  = GitRepository(repo_path=str(repo_path), shadow_mode=True)
    code_index  = CodeIndex.from_persist_dir(persist_dir=str(index_dir), file_repo=repository)
    environment = LocalBashEnvironment(cwd=str(repo_path))
    workspace   = Workspace(repository=repository, code_index=code_index, environment=environment)
    logger.info(f"Index loaded: {index_dir}")

    # ── 4. Build flow ──────────────────────────────────────────────────────
    from moatless.flow.search_tree import SearchTree
    from moatless.node import Node
    from moatless.context_data import current_project_id, current_trajectory_id

    raw_cfg = json.loads(FLOW_CFG.read_text())
    patched = _patch_flow_config(raw_cfg, MODEL)
    (output_dir / "flow.json").write_text(json.dumps(patched, indent=2))

    flow = SearchTree.from_dict(patched)
    flow.project_id    = "local_eval"
    flow.trajectory_id = instance_id
    current_project_id.set(flow.project_id)
    current_trajectory_id.set(flow.trajectory_id)

    root = Node.create_root(user_message=problem)
    flow._root = root

    # ── 5. Run ────────────────────────────────────────────────────────────
    logger.info("Starting agent flow …")
    flow_error = None
    try:
        final_node = await flow.run(workspace=workspace)
        logger.info(f"Flow finished. Final node: {final_node.node_id}")
    except Exception as exc:
        logger.exception(f"Flow error: {exc}")
        flow_error = str(exc)
        final_node = flow.root.get_last_node()

    # ── 6. Save trajectory ────────────────────────────────────────────────
    traj_data = flow.get_trajectory_data()
    traj_path = output_dir / "trajectory.json"
    traj_path.write_text(json.dumps(traj_data, indent=2, default=str))
    logger.info(f"Trajectory saved: {traj_path}")

    # ── 7. Extract + save patch ───────────────────────────────────────────
    patch = _extract_patch(flow, final_node)
    patch_path = output_dir / "patch.diff"
    patch_path.write_text(patch)
    logger.info(f"Patch saved: {patch_path} ({len(patch)} bytes)")

    # ── 8. Eval metrics (lightweight, no Docker) ──────────────────────────
    nodes = flow.root.get_all_nodes()
    actions_taken = [n.action.name for n in nodes if n.action]
    rewards = [n.value for n in nodes if n.value is not None and n.value != 0]
    max_reward = max(rewards) if rewards else None
    terminal_action = final_node.action.name if final_node.action else None

    # Detect finish reason from search_tree state
    finish_reason = getattr(flow, "_finish_reason", None) or (
        "error" if flow_error else "unknown"
    )

    eval_data = {
        "instance_id":     instance_id,
        "model":           MODEL,
        "patch_size":      len(patch),
        "patch_non_empty": len(patch) > 0,
        "max_reward":      max_reward,
        "terminal_action": terminal_action,
        "num_nodes":       len(nodes),
        "actions":         actions_taken,
        "finish_reason":   finish_reason,
        "flow_error":      flow_error,
        "status":          "error" if flow_error else "completed",
    }
    eval_path = output_dir / "eval.json"
    eval_path.write_text(json.dumps(eval_data, indent=2))
    logger.info(f"Eval saved:  {eval_path}")

    # ── 9. Print summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Instance:   {instance_id}")
    print(f"Model:      {MODEL}")
    print(f"Nodes:      {len(nodes)}")
    print(f"Actions:    {actions_taken}")
    print(f"Max reward: {max_reward}")
    print(f"Patch size: {len(patch)} chars")
    print(f"Output:     {output_dir}")
    print("=" * 60)

    if patch:
        print("\n─── PATCH ───")
        print(patch[:3000])
        if len(patch) > 3000:
            print(f"... ({len(patch) - 3000} more chars, see patch.diff)")

    return eval_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="django__django-11099",
                        help="SWE-bench instance ID")
    args = parser.parse_args()
    asyncio.run(main(args.instance))

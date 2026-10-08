#!/usr/bin/env python3
"""
Batch runner: run moatless agent on all instances that have a built FAISS index.
After each run, deletes the cloned repo to save disk space.
Skips instances that already have an eval.json in runs/.

Usage:
    python run_batch.py [--workers N]   # N parallel workers (default 1)
    python run_batch.py --list          # just list pending instances
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
INDEX_DIR  = SCRIPT_DIR / "index_store"
RUNS_DIR   = SCRIPT_DIR / "runs"
REPO_BASE  = Path("/private/tmp/repos")
LOG_PATH   = Path("/tmp/moatless_batch.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="a"),
    ],
)
logger = logging.getLogger("batch")


def get_all_instances() -> list[str]:
    """Return all instance IDs that have a built index (contain a docstore.json)."""
    instances = []
    for d in sorted(INDEX_DIR.iterdir()):
        if d.is_dir() and "__" in d.name and (d / "docstore.json").exists():
            instances.append(d.name)
    return instances


def already_done(instance_id: str) -> bool:
    """Return True if this instance has a completed eval.json in any run dir."""
    for run_dir in RUNS_DIR.glob(f"{instance_id}_*"):
        eval_file = run_dir / "eval.json"
        if eval_file.exists():
            try:
                data = json.loads(eval_file.read_text())
                if data.get("status") == "completed":
                    return True
            except Exception:
                pass
    return False


def delete_repo(instance_id: str):
    """Delete only the per-instance copy; keep shared base clone for reuse."""
    repo_path = REPO_BASE / f"swe-bench_{instance_id}"
    if repo_path.exists():
        shutil.rmtree(repo_path)
        logger.info(f"[{instance_id}] Deleted instance repo: {repo_path}")


INSTANCE_TIMEOUT = 25 * 60  # 25 minutes — kill and skip if exceeded


async def run_instance(instance_id: str, semaphore: asyncio.Semaphore) -> dict:
    """Run a single instance in a subprocess and return eval result."""
    async with semaphore:
        logger.info(f"[{instance_id}] Starting ...")
        start = datetime.now()

        venv_python = SCRIPT_DIR / ".venv/bin/python"
        python_exec = str(venv_python) if venv_python.exists() else sys.executable

        log_file = Path(f"/tmp/moatless_{instance_id}.log")
        proc = await asyncio.create_subprocess_exec(
            python_exec, str(SCRIPT_DIR / "run_single_instance.py"),
            "--instance", instance_id,
            stdout=log_file.open("w"),
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(SCRIPT_DIR),
        )
        timed_out = False
        try:
            returncode = await asyncio.wait_for(proc.wait(), timeout=INSTANCE_TIMEOUT)
        except asyncio.TimeoutError:
            timed_out = True
            proc.kill()
            await proc.wait()
            returncode = -1
            logger.warning(f"[{instance_id}] Timed out after {INSTANCE_TIMEOUT//60}min — killed, moving on")

        elapsed = (datetime.now() - start).total_seconds()

        # Read eval result
        eval_data = None
        for run_dir in sorted(RUNS_DIR.glob(f"{instance_id}_*"), reverse=True):
            eval_file = run_dir / "eval.json"
            if eval_file.exists():
                try:
                    eval_data = json.loads(eval_file.read_text())
                    eval_data["elapsed_sec"] = round(elapsed)
                    eval_file.write_text(json.dumps(eval_data, indent=2))
                except Exception:
                    pass
                break

        if eval_data is None:
            eval_data = {
                "instance_id": instance_id,
                "status": "timeout" if timed_out else "runner_error",
                "returncode": returncode,
                "elapsed_sec": round(elapsed),
            }
            logger.error(f"[{instance_id}] No eval.json produced. rc={returncode}. Log: {log_file}")

        status = eval_data.get("status", "?")
        patch_size = eval_data.get("patch_size", 0)
        logger.info(
            f"[{instance_id}] Done in {elapsed:.0f}s — status={status}, "
            f"patch={patch_size}b, reward={eval_data.get('max_reward')}"
        )

        # Delete repo after run to save disk
        delete_repo(instance_id)

        return eval_data


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel instances (default 1)")
    parser.add_argument("--list", action="store_true",
                        help="List pending instances and exit")
    args = parser.parse_args()

    all_instances = get_all_instances()
    pending = [i for i in all_instances if not already_done(i)]

    logger.info(f"Instances with index: {len(all_instances)}")
    logger.info(f"Already completed:    {len(all_instances) - len(pending)}")
    logger.info(f"Pending:              {len(pending)}")

    if args.list:
        for i in pending:
            print(i)
        return

    if not pending:
        logger.info("All done!")
        return

    semaphore = asyncio.Semaphore(args.workers)
    results = []

    for i, instance_id in enumerate(pending, 1):
        logger.info(f"─── [{i}/{len(pending)}] {instance_id} ───")
        result = await run_instance(instance_id, semaphore)
        results.append(result)

        # Running summary
        done_count = sum(1 for r in results if r.get("status") == "completed")
        patch_count = sum(1 for r in results if r.get("patch_non_empty"))
        logger.info(
            f"Progress: {i}/{len(pending)} done, "
            f"{done_count} completed, {patch_count} with patch"
        )

    # Final summary
    summary_path = RUNS_DIR / "batch_summary.json"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Batch summary saved: {summary_path}")

    completed   = [r for r in results if r.get("status") == "completed"]
    with_patch  = [r for r in completed if r.get("patch_non_empty")]
    errors      = [r for r in results if r.get("status") != "completed"]

    print("\n" + "=" * 60)
    print(f"BATCH COMPLETE")
    print(f"  Total:      {len(results)}")
    print(f"  Completed:  {len(completed)}")
    print(f"  With patch: {len(with_patch)}")
    print(f"  Errors:     {len(errors)}")
    print(f"  Summary:    {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

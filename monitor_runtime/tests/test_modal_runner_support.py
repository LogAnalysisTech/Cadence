from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitor_runtime.runners.support import (
    configure_monitor,
    extract_monitor_artifacts,
    materialize_phase_policy_snapshot,
    summarize_instance,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def restore_monitor_environment():
    import os

    original = dict(os.environ)
    yield
    prefixes = ("REALTIME_MONITOR_", "MONITOR_BASELINE_")
    for key in list(os.environ):
        if key.startswith(prefixes) or key == "MONITOR_EXPERIMENT_CONFIG":
            os.environ.pop(key, None)
    for key, value in original.items():
        if key.startswith(prefixes) or key == "MONITOR_EXPERIMENT_CONFIG":
            os.environ[key] = value


def test_phase_reviews_executed_does_not_double_count_interventions(tmp_path) -> None:
    """A review that produces an intervention message must count once, not twice.

    The intervention message is a downstream artifact of the same review that
    already appears as a decision entry with ``selected_capabilities`` set --
    adding ``len(interventions)`` on top double-counted every intervening review.
    """

    traj_path = tmp_path / "case.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "phase": "verifying",
                            "phase_monitor": {
                                "selected_capabilities": ["CAP-10"],
                                "decision": "intervene",
                            },
                        },
                    },
                    {
                        "role": "user",
                        "content": "A realtime monitor reviewed your previous phase.",
                        "extra": {"message_type": "monitor_intervention"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_instance(traj_path)

    assert summary["phase_reviews_executed"] == 1
    assert summary["monitor_interventions"] == 1


def test_nonblocking_guidance_is_not_counted_as_intervention(tmp_path) -> None:
    traj_path = tmp_path / "case.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Inspect the latest observation.",
                        "extra": {
                            "message_type": "monitor_guidance",
                            "metadata": {"executed_step_id": "step-1"},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_instance(traj_path)

    assert summary["monitor_interventions"] == 0
    assert summary["monitor_guidance"] == 1


def test_summary_distinguishes_native_autosubmission(tmp_path) -> None:
    traj_path = tmp_path / "case.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "info": {
                    "exit_status": "submitted (step_limit)",
                    "submission": "diff --git a/foo.py b/foo.py",
                },
                "messages": [],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_instance(traj_path)

    assert summary["submission_path"] == "autosubmission"


def test_vanilla_artifacts_include_telemetry_but_no_monitor_views(tmp_path) -> None:
    traj_path = tmp_path / "case.traj.json"
    instance_dir = tmp_path / "case"
    traj_path.write_text(
        json.dumps(
            {
                "info": {
                    "overhead": {
                        "schema_version": "realtime-overhead-1.0",
                        "events": [{"component": "agent", "operation": "agent_generation"}],
                    }
                },
                "messages": [],
            }
        ),
        encoding="utf-8",
    )

    extract_monitor_artifacts(
        "case",
        traj_path,
        instance_dir,
        include_monitor_views=False,
    )

    assert (instance_dir / "overhead.json").exists()
    assert not (instance_dir / "monitor_decisions.json").exists()
    assert not (instance_dir / "monitor_interventions.json").exists()
    assert not (instance_dir / "monitor_timeline.json").exists()
    assert "monitor_decisions" not in json.loads(
        (instance_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )


def test_overhead_is_authoritative_for_reviews_removed_by_intervention(tmp_path) -> None:
    traj_path = tmp_path / "case.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "info": {
                    "overhead": {
                        "summary": {
                            "by_operation": {
                                "phase_reviewer": {"event_count": 3},
                                "phase_policy_resolver": {"event_count": 2},
                            }
                        }
                    }
                },
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "phase_monitor": {"selected_capabilities": ["CAP-01"]}
                        },
                    },
                    {
                        "role": "user",
                        "extra": {"message_type": "monitor_intervention"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_instance(traj_path)

    assert summary["phase_reviews_executed"] == 3
    assert summary["phase_policy_resolutions"] == 2

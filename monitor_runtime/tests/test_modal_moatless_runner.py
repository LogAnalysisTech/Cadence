from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from monitor_runtime.runners.moatless import (
    DATASET_MAPPING,
    INDEX_MANIFEST_NAME,
    INDEX_SCHEMA_VERSION,
    REPO_CACHE_VERSION,
    _clone_instance_repository,
    _repo_cache_is_ready,
    _repo_manifest_path,
    _repo_mirror_dir,
    _index_dir,
    _index_is_ready,
    _index_metadata,
    _extract_patch,
    _monitor_records,
    _operation_event_count,
    _patch_flow,
    _rate_limit_failure,
    _sanitize_transient_patch,
    _terminal_status,
)


def test_moatless_runner_supports_both_swebench_datasets() -> None:
    assert DATASET_MAPPING == {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
    }


def test_rate_limit_detection_checks_remote_and_flow_errors() -> None:
    assert _rate_limit_failure({"status": "error", "error": "HTTP 429"})
    assert _rate_limit_failure(
        {"status": "completed", "flow_error": "RateLimitError: too many requests"}
    )
    assert not _rate_limit_failure(
        {"status": "completed", "flow_error": None, "patch_chars": 12}
    )


def _instance() -> dict[str, str]:
    return {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc123",
    }


def test_index_path_is_versioned_by_instance_and_commit() -> None:
    path = _index_dir(_instance())

    assert path.parts[-3:] == (
        INDEX_SCHEMA_VERSION,
        "django__django-12345",
        "abc123",
    )


def test_index_requires_matching_manifest_and_payload(tmp_path: Path) -> None:
    instance = _instance()
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    assert not _index_is_ready(index_dir, instance)

    (index_dir / INDEX_MANIFEST_NAME).write_text(
        json.dumps(_index_metadata(instance)), encoding="utf-8"
    )
    assert not _index_is_ready(index_dir, instance)

    (index_dir / "docstore.json").write_text("{}", encoding="utf-8")
    assert _index_is_ready(index_dir, instance)

    wrong = _index_metadata(instance)
    wrong["base_commit"] = "different"
    (index_dir / INDEX_MANIFEST_NAME).write_text(json.dumps(wrong), encoding="utf-8")
    assert not _index_is_ready(index_dir, instance)


def test_repository_cache_clones_exact_commit_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from monitor_runtime.runners import moatless as runner

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(
        ["git", "config", "user.email", "cache-test@example.invalid"],
        cwd=source,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Cache Test"], cwd=source, check=True)
    (source / "payload.txt").write_text("cached\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=source, check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path / "repo_store")
    repo = "example/project"
    mirror = _repo_mirror_dir(repo)
    mirror.parent.mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "-q", "--mirror", str(source), str(mirror)], check=True
    )
    _repo_manifest_path(repo).write_text(
        json.dumps(
            {
                "cache_version": REPO_CACHE_VERSION,
                "repo": repo,
                "verified_commits": [commit],
            }
        ),
        encoding="utf-8",
    )

    assert _repo_cache_is_ready(repo, [commit])
    assert not _repo_cache_is_ready(repo, ["0" * 40])

    checkout = tmp_path / "checkout"
    _clone_instance_repository(
        {"repo": repo, "base_commit": commit, "instance_id": "example__project-1"},
        checkout,
    )
    assert (checkout / "payload.txt").read_text(encoding="utf-8") == "cached\n"
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        == commit
    )
    assert subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=checkout, text=True
    ).strip() == str(mirror)


def test_repository_cache_requires_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from monitor_runtime.runners import moatless as runner

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path / "repo_store")
    repo = "example/project"
    _repo_mirror_dir(repo).mkdir(parents=True)

    assert not _repo_cache_is_ready(repo, ["abc123"])


def _fixture_flow_config() -> dict:
    return {
        "agent": {
            "completion_model": {"model_id": "x", "model": "x"},
            "system_prompt": (
                "Make multiple tool calls in one response: do the thing. "
                "Make all necessary changes in one response: also this."
            ),
            "actions": [
                {"action_class": "moatless.actions.view_code.ViewCode"},
                {"action_class": "moatless.actions.run_tests.RunTests"},
                {"action_class": "moatless.actions.run_python_script.RunPythonScript"},
                {
                    "action_class": "moatless.actions.create_python_file.CreatePythonFile"
                },
                {"action_class": "moatless.actions.finish.Finish"},
            ],
        }
    }


@pytest.mark.parametrize("monitor_mode", ["vanilla", "liveplan"])
def test_patch_flow_preserves_native_action_set(monitor_mode: str) -> None:
    config = _patch_flow(
        _fixture_flow_config(), model="fake-model", monitor_mode=monitor_mode
    )
    action_classes = {action["action_class"] for action in config["agent"]["actions"]}

    assert action_classes == {
        "moatless.actions.view_code.ViewCode",
        "moatless.actions.run_tests.RunTests",
        "moatless.actions.run_python_script.RunPythonScript",
        "moatless.actions.create_python_file.CreatePythonFile",
        "moatless.actions.finish.Finish",
    }


def test_patch_flow_changes_action_cardinality_prompt_for_online_monitors() -> None:
    vanilla = _patch_flow(
        _fixture_flow_config(), model="fake-model", monitor_mode="vanilla"
    )
    liveplan = _patch_flow(
        _fixture_flow_config(), model="fake-model", monitor_mode="liveplan"
    )
    cadence = _patch_flow(
        _fixture_flow_config(), model="fake-model", monitor_mode="liveplan",
        liveplan_variant="adaptive_interval",
    )
    fixed_interval = _patch_flow(
        _fixture_flow_config(),
        model="fake-model",
        monitor_mode="liveplan",
        liveplan_variant="refiner_only",
    )

    original = _fixture_flow_config()["agent"]["system_prompt"]
    assert vanilla["agent"]["system_prompt"] == original
    assert fixed_interval["agent"]["system_prompt"] == original
    assert cadence["agent"]["system_prompt"] == original
    assert "Make exactly one tool call" in liveplan["agent"]["system_prompt"]
    assert "/tmp/rafd-evidence" not in vanilla["agent"]["system_prompt"]
    assert "/tmp/rafd-evidence" not in liveplan["agent"]["system_prompt"]


def test_extract_patch_preserves_historical_leaf_selection() -> None:
    class FileContext:
        def __init__(self, patch: str):
            self.patch = patch

        def generate_git_patch(self, *, ignore_tests: bool) -> str:
            assert ignore_tests is True
            return self.patch

    class Node:
        def __init__(self, patch: str):
            self.file_context = FileContext(patch)

    abandoned_leaf = Node("abandoned patch")
    final_node = Node("selected final patch")

    class Root:
        def get_leaf_nodes(self):
            return [final_node, abandoned_leaf]

    class Flow:
        root = Root()

    assert _extract_patch(Flow(), final_node) == "abandoned patch"

    abandoned_leaf.file_context.patch = ""
    assert _extract_patch(Flow(), final_node) == "selected final patch"


def test_sanitize_transient_patch_drops_only_untracked_diagnostics(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "debug.py"
    tracked.write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "debug.py"], check=True)
    patch = """--- a/pkg/source.py
+++ b/pkg/source.py
@@ -1 +1 @@
-old
+new
--- a/debug.py
+++ b/debug.py
@@ -1 +1 @@
-old
+new
--- a/tmp/rafd-evidence/repro_case.py
+++ b/tmp/rafd-evidence/repro_case.py
@@ -0,0 +1 @@
+print('probe')
--- a/run_setup_and_pytest.py
+++ b/run_setup_and_pytest.py
@@ -0,0 +1 @@
+print('probe')
--- a/scripts/rafd_pprint_test_repro.py
+++ b/scripts/rafd_pprint_test_repro.py
@@ -0,0 +1 @@
+print('probe')
--- a/design_notes/proposal.md
+++ b/design_notes/proposal.md
@@ -0,0 +1 @@
+temporary proposal
--- a/tools/patch_dependency_for_runtime.py
+++ b/tools/patch_dependency_for_runtime.py
@@ -0,0 +1 @@
+temporary compatibility shim
"""

    sanitized, removed = _sanitize_transient_patch(
        patch, repo_path=tmp_path, problem_statement="Fix pkg/source.py behavior."
    )

    assert "pkg/source.py" in sanitized
    assert "debug.py" in sanitized
    assert "repro_case.py" not in sanitized
    assert "run_setup_and_pytest.py" not in sanitized
    assert "rafd_pprint_test_repro.py" not in sanitized
    assert "design_notes/proposal.md" not in sanitized
    assert "tools/patch_dependency_for_runtime.py" not in sanitized
    assert removed == [
        "tmp/rafd-evidence/repro_case.py",
        "run_setup_and_pytest.py",
        "scripts/rafd_pprint_test_repro.py",
        "design_notes/proposal.md",
        "tools/patch_dependency_for_runtime.py",
    ]


def test_sanitize_transient_patch_keeps_new_file_named_by_task(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    patch = """--- a/scripts/run_tests.py
+++ b/scripts/run_tests.py
@@ -0,0 +1 @@
+print('required')
"""

    sanitized, removed = _sanitize_transient_patch(
        patch,
        repo_path=tmp_path,
        problem_statement="Please add scripts/run_tests.py for maintainers.",
    )

    assert sanitized == patch
    assert removed == []


def test_monitor_records_separate_blocking_and_advisory_events() -> None:
    trajectory = {
        "phase_monitor": {"disposition": "advise"},
        "monitor_events": [
            {"event_id": "block-1", "event_type": "monitor_intervention"},
            {
                "event_id": "advice-1",
                "event_type": "monitor_advisory",
                "metadata": {"executed_step_id": "step-2"},
            },
            {
                "event_id": "guidance-1",
                "event_type": "monitor_guidance",
                "metadata": {"guidance_source": "refiner"},
            },
        ],
    }

    decisions, interventions, guidance = _monitor_records(trajectory)

    assert len(decisions) == 1
    assert [event["event_id"] for event in interventions] == ["block-1"]
    assert [event["event_id"] for event in guidance] == [
        "advice-1",
        "guidance-1",
    ]


def test_monitor_records_counts_one_decision_per_action_group() -> None:
    shared_group_decision = {
        "action_group_step_id": "proposed-group-7",
        "action_group_size": 3,
        "execute_proposed_step": True,
    }
    trajectory = {
        "action_steps": [
            {"phase_monitor": {**shared_group_decision, "action_group_index": 0}},
            {"phase_monitor": {**shared_group_decision, "action_group_index": 1}},
            {"phase_monitor": {**shared_group_decision, "action_group_index": 2}},
            {"phase_monitor": {"execute_proposed_step": True}},
        ]
    }

    decisions, interventions, guidance = _monitor_records(trajectory)

    assert len(decisions) == 2
    assert decisions[0]["action_group_step_id"] == "proposed-group-7"
    assert "action_group_step_id" not in decisions[1]
    assert interventions == []
    assert guidance == []


def test_paid_check_count_uses_reviewer_events_not_policy_skip_decisions() -> None:
    overhead = {
        "summary": {
            "by_operation": {
                "phase_reviewer": {"event_count": 4},
                "phase_policy_resolver": {"event_count": 3},
            }
        }
    }

    assert _operation_event_count(overhead, "phase_reviewer") == 4
    assert _operation_event_count(overhead, "phase_policy_resolver") == 3
    assert _operation_event_count(overhead, "rule_check") == 0


# -- Fix (Codex review, 2026-09-06): `status`/`flow_error`/`finish_reason`/
# `checks` in run_result.json can all still read as a clean, uneventful run
# even when the flow's own final completion call came back empty (a real
# `CompletionRejectError`, `node.error` set, zero action_steps) or the run
# simply exhausted its budget without the agent ever calling Finish/Reject.
# `_terminal_status` reads the final node's own `.error` and its terminal
# action's declared name directly, instead of trusting those wrapper-level
# fields. --


class _FakeAction:
    def __init__(self, title: str):
        self.model_config = {"title": title}


class _FakeActionStep:
    def __init__(self, title: str):
        self.action = _FakeAction(title)


class _FakeNode:
    def __init__(self, *, error: str | None = None, action_steps: list | None = None):
        self.error = error
        self.action_steps = action_steps or []


def test_terminal_status_reports_finish_when_terminal_action_is_finish() -> None:
    node = _FakeNode(action_steps=[_FakeActionStep("Finish")])

    assert _terminal_status(node) == ("finish", None)


def test_terminal_status_reports_reject_when_terminal_action_is_reject() -> None:
    node = _FakeNode(action_steps=[_FakeActionStep("Reject")])

    assert _terminal_status(node) == ("reject", None)


def test_terminal_status_reports_completion_error_for_empty_final_node() -> None:
    # Mirrors the real bug: the flow's own status/flow_error/finish_reason
    # all still say "completed" for a node like this.
    node = _FakeNode(error="Completion validation error: Completion response is empty", action_steps=[])

    agent_exit, terminal_node_error = _terminal_status(node)

    assert agent_exit == "completion_error"
    assert terminal_node_error == "Completion validation error: Completion response is empty"


def test_terminal_status_reports_budget_exceeded_when_run_just_stopped() -> None:
    # No error, but the last action wasn't Finish/Reject either -- the run
    # simply ran out of iterations/depth without the agent ending it itself.
    node = _FakeNode(action_steps=[_FakeActionStep("StringReplace")])

    assert _terminal_status(node) == ("budget_exceeded_or_unknown", None)


def test_operation_event_count_covers_monitor_baseline_operations() -> None:
    # `checks` in run_result.json only ever counted `phase_reviewer` (the
    # monitor_runtime family's own operation), so every monitor_baseline
    # (LivePlan/Fixed-Interval/Adaptive Interval) run reported `checks: 0`
    # regardless of how many refiner calls actually happened. These three
    # operations are what those conditions actually record.
    overhead = {
        "summary": {
            "by_operation": {
                "rule_check": {"event_count": 26},
                "refiner_call": {"event_count": 2},
                "refiner_cooldown": {"event_count": 2},
            }
        }
    }

    assert _operation_event_count(overhead, "rule_check") == 26
    assert _operation_event_count(overhead, "refiner_call") == 2
    assert _operation_event_count(overhead, "refiner_cooldown") == 2
    assert _operation_event_count(overhead, "phase_reviewer") == 0

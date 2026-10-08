"""Tests for `monitor_baseline.bridge.LivePlanBoundaryMonitor`.

The variant `monitor` case asserts blocking happens purely from the rule
engine, with no LLM call anywhere in the path (there is no model object at
all for that variant). The variant `refiner` case uses `plan_refiner.models
.MockModel` (never a real API) to check the refiner is actually consulted
once a blocking rule fires.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitor_baseline.bridge import LivePlanBoundaryMonitor  # noqa: E402
from plan_refiner.models import OpenRouterModel  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("MONITOR_BASELINE_VARIANT", raising=False)
    monkeypatch.delenv("MONITOR_BASELINE_RULES_CONFIG", raising=False)
    monkeypatch.delenv("MONITOR_BASELINE_REFINER_CONFIG", raising=False)


# -- Fix 6: rule-check and refiner-call latency/tokens must be recorded --


def test_overhead_ledger_records_rule_check_events():
    bridge = LivePlanBoundaryMonitor(variant="monitor", issue_description="Fix the bug")
    bridge.before_action(1, "grep -rn foo src/", "looking")
    payload = bridge.overhead.payload()
    events = payload["events"]
    assert events, "expected at least one overhead event after before_action"
    assert any(event["component"] == "monitor" and event["operation"] == "rule_check" for event in events)


def test_overhead_ledger_records_refiner_call_on_blocking_trigger():
    bridge = LivePlanBoundaryMonitor(
        variant="refiner", refiner_config=_MOCK_REFINER_CONFIG, issue_description="Fix the bug"
    )
    result = bridge.before_action(1, "submit", "giving up early")
    assert result.is_blocking is True
    events = bridge.overhead.payload()["events"]
    assert any(event["component"] == "monitor" and event["operation"] == "refiner_call" for event in events)


def test_variant_monitor_has_no_refiner():
    bridge = LivePlanBoundaryMonitor(variant="monitor", issue_description="Fix the bug in module.py")
    assert bridge._refiner is None
    assert bridge._monitor is not None


def test_openrouter_refiner_retries_empty_cap_exhausted_completion(monkeypatch):
    model = OpenRouterModel(
        model_name="openai/gpt-5.2-codex",
        model_kwargs={"max_tokens": 3000},
    )
    responses = iter(
        [
            {
                "choices": [{"message": {"content": ""}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 3000,
                    "total_tokens": 3100,
                    "cost": 0.03,
                },
            },
            {
                "choices": [{"message": {"content": "<new_plan>Inspect the failing path.</new_plan>"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cost": 0.002,
                },
            },
        ]
    )
    calls = []

    def fake_query(messages, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(model, "_query_api", fake_query)
    content, usage = model.query("system", "user")

    assert "Inspect the failing path" in content
    assert calls == [{}, {"max_tokens": 16384}]
    assert usage == {
        "input_tokens": 200,
        "output_tokens": 3020,
        "total_tokens": 3220,
        "cost": pytest.approx(0.032),
    }
    assert model.n_calls == 2


def test_variant_monitor_blocks_on_repeated_noop_without_llm():
    bridge = LivePlanBoundaryMonitor(variant="monitor", issue_description="Fix the bug")

    # Drive a dwell/repeated-view style pattern: repeatedly viewing the same
    # file in L_navigate should eventually trigger a (non-blocking) repeat
    # warning, and repeated plan-compliance-style ending without doing any
    # work should trigger the blocking plan_compliance rule.
    blocked_seen = False
    for step in range(1, 12):
        result = bridge.before_action(step, "grep -rn 'foo' src/", "looking around again")
        assert result.execute or result.is_blocking
        if result.is_blocking:
            blocked_seen = True
            break

    # Ending the run (a "submit"-shaped command) before ever patching or
    # testing should trigger the blocking plan_compliance rule.
    ending_result = bridge.before_action(50, "submit", "I'm done")
    assert ending_result.is_blocking is True
    assert ending_result.execute is False
    assert ending_result.message

    # Nothing above touched a model at all -- the bridge never constructed one.
    assert bridge._refiner is None


def test_before_action_blocked_step_not_committed():
    bridge = LivePlanBoundaryMonitor(variant="monitor", issue_description="Fix the bug")
    bridge.before_action(1, "submit", "premature submit")
    # after_action is only ever called by adapters when `executed=True`, which
    # only happens for actions that were NOT blocked; simulate that contract.
    assert bridge._committed == []


def test_after_action_commits_only_executed_steps():
    bridge = LivePlanBoundaryMonitor(variant="monitor", issue_description="Fix the bug")
    bridge.after_action(1, "grep -rn foo src/", "looking", "some output", executed=True)
    assert len(bridge._committed) == 1
    bridge.after_action(2, "grep -rn bar src/", "looking more", "", executed=False)
    assert len(bridge._committed) == 1


_MOCK_REFINER_CONFIG = str(Path(__file__).resolve().parent / "fixtures" / "mock_refiner.yaml")


def test_variant_refiner_uses_mock_model_and_records_last_refinement():
    bridge = LivePlanBoundaryMonitor(
        variant="refiner", refiner_config=_MOCK_REFINER_CONFIG, issue_description="Fix the bug"
    )
    from plan_refiner.models import MockModel

    assert isinstance(bridge._refiner.model, MockModel)

    # Force a blocking rule to fire (premature ending) so the refiner is invoked.
    result = bridge.before_action(1, "submit", "giving up early")
    assert result.is_blocking is True
    # Either the refiner ran (guidance captured) or it's within a cooling
    # period on the very first call (last_refinement_step starts as None,
    # so it should NOT be cooling on the first call).
    assert bridge._last_refinement_step == 1


def test_variant_refiner_only_has_no_monitor():
    bridge = LivePlanBoundaryMonitor(variant="refiner_only", issue_description="Fix the bug")
    assert bridge._monitor is None
    assert bridge._refiner is not None
    assert bridge.refiner_config == "config/refiner-only.yaml"
    assert bridge.refiner_interval == 5
    result = bridge.before_action(1, "anything at all", "thinking")
    assert result.execute is True
    assert result.is_blocking is False


def test_refiner_cooldown_is_not_counted_as_an_llm_call():
    bridge = LivePlanBoundaryMonitor(
        variant="refiner", refiner_config=_MOCK_REFINER_CONFIG, issue_description="Fix the bug"
    )
    first = bridge.before_action(1, "submit", "giving up early")
    assert first.is_blocking
    bridge._refiner.config.min_steps_between_refinements = 5
    second = bridge.before_action(2, "submit", "still giving up")
    assert second.is_blocking

    operations = [event["operation"] for event in bridge.overhead.payload()["events"]]
    assert operations.count("refiner_call") == 1
    assert operations.count("refiner_cooldown") == 1


def test_full_refiner_cooldown_uses_upstream_step_and_trigger_conditions():
    bridge = LivePlanBoundaryMonitor(
        variant="refiner", refiner_config=_MOCK_REFINER_CONFIG, issue_description="Fix the bug"
    )
    bridge._refiner.config.min_steps_between_refinements = 5
    bridge._last_refinement_step = 3
    bridge._monitor_triggers_since_refinement = 0

    assert bridge._monitor_refiner_is_cooling(3) is True
    bridge._monitor_triggers_since_refinement = 5
    assert bridge._monitor_refiner_is_cooling(3) is False
    bridge._monitor_triggers_since_refinement = 0
    assert bridge._monitor_refiner_is_cooling(8) is False


def test_repeated_blocks_keep_step_fixed_and_escape_external_cooldown_by_trigger_count():
    bridge = LivePlanBoundaryMonitor(
        variant="refiner", refiner_config=_MOCK_REFINER_CONFIG, issue_description="Fix the bug"
    )
    bridge._refiner.config.min_steps_between_refinements = 5

    first = bridge.before_action(0, "submit", "giving up")
    assert first.is_blocking is True
    for proposal in range(1, 6):
        assert bridge._monitor_refiner_is_cooling(0) is True
        bridge._record_monitor_refiner_cooldown(
            step_index=0,
            accounting_step_id=str(proposal + 1),
        )

    assert bridge._monitor_refiner_is_cooling(0) is False
    from plan_refiner.types import RefinerInput, StateSummary

    output = bridge._invoke_refiner(
        RefinerInput(
            issue_description="Fix the bug",
            trajectory=[],
            state_summary=StateSummary(),
            last_refinement_step=bridge._last_refinement_step,
        ),
        step_index=0,
        accounting_step_id="7",
    )
    assert output.is_cooling_period is True
    bridge._note_refinement(0, output)

    # Five retries are stopped by the orchestration cooldown. The sixth
    # reaches PlanRefiner, whose own step-only cooldown returns without an
    # LLM call; upstream nevertheless resets the trigger counter here.
    operations = [event["operation"] for event in bridge.overhead.payload()["events"]]
    assert operations.count("refiner_call") == 1
    assert operations.count("refiner_cooldown") == 6
    assert bridge._last_refinement_step == 0
    assert bridge._monitor_triggers_since_refinement == 0


def test_refiner_only_matches_upstream_zero_based_five_step_interval():
    bridge = LivePlanBoundaryMonitor(variant="refiner_only", issue_description="Fix the bug")
    from plan_refiner.models import MockModel

    bridge._refiner.model = MockModel()
    outputs = []
    for step_index in range(11):
        outputs.append(
            bridge.after_action(
                step_index,
                f"grep item-{step_index}",
                "look",
                "result",
                executed=True,
            )
        )

    assert [index for index, value in enumerate(outputs) if value] == [5, 10]
    operations = [event["operation"] for event in bridge.overhead.payload()["events"]]
    assert operations.count("refiner_call") == 2
    assert "rule_check" not in operations


def test_reset_run_state_clears_committed_and_counters():
    bridge = LivePlanBoundaryMonitor(variant="monitor", issue_description="Fix the bug")
    bridge.after_action(1, "grep foo", "look", "out", executed=True)
    assert bridge._committed
    bridge.reset_run_state(issue_description="A new task")
    assert bridge._committed == []
    assert bridge.issue_description == "A new task"
    assert bridge._last_refinement_step is None
    assert bridge._monitor_triggers_since_refinement == 0


# -- Fix 4: MONITOR_BASELINE_REFINER_CONFIG must drive the prompt template --


def test_refiner_config_drives_template_not_just_model():
    bridge = LivePlanBoundaryMonitor(
        variant="refiner", refiner_config=_MOCK_REFINER_CONFIG, issue_description="Fix the bug"
    )
    assert bridge._refiner.config.template_path == _MOCK_REFINER_CONFIG


def test_refiner_model_can_be_overridden_independently_from_prompt_config():
    bridge = LivePlanBoundaryMonitor(
        variant="refiner",
        refiner_config=_MOCK_REFINER_CONFIG,
        refiner_model="anthropic/claude-sonnet-4.5",
        issue_description="Fix the bug",
    )

    assert bridge._refiner.config.template_path == _MOCK_REFINER_CONFIG
    assert bridge._refiner.model.config.model_name == "anthropic/claude-sonnet-4.5"
    # mock_refiner.yaml sets min_steps_between_refinements=1; the vendored
    # default.yaml sets 5. If the bridge silently fell back to the default
    # RefinerConfig() (ignoring _refiner_config for anything but model
    # selection), this would read 5 instead.
    assert bridge._refiner.config.min_steps_between_refinements == 1
    assert bridge._refiner.config.trajectory_scope == "recent"
    assert bridge._refiner.config.update_plan is False


# -- Fix 2: non-blocking refiner call must see the current step's observation --


def test_variant_refiner_non_blocking_call_happens_in_after_action_with_fresh_trajectory():
    bridge = LivePlanBoundaryMonitor(
        variant="refiner", refiner_config=_MOCK_REFINER_CONFIG, issue_description="Fix the bug"
    )

    captured: list = []
    original_query = bridge._refiner.model.query

    def _capturing_query(system_prompt, user_prompt):
        # The trajectory text is embedded in the user prompt by the time the
        # model is queried -- capture it so we can check it includes the
        # just-executed step's observation.
        captured.append(user_prompt)
        return original_query(system_prompt, user_prompt)

    bridge._refiner.model.query = _capturing_query

    # A non-blocking rule match: distinct (non-repeating) navigation commands
    # in the same phase for 7 steps trip the non-blocking `dwell_L_navigate`
    # rule (block_execution=false, threshold=7 in default_rules.json) without
    # ever tripping a blocking rule.
    triggered = False
    for step in range(1, 8):
        command = f"grep -rn 'foo' src/file_{step}.py"
        thought = f"looking at file {step}"
        result = bridge.before_action(step, command, thought)
        assert result.execute is True
        assert result.is_blocking is False
        # `before_action` must not have called the refiner yet for a
        # non-blocking match -- only `after_action` (once the observation is
        # committed) may do that.
        if bridge._pending_refiner_trigger is not None:
            triggered = True
            assert not captured, "refiner must not be queried from before_action for a non-blocking match"
            # Commit this step's observation, then let after_action run the
            # deferred refiner call.
            observation = f"grep found nothing new at step {step}"
            bridge.after_action(step, command, thought, observation, executed=True)
            assert bridge._pending_refiner_trigger is None
            assert captured, "after_action should have invoked the refiner once the trigger was pending"
            assert observation in captured[-1], "refiner prompt must include the just-committed observation"
            break
        bridge.after_action(step, command, thought, f"nothing at step {step}", executed=True)

    assert triggered, "expected a non-blocking rule match to fire within the loop"


def test_failed_native_action_does_not_leak_pending_refiner_trigger_to_next_proposal():
    bridge = LivePlanBoundaryMonitor(
        variant="refiner", refiner_config=_MOCK_REFINER_CONFIG, issue_description="Fix the bug"
    )

    # Reach the same non-blocking dwell trigger, but simulate native action
    # execution failing by deliberately omitting after_action for that step.
    for step in range(1, 8):
        command = f"grep -rn 'foo' src/file_{step}.py"
        result = bridge.before_action(step, command, f"looking at file {step}")
        assert result.execute is True
        if bridge._pending_refiner_trigger is not None:
            break
        bridge.after_action(step, command, f"looking at file {step}", "nothing", executed=True)
    else:
        raise AssertionError("expected a non-blocking trigger")

    retry = bridge.before_action(8, "sed -n '1,80p' src/new.py", "retry with valid syntax")
    assert retry.execute is True
    assert bridge._pending_refiner_trigger is None
    assert bridge.after_action(8, "sed -n '1,80p' src/new.py", "retry", "ok", executed=True) is None

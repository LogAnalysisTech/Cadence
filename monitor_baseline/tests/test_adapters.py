"""Tests for `monitor_baseline.adapters.{swe_agent,mini_swe_agent}`.

Fix 3: neither adapter's `_*_task_context` helper was ever invoked, so
`LivePlanBoundaryMonitor.issue_description` stayed empty for the whole run.
Both adapters now lazy-fill it on the first `inspect_before_action` call.

Fix 7: both adapters' own `_proposal_counter_int` instance attribute is not
known to `NativePhaseMonitorIntegration.reset_run_state()`, so each adapter
now overrides `reset_run_state()` to reset it before delegating to `super()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitor_baseline.adapters.mini_swe_agent import MiniSWEAgentLivePlanIntegration  # noqa: E402
from monitor_baseline.bridge import BeforeActionResult, LivePlanBoundaryMonitor  # noqa: E402
from monitor_runtime.tests.mini_agent_fixtures import _FakeMiniAgent
from minisweagent.exceptions import FormatError, Submitted  # noqa: E402












def test_mini_swe_agent_adapter_lazy_fills_issue_description_on_first_inspect():
    monitor = LivePlanBoundaryMonitor(variant="monitor")
    assert monitor.issue_description == ""

    agent = _FakeMiniAgent(
        messages=[{"role": "user", "content": "Fix the off-by-one bug", "extra": {}}]
    )
    integration = MiniSWEAgentLivePlanIntegration(monitor)
    message = {"content": "looking", "extra": {}}
    action = {"command": "grep -rn foo src/"}
    integration.inspect_before_action(agent, message, action)

    assert monitor.issue_description == "Fix the off-by-one bug"


def test_mini_swe_agent_adapter_reset_run_state_resets_local_step_counter():
    monitor = LivePlanBoundaryMonitor(variant="monitor")
    agent = _FakeMiniAgent(messages=[{"role": "user", "content": "task", "extra": {}}])
    integration = MiniSWEAgentLivePlanIntegration(monitor)
    integration.inspect_before_action(agent, {"content": "t1", "extra": {}}, {"command": "grep foo"})
    integration.inspect_before_action(agent, {"content": "t2", "extra": {}}, {"command": "grep bar"})
    assert integration._proposal_counter_int == 2

    integration.reset_run_state()
    assert integration._proposal_counter_int == 0


def test_mini_swe_agent_liveplan_accepts_one_action_batch():
    monitor = LivePlanBoundaryMonitor(variant="monitor")
    message = {"role": "assistant", "content": "inspect", "extra": {}}
    agent = _FakeMiniAgent(messages=[message])
    integration = MiniSWEAgentLivePlanIntegration(monitor)

    integration.validate_action_batch(agent, message, [{"command": "grep foo"}])

    assert agent.messages == [message]


def test_mini_swe_agent_liveplan_rejects_multi_action_batch_atomically():
    monitor = LivePlanBoundaryMonitor(variant="monitor")
    integration = MiniSWEAgentLivePlanIntegration(monitor)
    event_id = monitor.overhead.record(
        component="agent", operation="agent_generation", latency_seconds=0.1
    )
    message = {
        "role": "assistant",
        "content": "inspect and edit",
        "extra": {"overhead_event_id": event_id},
    }
    agent = _FakeMiniAgent(messages=[message])

    try:
        integration.validate_action_batch(
            agent,
            message,
            [{"command": "grep foo"}, {"command": "sed -i s/foo/bar/ target.py"}],
        )
    except FormatError as exc:
        assert len(exc.messages) == 1
        assert exc.messages[0]["extra"]["monitor_baseline_batch_rejected"] is True
    else:
        raise AssertionError("Expected a multi-action LivePlan turn to be rejected")

    assert agent.messages == []
    event = next(e for e in monitor.overhead.payload()["events"] if e["event_id"] == event_id)
    assert event["outcome"] == "discarded_format"
    assert monitor.overhead.payload()["summary"]["by_component"]["monitor"]["event_count"] == 0


# -- Fix 3: adapters must call associate_generation/mark_generation_outcome --


def test_mini_swe_agent_adapter_marks_accepted_generation_outcome():
    monitor = LivePlanBoundaryMonitor(variant="monitor")
    agent = _FakeMiniAgent(messages=[{"role": "user", "content": "task", "extra": {}}])
    integration = MiniSWEAgentLivePlanIntegration(monitor)

    event_id = monitor.overhead.record(component="agent", operation="agent_generation", latency_seconds=0.1)
    message = {"content": "looking", "extra": {"overhead_event_id": event_id}}
    action = {"command": "grep -rn foo src/"}

    integration.inspect_before_action(agent, message, action)

    event = next(e for e in monitor.overhead.payload()["events"] if e["event_id"] == event_id)
    assert event["outcome"] == "accepted"
    assert event["step_id"] == "1"
    assert message["extra"]["overhead_step_id"] == "1"


def test_mini_swe_agent_adapter_marks_discarded_generation_outcome_when_blocked():
    monitor = LivePlanBoundaryMonitor(variant="monitor")
    agent = _FakeMiniAgent(messages=[{"role": "user", "content": "task", "extra": {}}])
    integration = MiniSWEAgentLivePlanIntegration(monitor)

    event_id = monitor.overhead.record(component="agent", operation="agent_generation", latency_seconds=0.1)
    message = {"content": "giving up early", "extra": {"overhead_event_id": event_id}}
    action = {"command": "submit"}

    decision = integration.inspect_before_action(agent, message, action)

    event = next(e for e in monitor.overhead.payload()["events"] if e["event_id"] == event_id)
    assert event["outcome"] == "discarded"
    assert decision.monitor_event is not None
    assert decision.monitor_event.discarded_proposed_step_id == "1"
    assert agent.messages[-1]["extra"]["discarded_proposed_step_id"] == "1"


def test_mini_swe_agent_adapter_force_submits_instead_of_trusting_agent_compliance():
    # A give_up-style verdict (BeforeActionResult.force_submit=True) must not
    # just ask the agent nicely to submit -- the adapter has to submit the
    # current diff itself, since an agent that ignores the request would
    # otherwise just keep getting re-blocked until the step budget runs out.
    monitor = LivePlanBoundaryMonitor(variant="monitor")
    monitor.before_action = lambda *args, **kwargs: BeforeActionResult(
        execute=False,
        message="Stop investigating and submit the current diff as-is, then end the task.",
        is_blocking=True,
        force_submit=True,
    )
    agent = _FakeMiniAgent(
        messages=[
            {"role": "user", "content": "task", "extra": {}},
            {"role": "assistant", "content": "still investigating", "extra": {}},
        ]
    )
    integration = MiniSWEAgentLivePlanIntegration(monitor)
    message = {"content": "still investigating", "extra": {}}
    action = {"command": "grep -rn foo src/"}

    try:
        integration.inspect_before_action(agent, message, action)
        raised = False
    except Submitted:
        raised = True

    assert raised, "force_submit must raise Submitted, not just inject a text request"
    assert agent.env.executed_commands == ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"]
    # The stale assistant proposal must be gone -- the submission is not a
    # regeneration the agent gets to see and potentially ignore again.
    assert agent.messages[-1]["role"] == "user"

"""Tests for `cadence.monitor.CadenceMonitor`.

Uses a stub model (never a real API) so the interval state machine and
block/advise wiring can be tested deterministically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadence.monitor import (  # noqa: E402
    BASE_INTERVAL,
    GROWTH_STEP,
    MAX_INTERVAL,
    CadenceMonitor,
)


@dataclass
class _StubModel:
    """Returns queued (verdict, plan_text) pairs in order; records every call."""

    responses: list[tuple[str, str]]
    calls: list[tuple[str, str]] = field(default_factory=list)

    @property
    def config(self):
        return self

    model_name = "stub-model"

    def query(self, system_prompt: str, user_prompt: str):
        self.calls.append((system_prompt, user_prompt))
        verdict, plan = self.responses[len(self.calls) - 1]
        text = f"<verdict>{verdict}</verdict>\n\n<new_plan>\n{plan}\n</new_plan>"
        return text, {"input_tokens": 10, "output_tokens": 5}


def _make_monitor(responses: list[tuple[str, str]]) -> CadenceMonitor:
    monitor = CadenceMonitor(refiner_model="stub", issue_description="Fix the bug")
    monitor._model = _StubModel(responses=responses)
    return monitor


@dataclass
class _RawStubModel:
    """Returns queued raw response texts verbatim, bypassing the <verdict> wrapper."""

    raw_texts: list[str]
    calls: list[tuple[str, str]] = field(default_factory=list)

    @property
    def config(self):
        return self

    model_name = "stub-model"

    def query(self, system_prompt: str, user_prompt: str):
        self.calls.append((system_prompt, user_prompt))
        return self.raw_texts[len(self.calls) - 1], {"input_tokens": 10, "output_tokens": 5}


def _make_monitor_raw(raw_texts: list[str]) -> CadenceMonitor:
    monitor = CadenceMonitor(refiner_model="stub", issue_description="Fix the bug")
    monitor._model = _RawStubModel(raw_texts=raw_texts)
    return monitor


def test_no_check_before_base_interval():
    monitor = _make_monitor([("advise", "keep going")])
    for step in range(BASE_INTERVAL):
        result = monitor.before_action(step, "ls", "looking around")
        assert result.execute is True
        assert result.message is None
    assert monitor._model.calls == []


def test_advise_widens_interval_and_never_blocks():
    monitor = _make_monitor([("advise", "step one"), ("advise", "step two")])
    result = monitor.before_action(BASE_INTERVAL, "ls", "t")
    assert result.execute is True
    assert result.is_blocking is False
    assert "step one" in result.message
    assert monitor.refiner_interval == BASE_INTERVAL + GROWTH_STEP

    next_step = BASE_INTERVAL + monitor.refiner_interval
    result = monitor.before_action(next_step, "ls", "t")
    assert result.is_blocking is False
    assert monitor.refiner_interval == BASE_INTERVAL + 2 * GROWTH_STEP


def test_recent_trajectory_includes_step_checked_at_previous_advise():
    # Regression for the off-by-one where the step approved by a check (and
    # committed right after) never appeared in any later check's trajectory.
    monitor = _make_monitor([("advise", "first"), ("advise", "second")])
    monitor.before_action(BASE_INTERVAL, "echo marker_action_5", "thought_5")
    monitor.after_action(BASE_INTERVAL, "echo marker_action_5", "thought_5", "observation_5", executed=True)

    next_step = BASE_INTERVAL + monitor.refiner_interval
    monitor.before_action(next_step, "ls", "t")

    _, second_user_prompt = monitor._model.calls[1]
    assert "marker_action_5" in second_user_prompt


def test_recent_trajectory_includes_corrective_step_after_block():
    # Regression: the corrective step taken right after a block must show up
    # in the next check, not just the steps after that.
    monitor = _make_monitor([("block", "stop"), ("advise", "next")])
    monitor.before_action(BASE_INTERVAL, "rm -rf /", "bad idea")
    monitor.after_action(
        BASE_INTERVAL, "echo corrective_marker", "retry", "observation_retry", executed=True
    )

    next_step = BASE_INTERVAL + monitor.refiner_interval  # reset to BASE_INTERVAL after block
    monitor.before_action(next_step, "ls", "t")

    _, second_user_prompt = monitor._model.calls[1]
    assert "corrective_marker" in second_user_prompt


def test_missing_verdict_tag_is_a_no_op_check():
    # No <verdict> tag at all: treated as if the check never happened.
    monitor = _make_monitor_raw(["<new_plan>Stop this action</new_plan>"])
    result = monitor.before_action(BASE_INTERVAL, "ls", "t")
    assert result.execute is True
    assert result.is_blocking is False
    assert result.message is None
    assert monitor.refiner_interval == BASE_INTERVAL
    events = monitor.overhead.payload()["events"]
    assert events[0]["outcome"] == "format_error"
    assert "Stop this action" in events[0]["metadata"]["raw_response"]


def test_block_with_empty_plan_is_a_no_op_check_not_a_silent_block():
    # Codex-reported case: block + empty <new_plan> used to discard the
    # action with no corrective feedback at all. Must now fail open instead.
    monitor = _make_monitor_raw(["<verdict>block</verdict>\n\n<new_plan>\n\n</new_plan>"])
    result = monitor.before_action(BASE_INTERVAL, "rm -rf /", "t")
    assert result.execute is True
    assert result.is_blocking is False
    assert result.message is None
    events = monitor.overhead.payload()["events"]
    assert events[0]["outcome"] == "format_error"


def test_advise_with_empty_plan_does_not_widen_interval():
    # Codex-reported case: advise + empty <new_plan> used to still widen the
    # interval even though there was no guidance to show.
    monitor = _make_monitor_raw(["<verdict>advise</verdict>\n\n<new_plan>\n\n</new_plan>"])
    monitor.before_action(BASE_INTERVAL, "ls", "t")
    assert monitor.refiner_interval == BASE_INTERVAL


def test_verdict_only_response_does_not_leak_tag_as_guidance():
    # Codex-reported case: a bare `<verdict>block</verdict>` with no
    # `<new_plan>` at all used to fall back to injecting the raw response
    # (i.e. the verdict tag itself) as if it were guidance.
    monitor = _make_monitor_raw(["<verdict>block</verdict>"])
    result = monitor.before_action(BASE_INTERVAL, "rm -rf /", "t")
    assert result.execute is True
    assert result.is_blocking is False
    assert result.message is None


def test_interval_caps_at_max():
    # Enough consecutive advises to blow past the cap.
    responses = [("advise", f"plan {i}") for i in range(10)]
    monitor = _make_monitor(responses)
    step = BASE_INTERVAL
    for _ in range(10):
        monitor.before_action(step, "ls", "t")
        step += monitor.refiner_interval
    assert monitor.refiner_interval == MAX_INTERVAL


def test_block_discards_action_and_resets_interval():
    monitor = _make_monitor([("advise", "widen"), ("block", "stop and redo")])
    monitor.before_action(BASE_INTERVAL, "ls", "t")
    widened = monitor.refiner_interval
    assert widened > BASE_INTERVAL

    result = monitor.before_action(BASE_INTERVAL + widened, "rm -rf /", "giving up")
    assert result.execute is False
    assert result.is_blocking is True
    assert "stop and redo" in result.message
    assert monitor.refiner_interval == BASE_INTERVAL


def test_overhead_ledger_records_refiner_call_with_verdict_metadata():
    monitor = _make_monitor([("block", "stop")])
    monitor.before_action(BASE_INTERVAL, "ls", "t")
    events = monitor.overhead.payload()["events"]
    refiner_events = [e for e in events if e["operation"] == "refiner_call"]
    assert len(refiner_events) == 1
    assert refiner_events[0]["metadata"]["verdict"] == "block"


def test_after_action_commits_step_and_returns_no_extra_guidance():
    monitor = _make_monitor([("advise", "plan")])
    assert monitor.executed_step_count == 0
    message = monitor.after_action(0, "ls", "t", "output", executed=True)
    assert message is None
    assert monitor.executed_step_count == 1


def test_reset_run_state_clears_interval_and_history():
    monitor = _make_monitor([("advise", "plan"), ("advise", "plan two")])
    monitor.before_action(BASE_INTERVAL, "ls", "t")
    assert monitor.refiner_interval != BASE_INTERVAL

    monitor.reset_run_state(issue_description="A different bug")
    assert monitor.refiner_interval == BASE_INTERVAL
    assert monitor.executed_step_count == 0
    assert monitor.issue_description == "A different bug"
    assert "(none yet)" in monitor._format_guidance_history()


def test_give_up_discards_action_and_resets_interval_like_block():
    monitor = _make_monitor([("advise", "widen"), ("give_up", "Submit the current diff and stop.")])
    monitor.before_action(BASE_INTERVAL, "ls", "t")
    widened = monitor.refiner_interval
    assert widened > BASE_INTERVAL

    result = monitor.before_action(BASE_INTERVAL + widened, "grep -r foo", "still stuck")
    assert result.execute is False
    assert result.is_blocking is True
    assert "Submit the current diff and stop." in result.message
    assert monitor.refiner_interval == BASE_INTERVAL


def test_guidance_history_replayed_in_next_prompt():
    monitor = _make_monitor([("advise", "first piece of guidance")])
    monitor.before_action(BASE_INTERVAL, "ls", "t")

    history = monitor._format_guidance_history()
    assert "[advise] first piece of guidance" in history


def test_guidance_history_capped_at_five_most_recent():
    responses = [("advise", f"guidance number {i}") for i in range(7)]
    monitor = _make_monitor(responses)
    step = BASE_INTERVAL
    for _ in range(7):
        monitor.before_action(step, "ls", "t")
        step += monitor.refiner_interval  # interval keeps widening/capping, still monotonic

    history = monitor._format_guidance_history()
    assert len(monitor._guidance_history) == 5
    # The two oldest rounds must have been evicted.
    assert "guidance number 0" not in history
    assert "guidance number 1" not in history
    assert "guidance number 6" in history


def test_malformed_response_does_not_pollute_guidance_history():
    monitor = _make_monitor_raw(["<verdict>block</verdict>"])  # no <new_plan> -> invalid
    monitor.before_action(BASE_INTERVAL, "ls", "t")
    assert monitor._guidance_history == []


def test_fixed_schedule_preserves_verdicts_and_recent_trajectory():
    monitor = CadenceMonitor(refiner_model='stub', fixed_interval=5)
    monitor._model = _StubModel(responses=[('advise', 'continue'), ('block', 'correct'), ('give_up', 'submit')])
    for step in range(16):
        result = monitor.before_action(step, 'echo test', 'inspect')
        assert monitor.refiner_interval == 5
        if step == 5:
            assert result.execute and not result.is_blocking
        elif step == 10:
            assert not result.execute and result.is_blocking and not result.force_submit
        elif step == 15:
            assert not result.execute and result.force_submit
        else:
            assert result.execute and result.message is None
        monitor.after_action(step, 'echo test', 'inspect', f'observation-{step}', result.execute)
    assert len(monitor._model.calls) == 3
    assert 'observation-5' in monitor._model.calls[1][1]
    monitor.reset_run_state('new task')
    assert monitor.refiner_interval == 5 and monitor._last_check_step is None
    assert not monitor._guidance_history


@pytest.mark.parametrize('fixed_interval', [-1, 1.5, True])
def test_invalid_fixed_interval_rejected(fixed_interval):
    with pytest.raises(ValueError):
        CadenceMonitor(fixed_interval=fixed_interval)


@pytest.mark.parametrize('base,growth', [(5, 2), (5, 4), (5, 6), (3, 2), (7, 2)])
def test_configurable_schedule_growth_cap_reset(base, growth):
    monitor = CadenceMonitor(refiner_model='stub', base_interval=base, growth_step=growth)
    monitor._model = _StubModel(responses=[('advise', 'continue')] * 10 + [('block', 'correct'), ('give_up', 'submit')])
    step = base
    for index in range(12):
        assert monitor.before_action(step - 1, 'ls', 'inspect').message is None
        result = monitor.before_action(step, 'ls', 'inspect')
        if index < 10:
            assert result.execute
            assert monitor.refiner_interval == min(20, base + (index + 1) * growth)
        else:
            assert not result.execute
            assert monitor.refiner_interval == base
            assert result.force_submit == (index == 11)
        step += monitor.refiner_interval
    monitor.reset_run_state('next issue')
    assert monitor.refiner_interval == base


@pytest.mark.parametrize('kwargs', [{'base_interval': 0}, {'base_interval': 21}, {'base_interval': True}, {'growth_step': 0}, {'growth_step': 1.5}])
def test_invalid_adaptive_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        CadenceMonitor(**kwargs)


@pytest.mark.parametrize('mode', ['severity', 'always-advisory', 'always-replacement'])
def test_execution_override_preserves_judgment_schedule_and_prompt_history(mode):
    monitor = CadenceMonitor(refiner_model='stub', execution_mode=mode)
    verdicts = ['advise', 'block', 'advise', 'give_up', 'advise']
    monitor._model = _StubModel([(v, f'guidance {i}') for i, v in enumerate(verdicts)])
    step = 5
    for i, (verdict, expected_interval) in enumerate(zip(verdicts, [7, 5, 7, 5, 7])):
        result = monitor.before_action(step, 'echo proposal', 'thought')
        expected_execute = mode == 'always-advisory' or (mode == 'severity' and verdict == 'advise')
        assert result.execute == expected_execute
        assert result.is_blocking == (not expected_execute)
        assert result.force_submit == (mode == 'severity' and verdict == 'give_up')
        assert result.judged_verdict == verdict
        assert result.execution_override == mode
        assert monitor.refiner_interval == expected_interval
        assert monitor._guidance_history[-1] == (verdict, f'guidance {i}')
        if i:
            assert f'[{verdicts[i-1]}] guidance {i-1}' in monitor._model.calls[-1][1]
        event = monitor.overhead.payload()['events'][-1]['metadata']
        assert event['verdict'] == event['judged_verdict'] == verdict
        assert event['interval_after'] == expected_interval
        assert event['execution_mode'] == ('advisory' if expected_execute else 'replacement')
        # A discarded proposal must not trigger another check before an action progresses.
        assert monitor.before_action(step, 'echo retry', 'retry').message is None
        step += expected_interval


@pytest.mark.parametrize('mode', ['always-advisory', 'always-replacement'])
def test_execution_override_does_not_act_on_invalid_monitor_response(mode):
    monitor = CadenceMonitor(refiner_model='stub', execution_mode=mode)
    monitor._model = _RawStubModel(['<verdict>block</verdict>'])
    result = monitor.before_action(5, 'echo proposal', 'thought')
    assert result.execute and not result.is_blocking and not result.force_submit
    assert result.message is None
    assert monitor.refiner_interval == 5
    assert monitor._guidance_history == []
    assert monitor.overhead.payload()['events'][-1]['metadata']['execution_mode'] is None

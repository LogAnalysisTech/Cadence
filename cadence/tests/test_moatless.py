"""Tests for `cadence.adapters.moatless`.

Real moatless `Node`/`ActionArguments` classes are not imported here (see
`test_moatless_role_classifier.py`'s docstring for why); lightweight
stand-ins carrying the same attribute shapes are used instead, matching the
convention in `test_moatless_monitor.py`.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitor_runtime.factory import create_moatless_integration  # noqa: E402
from monitor_baseline.adapters.moatless import MoatlessLivePlanIntegration  # noqa: E402
from cadence.adapters.moatless import (  # noqa: E402
    MoatlessCadenceIntegration,
)
from cadence.monitor import (  # noqa: E402
    BASE_INTERVAL,
    SYSTEM_PROMPT,
    CadenceMonitor,
)
from cadence.prompts_moatless import (  # noqa: E402
    MOATLESS_SYSTEM_PROMPT,
    format_moatless_trajectory,
)
from monitor_baseline.bridge import LivePlanBoundaryMonitor  # noqa: E402


def make_action(class_name: str, **fields):
    stub_cls = type(class_name, (SimpleNamespace,), {})
    obj = stub_cls(**fields)
    if not hasattr(obj, "to_prompt"):
        obj.to_prompt = lambda: f"{class_name}({fields})"
    return obj


class FakeActionStep:
    def __init__(self, action, observation=None):
        self.action = action
        self.observation = observation
        self.phase_monitor: dict = {}


class FakeNode:
    """Minimal stand-in for moatless `Node`: a single-branch parent chain."""

    def __init__(self, node_id, parent=None, action_steps=None, assistant_message: str = "", user_message: str = ""):
        self.node_id = node_id
        self.parent = parent
        self.action_steps = action_steps or []
        self.assistant_message = assistant_message
        self.user_message = user_message
        self.thoughts = None
        self.thinking_blocks = None
        self.monitor_events: list = []
        self.overhead_step_id = None
        self.overhead_generation_event_id = None

    def get_trajectory(self):
        chain = []
        node = self
        while node is not None:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))


@dataclass
class _StubModel:
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
    monitor = CadenceMonitor(refiner_model="stub", issue_description="")
    monitor._model = _StubModel(responses=responses)
    return monitor


def _first_response_node(user_message: str = "Fix the bug") -> FakeNode:
    """A root node whose own trajectory carries the task description, matching
    `_moatless_task_context`'s lookup of `get_trajectory()[0].user_message`."""

    return FakeNode(node_id=0, user_message=user_message)


def test_create_moatless_integration_dispatches_on_variant():
    adaptive = CadenceMonitor(refiner_model="stub")
    rule_based = LivePlanBoundaryMonitor(variant="monitor")

    assert isinstance(create_moatless_integration(adaptive), MoatlessCadenceIntegration)
    assert isinstance(create_moatless_integration(rule_based), MoatlessLivePlanIntegration)


def test_multi_action_group_reviewed_atomically_on_advise():
    monitor = _make_monitor([("advise", "keep going")])
    integration = MoatlessCadenceIntegration(monitor)
    root = _first_response_node()

    for _ in range(BASE_INTERVAL + 1):
        node = FakeNode(node_id=1, parent=root, assistant_message="looking")
        steps = [
            FakeActionStep(make_action("GrepToolArgs", pattern="foo")),
            FakeActionStep(make_action("ReadFileArgs", file_path="a.py")),
        ]
        node.action_steps = steps
        decision = asyncio.run(integration.inspect_before_action_group(agent=None, node=node, action_steps=steps))
        assert decision.execute_proposed_step is True
        for step in steps:
            step.observation = SimpleNamespace(message=f"observed {step.action.to_prompt()}")
        asyncio.run(integration.handle_after_action_group(agent=None, node=node, action_steps=steps))
        root = node

    # The check due at step BASE_INTERVAL reviews the whole two-action group
    # as one proposal and, on advise, attaches guidance to the last action's
    # observation only (matching what moatless serializes forward).
    assert "GrepToolArgs" in steps[0].observation.message
    assert "keep going" not in steps[0].observation.message
    assert "keep going" in steps[1].observation.message
    assert monitor._model.calls  # the check actually fired


def test_multi_action_group_discarded_atomically_on_block():
    monitor = _make_monitor([("block", "stop and reconsider")])
    integration = MoatlessCadenceIntegration(monitor)
    root = _first_response_node()

    node = FakeNode(node_id=1, parent=root, assistant_message="looking")
    steps = [
        FakeActionStep(make_action("GrepToolArgs", pattern="foo")),
        FakeActionStep(make_action("ReadFileArgs", file_path="a.py")),
        FakeActionStep(make_action("RunTestsArgs")),
    ]
    node.action_steps = steps

    # Let the root-reset (triggered on first sight of this root) happen up
    # front, then force the very next check to be due immediately -- this
    # must not be undone by another reset inside `inspect_before_action_group`.
    integration._maybe_reset_for_root(node)
    monitor._interval = 0
    decision = asyncio.run(integration.inspect_before_action_group(agent=None, node=node, action_steps=steps))

    assert decision.execute_proposed_step is False
    # The whole group is discarded, not just one action -- nothing in it executed.
    assert node.action_steps == []
    assert node.assistant_message is None
    assert "stop and reconsider" in node.user_message
    assert len(node.monitor_events) == 1


def test_lazy_fill_does_not_reset_state_agent_py_already_reset():
    """`_maybe_reset_for_root` must NOT re-reset the monitor: the real
    `ActionAgent.run()` already calls `reset_run_state()` on this exact
    new-root transition, before generating this node's actions -- and
    that reset already recorded an `agent_generation` overhead event this
    hook must not wipe (see the docstring in `_maybe_reset_for_root`)."""

    monitor = _make_monitor([("advise", "guidance")])
    integration = MoatlessCadenceIntegration(monitor)

    # Simulate the overhead event `ActionAgent.run()` records right after its
    # own (separate) reset-on-new-root, before ever calling this adapter.
    event_id = monitor.overhead.record(component="agent", operation="agent_generation", latency_seconds=0.1)

    root = _first_response_node("Fix bug A")
    node = FakeNode(node_id=1, parent=root)
    steps = [FakeActionStep(make_action("GrepToolArgs", pattern="a"))]
    node.action_steps = steps
    asyncio.run(integration.inspect_before_action_group(agent=None, node=node, action_steps=steps))

    # The pre-existing agent_generation event must survive this call.
    events = monitor.overhead.payload()["events"]
    assert any(e["event_id"] == event_id for e in events)
    assert monitor.issue_description == "Fix bug A"


def test_new_root_only_refills_issue_description_reset_is_the_callers_job():
    monitor = _make_monitor(
        [("advise", "first case guidance")] * BASE_INTERVAL + [("advise", "second case guidance")]
    )
    integration = MoatlessCadenceIntegration(monitor)

    case_one_root = _first_response_node("Fix bug A")
    root = case_one_root
    for _ in range(BASE_INTERVAL + 1):
        node = FakeNode(node_id=1, parent=root)
        steps = [FakeActionStep(make_action("GrepToolArgs", pattern="a"))]
        node.action_steps = steps
        asyncio.run(integration.inspect_before_action_group(agent=None, node=node, action_steps=steps))
        for step in steps:
            step.observation = SimpleNamespace(message="ok")
        asyncio.run(integration.handle_after_action_group(agent=None, node=node, action_steps=steps))
        root = node
    assert monitor.issue_description == "Fix bug A"
    assert monitor.refiner_interval > BASE_INTERVAL

    # In production, `ActionAgent.run()` calls `reset_run_state()` itself on
    # this transition, before this adapter ever sees case two's first node --
    # simulate that here rather than relying on this adapter to do it.
    integration.reset_run_state()

    case_two_root = _first_response_node("Fix bug B")
    node2 = FakeNode(node_id=1, parent=case_two_root)
    steps2 = [FakeActionStep(make_action("GrepToolArgs", pattern="b"))]
    node2.action_steps = steps2
    asyncio.run(integration.inspect_before_action_group(agent=None, node=node2, action_steps=steps2))

    assert monitor.issue_description == "Fix bug B"
    assert monitor.refiner_interval == BASE_INTERVAL


def test_persist_overhead_stores_payload_on_trajectory_root():
    monitor = CadenceMonitor(refiner_model="stub")
    integration = MoatlessCadenceIntegration(monitor)

    root = FakeNode(node_id=0)
    child = FakeNode(node_id=1, parent=root)

    integration.persist_overhead(child)

    assert root.overhead["schema_version"] == monitor.overhead.payload()["schema_version"]
    assert not hasattr(child, "overhead")


def test_moatless_integration_swaps_in_its_own_prompt_without_touching_the_default():
    monitor = CadenceMonitor(refiner_model="stub")
    assert monitor.system_prompt == SYSTEM_PROMPT  # unmodified before wrapping

    MoatlessCadenceIntegration(monitor)

    assert monitor.system_prompt == MOATLESS_SYSTEM_PROMPT
    assert monitor.system_prompt != SYSTEM_PROMPT
    assert "sed/perl" not in monitor.system_prompt
    assert "mini-swe-agent" not in monitor.system_prompt
    assert "StringReplace" in monitor.system_prompt

    # The module-level default used by mini-swe-agent must be untouched by
    # having constructed a moatless integration around a *different* monitor
    # instance.
    other_monitor = CadenceMonitor(refiner_model="stub")
    assert other_monitor.system_prompt == SYSTEM_PROMPT


def test_moatless_integration_also_swaps_in_the_bash_free_trajectory_formatter():
    """The proposed-action block's bash fence was fixed earlier, but the
    shared `TrajectoryFormatter` used for *historical* steps still wrapped
    every action in one too (Codex review, 2026-09-06) -- moatless's own
    action rendering (`ToolName(arg=value, ...)`) is not shell text."""

    monitor = CadenceMonitor(refiner_model="stub")
    MoatlessCadenceIntegration(monitor)

    assert monitor.format_trajectory is format_moatless_trajectory

    from plan_refiner.types import RefinerTrajectoryStep

    step = RefinerTrajectoryStep(step_index=1, thought="looking", action="GrepTool(pattern='foo')", observation="ok")
    rendered = monitor.format_trajectory([step])

    assert "```bash" not in rendered
    assert "GrepTool(pattern='foo')" in rendered

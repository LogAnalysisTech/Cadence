"""Tests for `monitor_baseline.moatless_monitor` and the moatless LivePlan adapter.

Fix 1: `MoatlessLivePlanIntegration` was missing `persist_overhead`, which
`coding_agent-baselines/moatless/moatless/agent/agent.py` calls unconditionally
after every model generation whenever a realtime phase monitor is attached --
so every moatless baseline run crashed with `AttributeError` on the very
first generation.

Fix 2: `derive_node_monitor_state(node, ...)` caches its result on the node
the first time it is called, and returns the cached object unconditionally
on every later call -- even after one of the node's own `action_steps` has
gone from `observation=None` (not yet executed) to a real observation
(executed). `invalidate_node_monitor_state` drops that cache so the next
`derive_node_monitor_state` call re-replays from scratch. Without this, the
non-blocking `refiner` variant's deferred `handle_after_action` ->
`_call_refiner` call would build its `state_summary` from a stale
pre-execution monitor that never saw the just-executed step's role/phase
contribution.

Real moatless `ActionArguments`/`Node` classes are not imported here (see
`test_moatless_role_classifier.py` docstring for why); lightweight stand-ins
carrying the same class names/attribute shapes are used instead.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitor_baseline.adapters.moatless import (  # noqa: E402
    MoatlessLivePlanIntegration,
    _build_moatless_trajectory,
    _describe_moatless_action,
)
from monitor_baseline.bridge import LivePlanBoundaryMonitor  # noqa: E402
from monitor_baseline.moatless_monitor import (  # noqa: E402
    derive_node_monitor_state,
    invalidate_node_monitor_state,
)

_MOCK_REFINER_CONFIG = str(Path(__file__).resolve().parent / "fixtures" / "mock_refiner.yaml")


def make_action(class_name: str, *, title: str | None = None, **fields):
    """Build a stand-in object with the given class name and attributes,
    including a real-shaped `model_dump()` so it behaves like an actual
    pydantic `ActionArguments` for `_describe_moatless_action`. Pass `title`
    to also stand in for the `ConfigDict(title=...)` some real action
    classes set (their exposed tool name, which can differ from the Python
    class name -- e.g. `GlobArgs` -> "GlobTool")."""

    stub_cls = type(class_name, (SimpleNamespace,), {})
    if title is not None:
        stub_cls.model_config = {"title": title}
    obj = stub_cls(**fields)

    def _model_dump(*, exclude=None, exclude_none=False):
        exclude = exclude or set()
        return {
            key: value
            for key, value in fields.items()
            if key not in exclude and not (exclude_none and value is None)
        }

    obj.model_dump = _model_dump
    return obj


class FakeActionStep:
    def __init__(self, action, observation=None):
        self.action = action
        self.observation = observation
        self.phase_monitor: dict = {}


class FakeNode:
    """Minimal stand-in for moatless `Node`: a single-branch parent chain."""

    def __init__(self, node_id, parent=None, action_steps=None, assistant_message: str = ""):
        self.node_id = node_id
        self.parent = parent
        self.action_steps = action_steps or []
        self.assistant_message = assistant_message
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


# -- Fix 1: persist_overhead must exist and behave like the reference adapter's --


def test_persist_overhead_stores_payload_on_trajectory_root():
    monitor = LivePlanBoundaryMonitor(variant="monitor")
    integration = MoatlessLivePlanIntegration(monitor)

    root = FakeNode(node_id=0)
    child = FakeNode(node_id=1, parent=root)
    grandchild = FakeNode(node_id=2, parent=child)

    integration.persist_overhead(grandchild)

    # `task_elapsed_seconds` is wall-clock and ticks between the two
    # `payload()` calls, so compare the stable parts instead of the whole dict.
    assert root.overhead["schema_version"] == monitor.overhead.payload()["schema_version"]
    assert root.overhead["events"] == monitor.overhead.payload()["events"]
    assert not hasattr(child, "overhead")
    assert not hasattr(grandchild, "overhead")


def test_persist_overhead_on_root_itself():
    monitor = LivePlanBoundaryMonitor(variant="monitor")
    integration = MoatlessLivePlanIntegration(monitor)

    root = FakeNode(node_id=0)
    integration.persist_overhead(root)

    assert root.overhead["schema_version"] == monitor.overhead.payload()["schema_version"]
    assert root.overhead["events"] == monitor.overhead.payload()["events"]


# -- Fix 2: cache must be invalidated once a node's own action executes --


def test_derive_node_monitor_state_caches_stale_result_without_invalidation():
    action = make_action("GrepToolArgs", pattern="foo", thoughts="looking")
    step = FakeActionStep(action, observation=None)
    node = FakeNode(node_id=1, action_steps=[step])

    state_before = derive_node_monitor_state(node)
    assert state_before.step_index == 0
    assert state_before.monitor.role_history == []

    # The action now executes.
    step.observation = SimpleNamespace(message="found nothing")

    # Without invalidation, the stale cached object is returned unconditionally.
    state_stale = derive_node_monitor_state(node)
    assert state_stale is state_before
    assert state_stale.step_index == 0
    assert state_stale.monitor.role_history == []


def test_invalidate_node_monitor_state_forces_fresh_replay_after_execution():
    action = make_action("GrepToolArgs", pattern="foo", thoughts="looking")
    step = FakeActionStep(action, observation=None)
    node = FakeNode(node_id=1, action_steps=[step])

    state_before = derive_node_monitor_state(node)
    assert state_before.step_index == 0

    step.observation = SimpleNamespace(message="found nothing")
    invalidate_node_monitor_state(node)
    state_after = derive_node_monitor_state(node)

    assert state_after is not state_before
    assert state_after.step_index == 1
    assert state_after.monitor.role_history == ["L_navigate"]


def test_invalidate_node_monitor_state_is_a_noop_when_nothing_cached():
    node = FakeNode(node_id=1)
    invalidate_node_monitor_state(node)  # must not raise


# -- Fix 2, adapter-level: refiner's deferred non-blocking call must see the
# just-executed step, not the stale pre-execution cache. --


def test_refiner_variant_sees_just_executed_step_after_invalidation():
    monitor = LivePlanBoundaryMonitor(
        variant="refiner", refiner_config=_MOCK_REFINER_CONFIG, issue_description="Fix the bug"
    )
    integration = MoatlessLivePlanIntegration(monitor)

    captured_inputs: list = []
    original_refine_plan = monitor._refiner.refine_plan

    def _capturing_refine_plan(refiner_input, current_step):
        captured_inputs.append(refiner_input)
        return original_refine_plan(refiner_input, current_step)

    monitor._refiner.refine_plan = _capturing_refine_plan

    prior_steps = [
        FakeActionStep(
            make_action(
                "GrepToolArgs",
                pattern=f"term-{index}",
                thoughts="looking around",
            ),
            observation=SimpleNamespace(message=f"result-{index}"),
        )
        for index in range(6)
    ]
    action = make_action("GrepToolArgs", pattern="term-6", thoughts="looking around")
    step = FakeActionStep(action, observation=None)
    node = FakeNode(node_id=1, action_steps=[*prior_steps, step])

    decision = asyncio.run(integration.inspect_before_action(agent=None, node=node, action_step=step))
    assert decision.execute_proposed_step is True
    # The seventh navigate step triggers the non-blocking dwell rule, but the
    # refiner must not be queried until that action's observation exists.
    assert not captured_inputs

    # `_execute()` runs and sets the observation, then swe-agent's own
    # post-action hook calls `handle_after_action`.
    step.observation = SimpleNamespace(message="grep found nothing new")
    asyncio.run(integration.handle_after_action(agent=None, node=node, action_step=step))

    assert captured_inputs, "handle_after_action should have invoked the deferred refiner call"
    state_summary = captured_inputs[-1].state_summary
    assert state_summary.step_count == 7
    assert state_summary.phase_history == ["L_navigate"] * 7


def test_refiner_variant_does_not_wake_on_a_clear_action():
    monitor = LivePlanBoundaryMonitor(
        variant="refiner", refiner_config=_MOCK_REFINER_CONFIG, issue_description="Fix the bug"
    )
    integration = MoatlessLivePlanIntegration(monitor)
    captured_inputs: list = []
    original_refine_plan = monitor._refiner.refine_plan

    def _capturing_refine_plan(refiner_input, current_step):
        captured_inputs.append(refiner_input)
        return original_refine_plan(refiner_input, current_step)

    monitor._refiner.refine_plan = _capturing_refine_plan
    step = FakeActionStep(make_action("GrepToolArgs", pattern="first"))
    node = FakeNode(node_id=1, action_steps=[step])

    decision = asyncio.run(
        integration.inspect_before_action(agent=None, node=node, action_step=step)
    )
    assert decision.execute_proposed_step is True
    assert step.phase_monitor["guidance_pending"] is False

    step.observation = SimpleNamespace(message="first result")
    asyncio.run(integration.handle_after_action(agent=None, node=node, action_step=step))

    assert captured_inputs == []
    assert node.monitor_events == []


def test_refiner_only_guidance_is_visible_in_next_observation():
    monitor = LivePlanBoundaryMonitor(
        variant="refiner_only",
        refiner_config=_MOCK_REFINER_CONFIG,
        issue_description="Fix the bug",
    )
    integration = MoatlessLivePlanIntegration(monitor)
    step = FakeActionStep(
        make_action("GrepToolArgs", pattern="foo", thoughts="looking"),
        observation=SimpleNamespace(message="result"),
    )
    node = FakeNode(node_id=1, action_steps=[step])

    asyncio.run(integration.handle_after_action(agent=None, node=node, action_step=step))

    assert "External planning guidance for the next step:" in step.observation.message
    assert node.monitor_events[0]["event_type"] == "monitor_guidance"


def test_refiner_only_checks_cadence_once_per_multi_action_node():
    """Moatless can put several actions in one model turn (one node). The
    `refiner_only` cadence counts *steps* (nodes), matching mini-swe-agent's
    "every N steps" -- `agent.py` calls `handle_after_action` once per action
    inside the node, so without a last-action guard a two-action node would
    advance the counter by two and could double-fire the refiner."""

    monitor = LivePlanBoundaryMonitor(
        variant="refiner_only",
        refiner_config=_MOCK_REFINER_CONFIG,
        issue_description="Fix the bug",
    )
    integration = MoatlessLivePlanIntegration(monitor)
    calls: list[int] = []
    original_refine_plan = monitor._refiner.refine_plan

    def _counting_refine_plan(refiner_input, current_step):
        calls.append(current_step)
        return original_refine_plan(refiner_input, current_step)

    monitor._refiner.refine_plan = _counting_refine_plan

    first_action = FakeActionStep(
        make_action("GrepToolArgs", pattern="foo"),
        observation=SimpleNamespace(message="first-result"),
    )
    second_action = FakeActionStep(
        make_action("ReadFileArgs", file_path="a.py"),
        observation=SimpleNamespace(message="second-result"),
    )
    node = FakeNode(node_id=1, action_steps=[first_action, second_action])

    asyncio.run(
        integration.handle_after_action(agent=None, node=node, action_step=first_action)
    )
    asyncio.run(
        integration.handle_after_action(agent=None, node=node, action_step=second_action)
    )

    # Exactly one refiner call for the whole node (its one step), not two.
    assert calls == [1]
    assert "External planning guidance for the next step:" not in (
        first_action.observation.message or ""
    )
    assert "External planning guidance for the next step:" in second_action.observation.message


def test_refiner_only_cadence_and_context_are_branch_local():
    monitor = LivePlanBoundaryMonitor(
        variant="refiner_only",
        refiner_config=_MOCK_REFINER_CONFIG,
        issue_description="Fix the bug",
    )
    integration = MoatlessLivePlanIntegration(monitor)
    captured_inputs: list = []
    original_refine_plan = monitor._refiner.refine_plan

    def _capturing_refine_plan(refiner_input, current_step):
        captured_inputs.append((refiner_input, current_step))
        return original_refine_plan(refiner_input, current_step)

    monitor._refiner.refine_plan = _capturing_refine_plan
    root = FakeNode(node_id=0)
    first_step = FakeActionStep(
        make_action("GrepToolArgs", pattern="branch-a"),
        observation=SimpleNamespace(message="a-result"),
    )
    first_branch = FakeNode(node_id=1, parent=root, action_steps=[first_step])
    asyncio.run(
        integration.handle_after_action(
            agent=None, node=first_branch, action_step=first_step
        )
    )

    second_step = FakeActionStep(
        make_action("GrepToolArgs", pattern="branch-b"),
        observation=SimpleNamespace(message="b-result"),
    )
    sibling_branch = FakeNode(node_id=2, parent=root, action_steps=[second_step])
    asyncio.run(
        integration.handle_after_action(
            agent=None, node=sibling_branch, action_step=second_step
        )
    )

    assert [current_step for _, current_step in captured_inputs] == [1, 1]
    sibling_input = captured_inputs[-1][0]
    assert sibling_input.last_refinement_step is None
    assert sibling_input.last_refinement_guidance is None


def test_rule_only_nonblocking_guidance_is_delivered_after_successful_action():
    monitor = LivePlanBoundaryMonitor(variant="monitor")
    integration = MoatlessLivePlanIntegration(monitor)
    prior = [
        FakeActionStep(
            make_action("GrepToolArgs", pattern=f"term-{index}", thoughts="looking"),
            observation=SimpleNamespace(message=f"result-{index}"),
        )
        for index in range(6)
    ]
    proposed = FakeActionStep(
        make_action("GrepToolArgs", pattern="term-6", thoughts="still looking")
    )
    node = FakeNode(node_id=1, action_steps=[*prior, proposed])

    decision = asyncio.run(
        integration.inspect_before_action(agent=None, node=node, action_step=proposed)
    )
    assert decision.execute_proposed_step is True
    assert proposed.phase_monitor["matched_rule_ids"] == ["dwell_L_navigate"]
    assert proposed.phase_monitor["guidance_pending"] is True
    assert node.monitor_events == []

    proposed.observation = SimpleNamespace(message="latest result")
    asyncio.run(integration.handle_after_action(agent=None, node=node, action_step=proposed))

    assert "External planning guidance for the next step:" in proposed.observation.message
    assert proposed.phase_monitor["guidance_delivered"] is True
    assert node.monitor_events[0]["metadata"]["guidance_source"] == "rule_match"


def test_structured_self_loop_is_blocked_and_recorded_with_phase_metadata():
    monitor = LivePlanBoundaryMonitor(variant="monitor")
    integration = MoatlessLivePlanIntegration(monitor)
    prior = FakeActionStep(
        make_action("GrepToolArgs", pattern="same", thoughts="first"),
        observation=SimpleNamespace(message="first result"),
    )
    proposed = FakeActionStep(
        make_action("GrepToolArgs", pattern="same", thoughts="again")
    )
    node = FakeNode(node_id=1, action_steps=[prior, proposed])

    decision = asyncio.run(
        integration.inspect_before_action(agent=None, node=node, action_step=proposed)
    )

    assert decision.execute_proposed_step is False
    assert "oscillation_self_loop" in proposed.phase_monitor["matched_rule_ids"]
    assert node.action_steps == [prior]
    event = node.monitor_events[-1]
    assert event["metadata"]["classified_phase"] == "L_navigate"
    assert "oscillation_self_loop" in event["metadata"]["matched_rule_ids"]


def test_refined_plan_phases_are_replayed_only_on_their_branch():
    root = FakeNode(node_id=0)
    branch_with_plan = FakeNode(node_id=1, parent=root)
    branch_with_plan.monitor_events = [
        {
            "event_type": "monitor_guidance",
            "message": "follow the revised plan",
            "metadata": {
                "guidance_source": "refiner",
                "refinement_step": 1,
                "refined_plan_phases": ["P", "V_regression_test"],
            },
        }
    ]
    sibling = FakeNode(node_id=2, parent=root)

    planned_state = derive_node_monitor_state(branch_with_plan)
    sibling_state = derive_node_monitor_state(sibling)
    planned_rule = planned_state.monitor.rule_engine.rules["plan_compliance"]
    sibling_rule = sibling_state.monitor.rule_engine.rules["plan_compliance"]

    assert [str(phase) for phase in planned_rule.intended_plan] == [
        "P",
        "V_regression_test",
    ]
    assert [str(phase) for phase in sibling_rule.intended_plan] != [
        "P",
        "V_regression_test",
    ]


def test_action_failure_cancels_branch_local_pending_refiner_trigger():
    monitor = LivePlanBoundaryMonitor(
        variant="refiner", refiner_config=_MOCK_REFINER_CONFIG
    )
    integration = MoatlessLivePlanIntegration(monitor)
    prior_steps = [
        FakeActionStep(
            make_action("GrepToolArgs", pattern=f"term-{index}"),
            observation=SimpleNamespace(message=f"result-{index}"),
        )
        for index in range(6)
    ]
    step = FakeActionStep(
        make_action("GrepToolArgs", pattern="foo", thoughts="looking")
    )
    node = FakeNode(node_id=1, action_steps=[*prior_steps, step])

    asyncio.run(integration.inspect_before_action(agent=None, node=node, action_step=step))
    assert id(step) in integration._pending_refiner_triggers

    asyncio.run(
        integration.handle_action_failure(
            agent=None,
            node=node,
            action_step=step,
            error=RuntimeError("boom"),
        )
    )
    assert id(step) not in integration._pending_refiner_triggers
    assert step.phase_monitor["post_action_monitor_cancellation"]["reason"] == "action_failed"


# -- Fix 3 (Codex review, 2026-09-06): the reviewer's trajectory must keep an
# action's arguments, not just its class name, or it cannot tell what a
# `StringReplace` actually searched for/changed, or whether two edits are
# really the same one repeated. --


def test_describe_moatless_action_keeps_arguments_not_just_class_name():
    action = make_action(
        "StringReplaceArgs", title="StringReplace", path="x.py", old_str="return 1", new_str="return 2"
    )

    text = _describe_moatless_action(action)

    assert text == "StringReplace(path='x.py', old_str='return 1', new_str='return 2')"


def test_describe_moatless_action_uses_the_declared_title_not_the_class_name():
    # e.g. real `GlobArgs` declares `ConfigDict(title="GlobTool")` -- the
    # exposed tool name differs from the python class name.
    action = make_action("GlobArgs", title="GlobTool", pattern="**/*.py")

    assert _describe_moatless_action(action).startswith("GlobTool(")


def test_describe_moatless_action_falls_back_to_class_name_without_model_dump():
    action = SimpleNamespace()  # no model_dump, no model_config -- not a real action at all

    assert _describe_moatless_action(action) == "SimpleNamespace" or "SimpleNamespace" in _describe_moatless_action(
        action
    )


# -- Fix 4 (Codex review, 2026-09-06): `to_prompt()` is each action's own
# short hand-written summary and is not guaranteed to mention every field
# that affects behavior -- e.g. real `SemanticSearchArgs.to_prompt()` never
# mentions `category`, so two searches that differ only by category render
# identically to the reviewer. Reading the model's real field values instead
# keeps this correct without trusting each action's prose to be complete. --


def test_describe_moatless_action_keeps_fields_to_prompt_would_have_dropped():
    # Mirrors the real bug: `SemanticSearchArgs.to_prompt()` renders only the
    # query, never `category`, so two runs differing only in category would
    # look identical to the reviewer and could be mistaken for a repeat.
    implementation_search = make_action(
        "SemanticSearchArgs", title="SemanticSearch", query="password hashing", category="implementation"
    )
    test_search = make_action("SemanticSearchArgs", title="SemanticSearch", query="password hashing", category="test")

    assert _describe_moatless_action(implementation_search) != _describe_moatless_action(test_search)
    assert "category='implementation'" in _describe_moatless_action(implementation_search)
    assert "category='test'" in _describe_moatless_action(test_search)


def test_build_moatless_trajectory_preserves_action_arguments():
    action = make_action(
        "StringReplaceArgs", title="StringReplace", path="x.py", old_str="return 1", new_str="return 2"
    )
    step = FakeActionStep(action, observation=SimpleNamespace(message="No match found for old_str"))
    node = FakeNode(node_id=1, action_steps=[step])

    trajectory = _build_moatless_trajectory(node)

    assert len(trajectory) == 1
    assert "old_str='return 1'" in trajectory[0].action
    assert "StringReplaceArgs" != trajectory[0].action

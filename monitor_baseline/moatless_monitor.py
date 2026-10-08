"""Per-node phase-monitor state derivation for moatless's branching `SearchTree`.

A single global `StatefulPhaseMonitor` would conflate sibling branches into
one fake dwell/oscillation/phase history, since moatless's `SearchTree`/
`Node` is a tree, not a linear trajectory (see
`monitor_runtime/modal_moatless_runner.py`). Instead, each node's monitor
state is derived by deterministic replay of the root-to-node path of
already-*executed* actions (i.e. `action_step.observation is not None`)
through a fresh `StatefulPhaseMonitor`, then cached on the node so sibling
branches never share mutable state and repeated calls for the same node are
cheap.

This module reuses `StatefulPhaseMonitor` as its role-history/rule-engine
container but drives it with `_replay_role_step`, a lean re-implementation of
STEP 3/4 of `plan_monitor/monitor.py::StatefulPhaseMonitor.on_step` (role
history update, phase-transition/plan-compliance checks, then one
rule-engine evaluation per step) -- without the pre-emptive backup/rollback
machinery `check_step_pre_emptively` needs, because a pure replay of already
-committed history never needs to be rolled back. `on_step`/
`check_step_pre_emptively` themselves are not reusable as-is here because
they always derive roles by parsing bash command text via `CommandParser`;
moatless supplies the role directly from `moatless_role_classifier` instead
(per the user's original choice to bypass the parser entirely for moatless).

For graph-based rules, structured Moatless actions are represented directly
as graph nodes whose identity is their canonical action payload.  This keeps
LivePlan's self-loop and multi-node-cycle methodology without pretending that
Moatless tool calls are shell commands or routing them through CommandParser.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

from plan_monitor.monitor import StatefulPhaseMonitor
from plan_monitor.phases import MonitorResult, Phase

from .moatless_role_classifier import classify_moatless_action, is_ending_action

_CACHE_ATTR = "_monitor_baseline_state"


@dataclass
class NodeMonitorState:
    monitor: StatefulPhaseMonitor
    last_result: Optional[MonitorResult]
    step_index: int

    @property
    def current_phase(self):
        return self.monitor.get_current_phase()

    def should_block(self) -> bool:
        return bool(self.last_result and self.last_result.should_block_and_refine())

    def messages(self) -> list[str]:
        return self.last_result.get_all_messages() if self.last_result else []


def derive_node_monitor_state(node: Any, *, rules_config: Optional[str] = None) -> NodeMonitorState:
    """Return the (cached) monitor state for `node`'s root-to-node path.

    Only action steps with a non-`None` observation (i.e. already executed)
    are replayed, so the returned state never includes a still-pending
    proposal. Cached on the node instance itself, keyed implicitly by object
    identity, so distinct sibling nodes never share state.
    """

    cached = getattr(node, _CACHE_ATTR, None)
    if cached is not None:
        return cached

    monitor = StatefulPhaseMonitor(rules_config=rules_config)
    last_result: Optional[MonitorResult] = None
    step_index = 0

    for trajectory_node in node.get_trajectory():
        for action_step in trajectory_node.action_steps:
            if action_step.observation is None:
                continue
            step_index += 1
            action = action_step.action
            thought = getattr(action, "thoughts", None) or trajectory_node.assistant_message or ""
            command_text = _describe_action(action)
            _append_structured_graph_step(monitor, action, command_text, step_index)

            if is_ending_action(action):
                last_result = _merge_results(
                    _check_ending(monitor, step_index),
                    _check_structured_oscillation(monitor, step_index),
                )
                continue

            role = classify_moatless_action(
                action,
                has_patch=_has_prior_patch(monitor),
                created_tests=monitor.created_tests,
            )
            if role == "general":
                last_result = _check_structured_oscillation(monitor, step_index)
                continue
            last_result = _replay_role_step(monitor, role, command_text, thought, step_index)

        _apply_refined_plan_events(monitor, trajectory_node)

    state = NodeMonitorState(monitor=monitor, last_result=last_result, step_index=step_index)
    try:
        setattr(node, _CACHE_ATTR, state)
    except Exception:
        pass
    return state


def invalidate_node_monitor_state(node: Any) -> None:
    """Drop the cached monitor state for `node` so the next `derive_node_monitor_state`
    call re-replays its (now possibly changed) action_steps from scratch."""
    try:
        if hasattr(node, _CACHE_ATTR):
            delattr(node, _CACHE_ATTR)
    except Exception:
        pass


def evaluate_node_action(
    node: Any,
    action_step: Any,
    *,
    rules_config: Optional[str] = None,
) -> Optional[MonitorResult]:
    """Evaluate a newly proposed (not-yet-executed) `action_step` against `node`'s state.

    Operates on a deep copy of the cached monitor so a proposal that gets
    blocked (and is never actually executed/committed) cannot corrupt the
    cached state that later calls -- or a later child node's replay -- would
    read.
    """

    parent_state = derive_node_monitor_state(node, rules_config=rules_config)
    monitor = copy.deepcopy(parent_state.monitor)
    step_index = parent_state.step_index + 1
    action = action_step.action
    thought = getattr(action, "thoughts", None) or getattr(node, "assistant_message", None) or ""
    command_text = _describe_action(action)
    _append_structured_graph_step(monitor, action, command_text, step_index)

    if is_ending_action(action):
        return _merge_results(
            _check_ending(monitor, step_index),
            _check_structured_oscillation(monitor, step_index),
        )

    role = classify_moatless_action(
        action,
        has_patch=_has_prior_patch(monitor),
        created_tests=monitor.created_tests,
    )
    if role == "general":
        return _check_structured_oscillation(monitor, step_index)
    result = _replay_role_step(monitor, role, command_text, thought, step_index)
    # Match StatefulPhaseMonitor.check_step_pre_emptively(): phase/state
    # updates with no rule match are committed during trajectory replay, but a
    # clear proposal returns None and must not wake the full LivePlan refiner.
    return result if result.rule_matches else None


def _has_prior_patch(monitor: StatefulPhaseMonitor) -> bool:
    return any(role == "P" for role in monitor.role_history)


def _describe_action(action: Any) -> str:
    name = type(action).__name__
    for field in (
        "command",
        "path",
        "file_path",
        "script_path",
        "pattern",
        "query",
        "class_name",
        "function_name",
        "code_snippet",
    ):
        value = getattr(action, field, None)
        if isinstance(value, str) and value:
            return f"{name}:{value}"
    return name


def _canonical_action_payload(action: Any) -> str:
    """Return a stable identity for one structured Moatless action."""

    dumper = getattr(action, "model_dump", None)
    if callable(dumper):
        try:
            payload = dumper(mode="json", exclude_none=True)
        except TypeError:
            payload = dumper(exclude_none=True)
        if isinstance(payload, dict):
            # Thoughts and self-reported phase are context, not action identity;
            # LivePlan's shell graph likewise hashes tool/arguments only.
            payload = {
                key: value
                for key, value in payload.items()
                if key not in {"thoughts", "phase"}
            }
    else:
        payload = {
            key: value
            for key, value in vars(action).items()
            if key not in {"thoughts", "phase"} and not key.startswith("_")
        }
    return json.dumps(
        {"action": type(action).__name__, "arguments": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _append_structured_graph_step(
    monitor: StatefulPhaseMonitor,
    action: Any,
    command_text: str,
    step_index: int,
) -> None:
    """Add one structured action to LivePlan's graph-rule input shape."""

    builder = getattr(monitor, "graph_builder", None)
    if builder is None:
        return
    canonical = _canonical_action_payload(action)
    node_key = "moatless:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    graph = builder.G
    if node_key in graph:
        graph.nodes[node_key].setdefault("step_indices", []).append(step_index)
    else:
        graph.add_node(
            node_key,
            label=type(action).__name__,
            args=canonical,
            flags={},
            phases=[],
            step_indices=[step_index],
            thoughts=[],
            observations=[],
            outcome=[],
            full_action=command_text,
        )
    if builder.previous_node is not None:
        graph.add_edge(builder.previous_node, node_key, type="exec", step=step_index)
    builder.previous_node = node_key
    builder.node_signature_to_key[canonical] = node_key
    # ``PlanRefiner.build_state_summary_from_monitor`` exposes this value as
    # "Total Steps Executed".  Moatless supplies real 1-based trajectory
    # indices, so retain that semantic instead of GraphBuilder's internal
    # next-index convention.
    monitor.step_counter = max(monitor.step_counter, step_index)


def _check_structured_oscillation(
    monitor: StatefulPhaseMonitor, step_index: int
) -> Optional[MonitorResult]:
    if not (monitor.enable_rules and monitor.rule_engine and monitor.graph_builder):
        return None
    rule = monitor.rule_engine.rules.get("oscillation")
    if not rule or not rule.enabled:
        return None
    match = rule.check(graph=monitor.graph_builder.G, step_index=step_index)
    if not match:
        return None
    match.step_index = step_index
    return MonitorResult(
        current_phase=monitor.current_phase,
        phase_changed=False,
        category_changed=False,
        previous_phase=monitor.previous_phase,
        rule_matches=[match],
    )


def _merge_results(
    primary: Optional[MonitorResult], secondary: Optional[MonitorResult]
) -> Optional[MonitorResult]:
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    primary.rule_matches.extend(secondary.rule_matches)
    return primary


def _apply_refined_plan_events(monitor: StatefulPhaseMonitor, node: Any) -> None:
    """Replay branch-local intended-plan updates emitted by the refiner."""

    for event in getattr(node, "monitor_events", []):
        if not isinstance(event, dict):
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            continue
        phases = metadata.get("refined_plan_phases")
        if phases:
            monitor.update_intended_plan(list(phases))


def _replay_role_step(
    monitor: StatefulPhaseMonitor,
    role: str,
    command_text: str,
    thought: str,
    step_index: int,
) -> MonitorResult:
    """Lean re-implementation of `on_step` STEP 3/4 for one already-classified role."""

    role_history_len_before = len(monitor.role_history)
    monitor.role_history.append(role)
    new_phase = Phase(role)

    all_rule_matches: list = []
    prev_phase_for_transition = None
    phase_changed = False
    if monitor.current_phase is None:
        monitor.current_phase = new_phase
        phase_changed = True
    elif new_phase != monitor.current_phase:
        prev_phase_for_transition = monitor.current_phase
        monitor.previous_phase = monitor.current_phase
        monitor.current_phase = new_phase
        phase_changed = True

    if phase_changed and monitor.enable_rules and monitor.rule_engine:
        plan_rule = monitor.rule_engine.rules.get("plan_compliance")
        if plan_rule and plan_rule.enabled:
            plan_match = plan_rule.check(
                current_phase=new_phase,
                previous_phase=prev_phase_for_transition,
                step_index=step_index,
                command=command_text,
                role_history=monitor.role_history[:role_history_len_before],
            )
            if plan_match:
                all_rule_matches.append(plan_match)

    phase_transition = monitor.previous_phase if phase_changed else None
    if monitor.enable_rules and monitor.rule_engine:
        rule_matches = monitor.rule_engine.evaluate(
            current_phase=new_phase,
            previous_phase=phase_transition,
            graph=monitor.graph_builder.G if monitor.graph_builder else None,
            step_index=step_index,
            command=command_text,
            outcome=None,
            role_history=monitor.role_history[:role_history_len_before],
            check_repeated_view=True,
            skip_plan_compliance=True,
        )
        if rule_matches:
            all_rule_matches.extend(rule_matches)

    return MonitorResult(
        current_phase=new_phase,
        phase_changed=phase_changed,
        category_changed=False,
        previous_phase=monitor.previous_phase,
        rule_matches=all_rule_matches,
    )


def _check_ending(monitor: StatefulPhaseMonitor, step_index: int) -> Optional[MonitorResult]:
    """Ending-flag equivalent of `monitor.py` STEP-1's `role == "general"` branch.

    `plan_compliance.ending_flags` (default config) checks for the literal
    substring `"submit"`, so moatless's Finish/VerifiedFinish/Reject actions
    are described with a synthetic `"submit"` command string here to trigger
    the same "did we skip required phases before ending" check SWE-agent's
    own `submit` command triggers upstream.
    """

    if not (monitor.enable_rules and monitor.rule_engine):
        return None
    plan_rule = monitor.rule_engine.rules.get("plan_compliance")
    if not plan_rule or not plan_rule.enabled:
        return None
    plan_match = plan_rule.check(
        current_phase=None,
        previous_phase=monitor.current_phase,
        step_index=step_index,
        command="submit",
    )
    if not plan_match:
        return None
    return MonitorResult(
        current_phase=monitor.current_phase,
        phase_changed=False,
        category_changed=False,
        previous_phase=monitor.previous_phase,
        rule_matches=[plan_match],
    )

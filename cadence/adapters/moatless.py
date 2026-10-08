from __future__ import annotations

from typing import Any, Optional

from monitor_runtime.integration import NativePhaseMonitorIntegration
from monitor_runtime.types import MonitorDecision, MonitorEvent

from ..monitor import CadenceMonitor
from monitor_runtime.adapters.moatless import _describe_moatless_action, _environment_observation, _executed_action_count, _moatless_task_context
from ..prompts_moatless import (
    MOATLESS_ACTION_BLOCK_TEMPLATE,
    MOATLESS_SYSTEM_PROMPT,
    format_moatless_trajectory,
)


class MoatlessCadenceIntegration(NativePhaseMonitorIntegration):
    """Adaptive Interval integration for moatless's `ActionAgent`, reviewing one
    model response (a full action-step group) atomically.

    moatless's own runtime already treats "one model response" as the unit a
    monitor should review: `ActionAgent.run` looks for an
    `inspect_before_action_group`/`handle_after_action_group` pair on the
    installed monitor and, when present, checks the *entire* proposed action
    group before any of it executes (see `coding_agent-baselines/moatless/
    moatless/agent/agent.py`, `group_checked` branch). That is exactly the
    unit `CadenceMonitor.before_action`/`after_action` already
    expect ("the proposed next action"), so this integration uses the group
    hooks rather than moatless's alternative per-action path -- a multi-action
    response is reviewed, discarded, or executed as one thing, never split.

    Branch-safety note (contrast with `MoatlessLivePlanIntegration`): that
    adapter derives each node's monitor state by replaying its own
    root-to-node path (`derive_node_monitor_state`), because moatless's
    `Node` tree can in principle branch and a single shared monitor instance
    would conflate sibling branches. This integration does NOT do that -- it
    keeps `CadenceMonitor`'s state (interval, guidance history,
    committed trajectory) on one instance shared across the whole run, reset
    only when a new root is seen. That is only correct because this project
    always configures moatless's `SearchTree` to run as a strict linear
    chain (`max_expansions=1`, `auto_expand_root=False`, `SimpleSelector` --
    see `.moatless/flows/swebench_claude.json`): with those settings every
    node has exactly one child, so there are never sibling branches to
    conflate. If that configuration ever changes to real tree search, this
    integration needs the same per-branch replay treatment as the rule-engine
    adapter before it can be trusted again.

    give_up (see `CadenceMonitor`'s docstring) is deliberately
    NOT wired to a deterministic forced submission here, unlike the
    mini-swe-agent adapter (which runs moatless's own submit command
    directly). moatless's termination path is a model-authored action
    (`VerifiedFinishArgs`/`FinishArgs`) that carries its own required fields
    (a finish reason, test verification) and is executed through the same
    machinery as any other action; fabricating one bypasses validation this
    integration hasn't verified is safe to skip. give_up is therefore treated
    like block here -- the group is discarded and the agent is told to stop
    investigating and finish now -- which is only a request the agent can
    ignore, same as an ordinary block, not a guaranteed saving: an agent that
    keeps proposing more investigation instead of finishing will keep
    burning budget exactly like it would under plain block. This is a known
    semantic gap relative to the mini-swe-agent adapter's give_up, not a
    weaker form of the same guarantee (Codex review, 2026-09-06).

    The reviewer prompt, proposed-action rendering, and trajectory-history
    rendering are all moatless-specific here (`cadence.prompts_moatless.MOATLESS_SYSTEM_PROMPT`/`MOATLESS_ACTION_BLOCK_TEMPLATE`/
    `format_moatless_trajectory`), not `adaptive_fixed_interval.py`'s module
    defaults: those default to mini-swe-agent's vocabulary (shell one-liners
    as the worked example, bash-fenced Thought/Action blocks for both the
    proposed action and every historical step), none of which means anything
    against moatless's structured tool calls. Swapping all three per-instance
    (rather than editing the shared module constants) keeps the mini-swe-agent
    prompt -- and the experiment results already validated against it --
    completely untouched.
    """

    def __init__(self, monitor: CadenceMonitor):
        super().__init__(monitor)
        monitor.system_prompt = MOATLESS_SYSTEM_PROMPT
        monitor.action_block_template = MOATLESS_ACTION_BLOCK_TEMPLATE
        monitor.format_trajectory = format_moatless_trajectory
        self._run_root: Any | None = None

    def _maybe_reset_for_root(self, node: Any) -> None:
        root = node
        while getattr(root, "parent", None) is not None:
            root = root.parent
        if self._run_root is not root:
            # `ActionAgent.run()` already calls `self.reset_run_state()` on
            # this exact new-root transition, before it ever reaches this
            # hook (see `coding_agent-baselines/moatless/moatless/agent/
            # agent.py`'s own `_realtime_monitor_root` check, which runs
            # ahead of action generation). Calling it again here would
            # `overhead.reset()` a second time and wipe the `agent_generation`
            # event that generation step just recorded -- confirmed by replay
            # (Codex review, 2026-09-06): the first model call of every case
            # silently disappeared from the ledger. This method only needs to
            # track root identity for the lazy `issue_description` fill
            # below, exactly like `MoatlessLivePlanIntegration._maybe_reset_for_root`
            # (which also does not re-reset).
            self._run_root = root

    async def inspect_before_action_group(
        self, agent: Any, node: Any, action_steps: list[Any]
    ) -> MonitorDecision:
        """Review one full model response before any of its actions execute."""

        self._maybe_reset_for_root(node)
        if not self.monitor.issue_description:
            self.monitor.issue_description = _moatless_task_context(node)

        step_index = self.monitor.executed_step_count
        step_id = str(step_index)
        command = _describe_proposed_group(action_steps)
        thought = _group_thought(node, action_steps)
        event_id = getattr(node, "overhead_generation_event_id", None)

        result = self.monitor.before_action(
            step_index, command, thought, accounting_step_id=step_id
        )

        self.associate_generation(event_id, phase=result.classified_phase, step_id=step_id)
        node.overhead_step_id = step_id
        for action_step in action_steps:
            action_step.phase_monitor = {
                **getattr(action_step, "phase_monitor", {}),
                "monitor_baseline_variant": self.monitor.variant,
                "execute_proposed_step": result.execute,
                "is_blocking": result.is_blocking,
                "liveplan_step_index": step_index,
                "refiner_config": self.monitor.refiner_config,
                "refiner_interval": self.monitor.refiner_interval,
            }

        if result.execute:
            self.mark_generation_outcome(event_id, "accepted")
            if result.message:
                # Non-blocking guidance is delivered once the group's own
                # observations exist -- see `handle_after_action_group`.
                node._adaptive_interval_pending_guidance = result.message
            return MonitorDecision(execute_proposed_step=True, reason="monitor_baseline")

        self.mark_generation_outcome(event_id, "discarded")
        event = MonitorEvent(
            event_id=f"monitor_baseline-{step_id}",
            message=result.message or "",
            event_type="monitor_intervention",
            discarded_proposed_step_id=step_id,
            verdict="intervene",
            metadata={
                "monitor_baseline_variant": self.monitor.variant,
                "liveplan_step_index": step_index,
                "refiner_config": self.monitor.refiner_config,
                "refiner_interval": self.monitor.refiner_interval,
                "force_submit_requested": result.force_submit,
            },
        )
        self._record_monitor_event(node, event)
        self._inject_intervention_group(node, event)
        return MonitorDecision(
            execute_proposed_step=False,
            monitor_event=event,
            reason="monitor_baseline_blocked",
        )

    async def handle_after_action_group(
        self, agent: Any, node: Any, action_steps: list[Any]
    ) -> None:
        """Post-action hook: run once, after every action in the group has executed."""

        self._maybe_reset_for_root(node)
        pending_guidance = getattr(node, "_adaptive_interval_pending_guidance", None)
        node._adaptive_interval_pending_guidance = None

        step_index = self.monitor.executed_step_count
        step_id = str(step_index)
        command = _describe_proposed_group(action_steps)
        thought = _group_thought(node, action_steps)
        observation = _describe_executed_group(action_steps)
        self.monitor.after_action(
            step_index, command, thought, observation, executed=True, accounting_step_id=step_id
        )

        if pending_guidance:
            self._append_guidance_to_group_observation(action_steps, pending_guidance)
            self._record_guidance_event(node, pending_guidance)

    def persist_overhead(self, node: Any) -> None:
        """Store one cumulative ledger on the trajectory root, avoiding per-node duplication."""

        root = node
        while getattr(root, "parent", None) is not None:
            root = root.parent
        root.overhead = self.overhead_payload()

    def _record_monitor_event(self, node: Any, event: MonitorEvent) -> None:
        record = self.event_record(event)
        record["after_action_step_count"] = _executed_action_count(node)
        node.monitor_events = [*node.monitor_events, record]

    def _record_guidance_event(self, node: Any, message: str) -> None:
        record = {
            "event_type": "monitor_guidance",
            "event_id": f"monitor_baseline-guidance-{getattr(node, 'node_id', id(node))}-{len(node.monitor_events)}",
            "message": message,
            "verdict": "advise",
            "metadata": {"blocking": False},
            "after_action_step_count": _executed_action_count(node),
        }
        node.monitor_events = [*node.monitor_events, record]

    def _inject_intervention_group(self, node: Any, event: MonitorEvent) -> None:
        """Discard the whole proposed group and prepare the node for regeneration.

        Called from `inspect_before_action_group`, before any action in the
        group has executed, so there is nothing to preserve -- the entire
        proposal is dropped, mirroring `MoatlessLivePlanIntegration._inject_intervention`
        but for the whole group rather than a single action.
        """

        self.note_intervention(event.event_id)
        node.action_steps = []
        node.assistant_message = None
        node.thoughts = None
        node.thinking_blocks = None
        node.user_message = (
            "Your previous action was not executed. Follow this guidance and try again.\n\n"
            f"{event.message}"
        )

    @staticmethod
    def _append_guidance_to_group_observation(action_steps: list[Any], message: str) -> None:
        """Put advice in the last action's observation, the context moatless
        actually serializes for the next generation."""

        for action_step in reversed(action_steps):
            observation = getattr(action_step, "observation", None)
            if observation is None:
                continue
            original = str(getattr(observation, "message", "") or "")
            guidance = "External planning guidance for the next step:\n" + message
            observation.message = f"{original}\n\n{guidance}" if original else guidance
            return
        raise RuntimeError("Cannot deliver Adaptive Interval guidance without an observation")


def _describe_proposed_group(action_steps: list[Any]) -> str:
    """Render every not-yet-executed action in the group as one command block."""

    lines = [_describe_moatless_action(step.action) for step in action_steps]
    return "\n\n".join(lines)


def _describe_executed_group(action_steps: list[Any]) -> str:
    """Render every executed action's observation as one combined observation."""

    parts = []
    for step in action_steps:
        observation = getattr(step, "observation", None)
        if observation is None:
            continue
        message = _environment_observation(getattr(observation, "message", None) or "")
        if message:
            parts.append(message)
    return "\n\n".join(parts)


def _group_thought(node: Any, action_steps: list[Any]) -> str:
    for step in action_steps:
        thoughts = getattr(step.action, "thoughts", None)
        if thoughts:
            return str(thoughts)
    return str(getattr(node, "assistant_message", "") or "")

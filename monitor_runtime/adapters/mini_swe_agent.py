from __future__ import annotations

from typing import Any, Optional

from monitor_runtime.integration import NativePhaseMonitorIntegration
from monitor_runtime.types import MonitorDecision, MonitorEvent



class MiniSWEAgentMonitorIntegration(NativePhaseMonitorIntegration):
    """Shared pre-/post-action integration for mini-swe-agent's DefaultAgent.

    Ownership of non-blocking guidance (see the monitor's `before_action` result contract): `inspect_before_action` stashes non-blocking
    guidance text on `action["_monitor_baseline_pending_guidance"]`. The
    small post-action hook installed in `execute_actions` (right after
    `env.execute(action)`, once the observation is known) calls
    `handle_after_action(agent, message, action, result)`, which reads that
    stashed text (or the periodic `refiner_only` guidance from
    `bridge.after_action`) and returns whichever is present; the hook then
    appends it as the next user turn. This adapter -- not the bridge --
    owns injecting the message into agent history.
    """

    def __init__(self, monitor: Any):
        super().__init__(monitor)
        self._proposal_counter_int = 0

    def reset_run_state(self) -> None:
        self._proposal_counter_int = 0
        super().reset_run_state()

    def validate_action_batch(
        self, agent, message: dict[str, Any], actions: list[dict[str, Any]]
    ) -> None:
        """Require the single-action turn assumed by upstream LivePlan orchestration."""

        if len(actions) == 1:
            return

        # Keep mini-swe-agent optional when the shared baseline package is
        # imported from a SWE-agent-only or Moatless environment.
        from minisweagent.exceptions import FormatError

        event_id = message.get("extra", {}).get("overhead_event_id")
        self.mark_generation_outcome(event_id, "discarded_format")
        if agent.messages and agent.messages[-1] is message:
            agent.messages.pop()
        raise FormatError(
            agent.model.format_message(
                role="user",
                content=(
                    "LivePlan requires exactly one tool action per response. "
                    "Choose the single next action to execute, then wait for its observation "
                    "before proposing another action."
                ),
                extra={
                    "interrupt_type": "FormatError",
                    "monitor_baseline_batch_rejected": True,
                },
            )
        )

    def inspect_before_action(
        self, agent, message: dict[str, Any], action: dict[str, Any], action_index: int = 0
    ) -> MonitorDecision:
        """Inspect one parsed action immediately before `env.execute()`."""

        if not self.monitor.issue_description:
            self.monitor.issue_description = _mini_task_context(agent)

        self._proposal_counter_int += 1
        proposal_id = str(self._proposal_counter_int)
        step_index = self.monitor.executed_step_count
        command = action.get("command", "")
        thought = _stringify_content(message.get("content", ""))
        result = self.monitor.before_action(
            step_index,
            command,
            thought,
            accounting_step_id=proposal_id,
        )

        self.associate_generation(
            message.get("extra", {}).get("overhead_event_id"),
            phase=result.classified_phase,
            step_id=proposal_id,
        )
        message.setdefault("extra", {})["overhead_step_id"] = proposal_id

        action["_monitor_baseline_step_index"] = step_index
        action["_monitor_baseline_proposal_id"] = proposal_id
        execution_metadata = {
            key: value for key, value in {
                "judged_verdict": result.judged_verdict,
                "execution_mode": result.execution_mode,
                "execution_override": result.execution_override,
            }.items() if value is not None
        }
        message.setdefault("extra", {}).setdefault("phase_monitor", {})
        message["extra"]["phase_monitor"].update(
            {
                "monitor_baseline_variant": self.monitor.variant,
                "execute_proposed_step": result.execute,
                "is_blocking": result.is_blocking,
                "classified_phase": result.classified_phase,
                "liveplan_step_index": step_index,
                "matched_rule_ids": list(result.rule_ids),
                "guidance_pending": bool(result.execute and result.message),
                **execution_metadata,
                "refiner_config": self.monitor.refiner_config,
                "refiner_interval": self.monitor.refiner_interval,
            }
        )

        if result.execute:
            self.mark_generation_outcome(message.get("extra", {}).get("overhead_event_id"), "accepted")
            if result.message:
                action["_monitor_baseline_pending_guidance"] = result.message
            return MonitorDecision(execute_proposed_step=True, reason="monitor_baseline")

        if result.force_submit:
            # A give_up-style verdict: the monitor judged the run
            # unsalvageable and asked the agent to submit and stop, but a
            # text instruction is only a request, not a guarantee -- an
            # agent that doesn't comply just keeps getting re-blocked, which
            # burns the rest of the step budget for nothing (see
            # CadenceMonitor's give_up docstring). Submit the
            # current diff ourselves instead of trusting compliance: this
            # raises Submitted, exactly like the agent's own submit command
            # would, and ends the run deterministically.
            self.mark_generation_outcome(
                message.get("extra", {}).get("overhead_event_id"), "force_submitted"
            )
            self.note_intervention(f"monitor_baseline-{proposal_id}-force-submit")
            if agent.messages and agent.messages[-1].get("role") == "assistant":
                agent.messages.pop()
            agent.env.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"})

        self.mark_generation_outcome(message.get("extra", {}).get("overhead_event_id"), "discarded")
        event = MonitorEvent(
            event_id=f"monitor_baseline-{proposal_id}",
            message=result.message or "",
            event_type="monitor_intervention",
            discarded_proposed_step_id=proposal_id,
            verdict="intervene",
            metadata={
                **execution_metadata,
                "monitor_baseline_variant": self.monitor.variant,
                "classified_phase": result.classified_phase,
                "liveplan_step_index": step_index,
                "matched_rule_ids": list(result.rule_ids),
                "refiner_config": self.monitor.refiner_config,
                "refiner_interval": self.monitor.refiner_interval,
            },
        )
        self._inject_intervention(agent, event)
        return MonitorDecision(
            execute_proposed_step=False,
            monitor_event=event,
            reason="monitor_baseline_blocked",
        )

    def handle_after_action(
        self, agent, message: dict[str, Any], action: dict[str, Any], result: Any
    ) -> Optional[str]:
        """Post-action hook: run right after `env.execute(action)`, once the observation is known."""

        step_index = action.get("_monitor_baseline_step_index")
        if step_index is None:
            step_index = self.monitor.executed_step_count
        proposal_id = str(
            action.get("_monitor_baseline_proposal_id")
            or message.get("extra", {}).get("overhead_step_id")
            or self._proposal_counter_int
        )
        command = action.get("command", "")
        thought = _stringify_content(message.get("content", ""))
        observation = result.get("output", "") if isinstance(result, dict) else str(result)
        pending = action.pop("_monitor_baseline_pending_guidance", None)
        periodic = self.monitor.after_action(
            step_index,
            command,
            thought,
            observation,
            executed=True,
            accounting_step_id=proposal_id,
        )
        return pending or periodic

    def _inject_intervention(self, agent, event: MonitorEvent) -> None:
        """Discard the current proposal and append feedback as a user message."""

        self.note_intervention(event.event_id)
        if agent.messages and agent.messages[-1].get("role") == "assistant":
            agent.messages.pop()
        agent.add_messages(
            agent.model.format_message(
                role="user",
                content=event.message,
                extra={
                    "message_type": "monitor_intervention",
                    **self.event_record(event),
                },
            )
        )


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
        )
    return str(content) if content is not None else ""


def _mini_task_context(agent) -> str:
    for message in getattr(agent, "messages", []):
        if message.get("role") == "user" and message.get("extra", {}).get("message_type") != "monitor_intervention":
            return _stringify_content(message.get("content", ""))
    return ""

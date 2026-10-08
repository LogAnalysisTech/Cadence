from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

from monitor_runtime.adapters.moatless import _describe_moatless_action, _environment_observation, _executed_action_count, _moatless_task_context

from monitor_runtime.integration import NativePhaseMonitorIntegration
from monitor_runtime.types import MonitorDecision, MonitorEvent

from ..bridge import LivePlanBoundaryMonitor
from ..moatless_monitor import derive_node_monitor_state, evaluate_node_action, invalidate_node_monitor_state


class MoatlessLivePlanIntegration(NativePhaseMonitorIntegration):
    """LivePlan (Agent-Planner) pre-/post-action integration for moatless's branching `SearchTree`.

    Unlike the swe-agent/mini-swe-agent adapters, this adapter does NOT use
    `LivePlanBoundaryMonitor.before_action`/`.after_action`: those assume one
    linear trajectory, but moatless's `Node` tree can branch, so a single
    shared `StatefulPhaseMonitor` would conflate sibling branches (plan item
    6). Instead this adapter uses
    `monitor_baseline.moatless_monitor.evaluate_node_action`/
    `derive_node_monitor_state`, which derive each node's monitor state by
    replaying its own root-to-node path (cached per node).

    Accepted advice is attached to the completed action's observation, which
    is the context Moatless actually serializes for the next generation. The
    same advice is retained in ``monitor_events`` as experiment provenance.
    Refiner triggers, cooldown context, and intended-plan updates are stored
    and replayed per branch rather than shared between sibling nodes.
    """

    def __init__(self, monitor: LivePlanBoundaryMonitor):
        super().__init__(monitor)
        # Rule-Based and full LivePlan inspect every model decision online;
        # fixed-interval advice retains the historical Moatless action shape.
        self.requires_single_action = monitor.variant in {"monitor", "refiner"}
        self._rules_config = getattr(monitor, "rules_config", None)
        self._run_root: Any | None = None
        self._pending_refiner_triggers: dict[int, Any] = {}

    def _maybe_reset_for_root(self, node) -> None:
        root = node
        while getattr(root, "parent", None) is not None:
            root = root.parent
        if self._run_root is not root:
            self._run_root = root
            self._pending_refiner_triggers = {}

    async def inspect_before_action(self, agent, node, action_step) -> MonitorDecision:
        """Inspect one structured ActionStep immediately before execution."""

        self._maybe_reset_for_root(node)
        event_id = getattr(node, "overhead_generation_event_id", None)

        if self.monitor.variant == "refiner_only":
            # No monitor object exists in this variant (plan item 4).
            self.associate_generation(event_id, phase=None, step_id=None)
            self.mark_generation_outcome(event_id, "accepted")
            self._write_phase_monitor(
                action_step,
                execute=True,
                blocked=False,
                phase=None,
                step_index=_executed_node_count(node),
                rule_ids=[],
                guidance_pending=False,
            )
            return MonitorDecision(execute_proposed_step=True, reason="monitor_baseline")

        started = perf_counter()
        result = evaluate_node_action(node, action_step, rules_config=self._rules_config)
        self.monitor.overhead.record(
            component="monitor",
            operation="rule_check",
            latency_seconds=perf_counter() - started,
            step_id=str(getattr(node, "node_id", id(node))),
        )
        step_id = str(getattr(node, "node_id", id(node)))
        parent_state = derive_node_monitor_state(node, rules_config=self._rules_config)
        phase = (
            str(result.current_phase)
            if result is not None and result.current_phase is not None
            else (str(parent_state.current_phase) if parent_state.current_phase is not None else None)
        )
        self.associate_generation(event_id, phase=phase, step_id=step_id)
        node.overhead_step_id = step_id
        if result is None:
            self.mark_generation_outcome(event_id, "accepted")
            self._write_phase_monitor(
                action_step,
                execute=True,
                blocked=False,
                phase=phase,
                step_index=parent_state.step_index,
                rule_ids=[],
                guidance_pending=False,
            )
            return MonitorDecision(execute_proposed_step=True, reason="monitor_baseline")

        blocked = result.should_block_and_refine()

        if self.monitor.variant == "monitor":
            messages = result.get_all_messages()
            message = "\n".join(messages) if messages else None
        elif blocked:
            # The action never executes, so the node's trajectory (as of
            # right now) is exactly the trajectory the refiner should see --
            # matches upstream behavior.
            message = self._call_refiner(node, result)
        else:
            # variant == "refiner", non-blocking: calling the refiner now
            # would hand it a trajectory missing this action's own
            # observation (`_build_moatless_trajectory` only includes steps
            # with `observation is not None`). Defer to `handle_after_action`,
            # which runs once `_execute()` has populated it (plan item 2).
            self._pending_refiner_triggers[id(action_step)] = result
            message = None

        rule_ids = [
            str(match.rule_id)
            for match in result.rule_matches
            if getattr(match, "rule_id", None)
        ]
        self._write_phase_monitor(
            action_step,
            execute=not blocked,
            blocked=blocked,
            phase=phase,
            step_index=parent_state.step_index,
            rule_ids=rule_ids,
            guidance_pending=bool(
                not blocked and (message or self.monitor.variant == "refiner")
            ),
        )

        if not blocked:
            self.mark_generation_outcome(event_id, "accepted")
            if message:
                action_step.phase_monitor["monitor_baseline_pending_guidance"] = message
            return MonitorDecision(execute_proposed_step=True, reason="monitor_baseline")

        self.mark_generation_outcome(event_id, "discarded")
        event = MonitorEvent(
            event_id=f"monitor_baseline-{getattr(node, 'node_id', id(node))}-{len(node.monitor_events)}",
            message=(message.text if isinstance(message, _Refinement) else message) or "",
            event_type="monitor_intervention",
            discarded_proposed_step_id=step_id,
            verdict="intervene",
            metadata={
                "monitor_baseline_variant": self.monitor.variant,
                "execute_proposed_step": False,
                "is_blocking": True,
                "classified_phase": phase,
                "liveplan_step_index": parent_state.step_index,
                "matched_rule_ids": rule_ids,
                "refiner_config": self.monitor.refiner_config,
                "refiner_interval": self.monitor.refiner_interval,
            },
        )
        if isinstance(message, _Refinement):
            event.metadata.update(message.metadata)
        self._record_monitor_event(node, event)
        invalidate_node_monitor_state(node)
        self._inject_intervention(node, event, action_step)
        self.persist_overhead(node)
        return MonitorDecision(
            execute_proposed_step=False,
            monitor_event=event,
            reason="monitor_baseline_blocked",
        )

    def _call_refiner(
        self, node, result, *, state_includes_current_action: bool = False
    ) -> Optional["_Refinement"]:
        refiner = getattr(self.monitor, "_refiner", None)
        if refiner is None:
            return None
        from plan_refiner.refiner import PlanRefiner
        from plan_refiner.types import RefinerInput

        parent_state = derive_node_monitor_state(node, rules_config=self._rules_config)
        current_step = parent_state.step_index + (0 if state_includes_current_action else 1)
        last_step, last_guidance = self._branch_refinement_context(node)
        refiner_input = RefinerInput(
            issue_description=self.monitor.issue_description or _moatless_task_context(node),
            trajectory=_build_moatless_trajectory(node),
            state_summary=PlanRefiner.build_state_summary_from_monitor(
                parent_state.monitor,
                result.rule_matches if result else None,
                exclude_rule_patterns=["plan_compliance"],
            ),
            last_refinement_guidance=last_guidance,
            last_refinement_step=last_step,
        )
        output = self.monitor._invoke_refiner(refiner_input, step_index=current_step)
        if output.is_cooling_period:
            messages = result.get_all_messages() if result else []
            text = output.cooling_period_message or ("\n".join(messages) if messages else None)
            return _Refinement(text=text, step_index=current_step, cooling=True)

        text = output.new_plan or output.raw_response or None
        return _Refinement(
            text=text,
            step_index=current_step,
            refined_plan_phases=list(output.refined_plan_phases),
        )

    async def handle_after_action(self, agent, node, action_step) -> None:
        """Post-action hook: run after `_execute(node, action_step)` succeeds.

        Handles two deferred cases: variant `refiner`'s non-blocking refiner
        call (plan item 2 -- keyed to the accepted action by
        `self._pending_refiner_triggers`, now safe to call because
        `_build_moatless_trajectory` will include this step's observation),
        and variant `refiner_only`'s periodic trigger (plan item 4). Other
        variants/cases surface their guidance from `inspect_before_action`
        via `_record_guidance_event` instead.
        """

        self._maybe_reset_for_root(node)
        invalidate_node_monitor_state(node)

        pending_rule_guidance = action_step.phase_monitor.pop(
            "monitor_baseline_pending_guidance", None
        )
        if pending_rule_guidance:
            self._append_guidance_to_observation(action_step, pending_rule_guidance)
            self._record_guidance_event(
                node,
                pending_rule_guidance,
                blocking=False,
                metadata={"guidance_source": "rule_match"},
            )
            action_step.phase_monitor["guidance_delivered"] = True

        trigger = self._pending_refiner_triggers.pop(id(action_step), None)
        if self.monitor.variant == "refiner" and trigger is not None:
            refinement = self._call_refiner(
                node, trigger, state_includes_current_action=True
            )
            if refinement and refinement.text:
                self._append_guidance_to_observation(action_step, refinement.text)
                self._record_guidance_event(
                    node,
                    refinement.text,
                    blocking=False,
                    metadata=refinement.metadata,
                )
                action_step.phase_monitor["guidance_delivered"] = True
                action_step.phase_monitor["refiner_cooling"] = refinement.cooling
                if refinement.refined_plan_phases:
                    invalidate_node_monitor_state(node)
            return

        if self.monitor.variant != "refiner_only":
            return
        if action_step is not node.action_steps[-1]:
            # Moatless can execute several actions from one model turn (one
            # node); only check the cadence once that whole turn has
            # executed, not once per action inside it -- otherwise a
            # multi-action turn advances the "every N steps" counter by more
            # than one step and the refiner fires off-cadence.
            return
        branch_step_count = _executed_node_count(node)
        interval = getattr(self.monitor, "_refiner_only_interval", 0)
        if interval <= 0 or branch_step_count % interval != 0:
            return

        refiner = getattr(self.monitor, "_refiner", None)
        if refiner is None:
            return
        from plan_refiner.types import RefinerInput, StateSummary

        last_step, last_guidance = self._branch_refinement_context(node)
        refiner_input = RefinerInput(
            issue_description=self.monitor.issue_description or _moatless_task_context(node),
            trajectory=_build_moatless_trajectory(node),
            state_summary=StateSummary(),
            last_refinement_guidance=last_guidance,
            last_refinement_step=last_step,
        )
        output = self.monitor._invoke_refiner(
            refiner_input, step_index=branch_step_count
        )
        if output.is_cooling_period:
            if output.cooling_period_message:
                self._append_guidance_to_observation(
                    action_step, output.cooling_period_message
                )
                self._record_guidance_event(
                    node,
                    output.cooling_period_message,
                    blocking=False,
                    metadata={
                        "guidance_source": "refiner",
                        "refinement_step": branch_step_count,
                        "cooling": True,
                    },
                )
                action_step.phase_monitor["guidance_delivered"] = True
                action_step.phase_monitor["refiner_cooling"] = True
            return
        text = output.new_plan or output.raw_response or None
        if text:
            self._append_guidance_to_observation(action_step, text)
            self._record_guidance_event(
                node,
                text,
                blocking=False,
                metadata={
                    "guidance_source": "refiner",
                    "refinement_step": branch_step_count,
                    "cooling": False,
                },
            )
            action_step.phase_monitor["guidance_delivered"] = True
            action_step.phase_monitor["refiner_cooling"] = False

    async def handle_action_failure(self, agent, node, action_step, error: BaseException) -> None:
        """Discard pending advice tied to an action that never committed."""

        self._pending_refiner_triggers.pop(id(action_step), None)
        action_step.phase_monitor.pop("monitor_baseline_pending_guidance", None)
        action_step.phase_monitor["post_action_monitor_cancellation"] = {
            "reason": "action_failed",
            "error_type": type(error).__name__,
        }

    def persist_overhead(self, node) -> None:
        """Store one cumulative ledger on the trajectory root, avoiding per-node duplication."""

        root = node
        while getattr(root, "parent", None) is not None:
            root = root.parent
        root.overhead = self.overhead_payload()

    def _record_guidance_event(
        self,
        node,
        message: str,
        *,
        blocking: bool,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        record = {
            "event_type": "monitor_guidance",
            "event_id": f"monitor_baseline-guidance-{getattr(node, 'node_id', id(node))}-{len(node.monitor_events)}",
            "message": message,
            "verdict": "intervene" if blocking else "advise",
            "metadata": {"blocking": blocking, **(metadata or {})},
            "after_action_step_count": _executed_action_count(node),
        }
        node.monitor_events = [*node.monitor_events, record]

    @staticmethod
    def _append_guidance_to_observation(action_step, message: str) -> None:
        """Put advice in the history source consumed by Moatless memory."""

        observation = getattr(action_step, "observation", None)
        if observation is None:
            raise RuntimeError("Cannot deliver LivePlan guidance without an observation")
        original = str(getattr(observation, "message", "") or "")
        guidance = "External planning guidance for the next step:\n" + message
        observation.message = f"{original}\n\n{guidance}" if original else guidance

    def _write_phase_monitor(
        self,
        action_step,
        *,
        execute: bool,
        blocked: bool,
        phase: Optional[str],
        step_index: int,
        rule_ids: list[str],
        guidance_pending: bool,
    ) -> None:
        # Reassignment is required for Pydantic's exclude_unset trajectory dump.
        action_step.phase_monitor = {
            **getattr(action_step, "phase_monitor", {}),
            "monitor_baseline_variant": self.monitor.variant,
            "execute_proposed_step": execute,
            "is_blocking": blocked,
            "classified_phase": phase,
            "liveplan_step_index": step_index,
            "matched_rule_ids": rule_ids,
            "guidance_pending": guidance_pending,
            "refiner_config": self.monitor.refiner_config,
            "refiner_interval": self.monitor.refiner_interval,
        }

    @staticmethod
    def _branch_refinement_context(node) -> tuple[Optional[int], Optional[str]]:
        latest: tuple[Optional[int], Optional[str]] = (None, None)
        for trajectory_node in node.get_trajectory():
            for event in getattr(trajectory_node, "monitor_events", []):
                metadata = event.get("metadata", {}) if isinstance(event, dict) else {}
                if metadata.get("guidance_source") != "refiner" or metadata.get("cooling"):
                    continue
                latest = (metadata.get("refinement_step"), event.get("message"))
        return latest

    def _record_monitor_event(self, node, event: MonitorEvent) -> None:
        record = self.event_record(event)
        record["after_action_step_count"] = _executed_action_count(node)
        node.monitor_events = [*node.monitor_events, record]

    def _inject_intervention(self, node, event: MonitorEvent, action_step) -> None:
        """Remove the rejected proposal and prepare the node for regeneration."""

        self.note_intervention(event.event_id)
        try:
            proposed_index = node.action_steps.index(action_step)
        except ValueError:
            proposed_index = len(node.action_steps)
        node.action_steps = node.action_steps[:proposed_index]
        node.assistant_message = None
        node.thoughts = None
        node.thinking_blocks = None
        node.user_message = (
            "Your previous action was not executed. Follow this guidance and try again.\n\n"
            f"{event.message}"
        )




def _build_moatless_trajectory(node) -> list:
    from plan_refiner.refiner import PlanRefiner
    from plan_monitor.phases import ActionEvent

    raw: list[tuple[ActionEvent, str, str]] = []
    step_index = 0
    for trajectory_node in node.get_trajectory():
        for action_step in trajectory_node.action_steps:
            if action_step.observation is None:
                continue
            step_index += 1
            action = action_step.action
            thought = getattr(action, "thoughts", None) or trajectory_node.assistant_message or ""
            observation = _environment_observation(
                getattr(action_step.observation, "message", None) or ""
            )
            command_text = _describe_moatless_action(action)
            raw.append((ActionEvent(step_index=step_index, command=command_text), thought, observation))
    return PlanRefiner.build_trajectory_steps(raw)






def _executed_node_count(node) -> int:
    """Count trajectory nodes with at least one executed action.

    `refiner_only` (Fixed-Interval) must fire every N *steps*, matching the
    mini-swe-agent baseline where one step is one model turn -- Moatless can
    put several actions in a single turn's node, and `_executed_action_count`
    would then advance the cadence by more than one per turn.
    """

    return sum(
        any(action_step.observation is not None for action_step in trajectory_node.action_steps)
        for trajectory_node in node.get_trajectory()
    )




@dataclass(frozen=True)
class _Refinement:
    text: Optional[str]
    step_index: int
    cooling: bool = False
    refined_plan_phases: list[str] = field(default_factory=list)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "guidance_source": "refiner",
            "refinement_step": self.step_index,
            "cooling": self.cooling,
            "refined_plan_phases": list(self.refined_plan_phases),
        }

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

from monitor_runtime.finding_lifecycle import FindingLifecycleConfig
from monitor_runtime.overhead import OverheadLedger
from monitor_runtime.types import BeforeActionResult

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from plan_monitor.monitor import StatefulPhaseMonitor  # noqa: E402
from plan_monitor.phases import ActionEvent  # noqa: E402
from plan_refiner.refiner import PlanRefiner, RefinerConfig  # noqa: E402
from plan_refiner.models import OpenRouterModel  # noqa: E402
from plan_refiner.types import RefinerInput  # noqa: E402

VALID_VARIANTS = {"monitor", "refiner_only", "refiner"}


def monitor_baseline_variant() -> str:
    value = os.getenv("MONITOR_BASELINE_VARIANT", "monitor").strip().lower()
    if value not in VALID_VARIANTS:
        raise ValueError(
            f"MONITOR_BASELINE_VARIANT must be one of {sorted(VALID_VARIANTS)}, got {value!r}"
        )
    return value




class LivePlanBoundaryMonitor:
    """Adapts Agent-Planner's (LivePlan) rule monitor + LLM refiner to this
    project's native-agent-loop integration contract.

    Ownership note for non-blocking guidance (plan item 1): `before_action`
    computes the guidance text for non-blocking rule matches (variant
    `monitor`) or refiner output (variant `refiner`) and returns it via
    `BeforeActionResult.message` with `is_blocking=False`. The bridge itself
    does not re-surface that message later -- it is the *adapter*'s
    responsibility to stash it (e.g. on the step/message extra dict) when
    `before_action` returns non-blocking guidance, and to append it to the
    next turn once the action's observation is known (typically from the
    per-agent post-action hook that also calls `after_action`). This keeps
    the bridge itself agent-shape-agnostic: it never touches agent history.
    """

    is_null_monitor = False
    requires_agent_phase_reporting = False

    def __init__(
        self,
        *,
        variant: Optional[str] = None,
        rules_config: Optional[str] = None,
        refiner_config: Optional[str] = None,
        refiner_model: Optional[str] = None,
        issue_description: str = "",
    ) -> None:
        self.variant = variant or monitor_baseline_variant()
        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"Unknown monitor_baseline variant: {self.variant!r}")

        self._rules_config = rules_config
        configured_refiner = refiner_config or os.getenv("MONITOR_BASELINE_REFINER_CONFIG")
        self._refiner_config = configured_refiner or (
            "config/refiner-only.yaml" if self.variant == "refiner_only" else "config/default.yaml"
        )
        self._refiner_model = refiner_model or os.getenv("MONITOR_BASELINE_REFINER_MODEL")

        self._monitor: Optional[StatefulPhaseMonitor] = None
        if self.variant in {"monitor", "refiner"}:
            self._monitor = StatefulPhaseMonitor(rules_config=self._rules_config)

        self._refiner: Optional[PlanRefiner] = None
        self._refiner_only_interval: int = 0
        if self.variant in {"refiner", "refiner_only"}:
            model = self._create_refiner_model()
            self._refiner = PlanRefiner(
                model, config=RefinerConfig(template_path=self._refiner_config)
            )
            self._refiner_only_interval = PlanRefiner.get_min_steps_from_config(
                self._refiner_config
            )

        self.overhead = OverheadLedger()
        # LivePlan keeps its own state machine and does not use our finding memory.
        self.finding_lifecycle = FindingLifecycleConfig(enabled=False)
        self.open_findings: tuple = ()

        self.rules_config = self._rules_config
        self._committed: list[tuple[ActionEvent, str, str]] = []
        self._last_refinement_step: Optional[int] = None
        self._last_refinement_guidance: Optional[str] = None
        self._monitor_triggers_since_refinement: int = 0
        self._executed_step_count: int = 0
        self._pending_refiner_trigger: Optional[list] = None
        self.issue_description = issue_description

    def _create_refiner_model(self):
        if not self._refiner_model:
            return PlanRefiner.create_model_from_config(self._refiner_config)

        import yaml
        import plan_refiner

        config_path = Path(self._refiner_config)
        if not config_path.is_absolute():
            config_path = Path(plan_refiner.__file__).parent / config_path
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        model_config = config.get("external_planner", {}).get("model", {})
        return OpenRouterModel(
            model_name=self._refiner_model,
            model_kwargs=model_config.get("model_kwargs", {}),
            set_cache_control=model_config.get("set_cache_control"),
        )

    # -- accessors ---------------------------------------------------------

    @property
    def current_phase(self) -> Optional[str]:
        """The LivePlan-classified current phase, or `None` for `refiner_only`
        (which has no monitor object) or before the first step is observed.

        Exposed so adapters can attribute overhead-ledger entries to a
        LivePlan-classified phase without reaching into the private
        `_monitor` attribute (Fix 3: this project's own `phase` field is
        agent-self-reported and deliberately unused by this baseline).
        """

        if self._monitor is None:
            return None
        phase = self._monitor.get_current_phase()
        return str(phase) if phase is not None else None

    @property
    def refiner_config(self) -> Optional[str]:
        return self._refiner_config if self._refiner is not None else None

    @property
    def refiner_interval(self) -> Optional[int]:
        if self._refiner is None:
            return None
        return int(self._refiner.config.min_steps_between_refinements or 0)

    @property
    def executed_step_count(self) -> int:
        """Number of actions committed to the native executable trajectory."""

        return self._executed_step_count

    # -- lifecycle -----------------------------------------------------

    def reset_run_state(self, issue_description: str = "") -> None:
        if self._monitor is not None:
            self._monitor.reset()
        self._committed = []
        self.overhead.reset()
        self._last_refinement_step = None
        self._last_refinement_guidance = None
        self._monitor_triggers_since_refinement = 0
        self._executed_step_count = 0
        self._pending_refiner_trigger = None
        self.issue_description = issue_description

    # -- overhead helpers ------------------------------------------------

    def _refiner_model_name(self) -> Optional[str]:
        if self._refiner is None:
            return None
        model = getattr(self._refiner, "model", None)
        config = getattr(model, "config", None)
        name = getattr(config, "model_name", None)
        if name:
            return str(name)
        name = getattr(model, "model_name", None)
        return str(name) if name else None

    def _record_refiner_call(
        self,
        *,
        step_index: int,
        accounting_step_id: Optional[str],
        latency_seconds: float,
        output: Any,
    ) -> None:
        outcome = "cooling" if getattr(output, "is_cooling_period", False) else "proposed"
        usage = getattr(output, "usage", None) if outcome != "cooling" else None
        self.overhead.record(
            component="monitor",
            operation="refiner_cooldown" if outcome == "cooling" else "refiner_call",
            latency_seconds=latency_seconds,
            usage=usage,
            model=self._refiner_model_name(),
            step_id=accounting_step_id or str(step_index),
            outcome=outcome,
            metadata={"liveplan_step_index": step_index},
        )

    def _invoke_refiner(
        self,
        refiner_input: RefinerInput,
        *,
        step_index: int,
        accounting_step_id: Optional[str] = None,
    ):
        """Run and account for one refiner decision, including failed API attempts."""

        assert self._refiner is not None
        started = perf_counter()
        try:
            output = self._refiner.refine_plan(refiner_input, current_step=step_index)
        except Exception:
            self.overhead.record(
                component="monitor",
                operation="refiner_call",
                latency_seconds=perf_counter() - started,
                model=self._refiner_model_name(),
                step_id=accounting_step_id or str(step_index),
                outcome="failed",
                metadata={"liveplan_step_index": step_index},
            )
            raise
        self._record_refiner_call(
            step_index=step_index,
            accounting_step_id=accounting_step_id,
            latency_seconds=perf_counter() - started,
            output=output,
        )
        return output

    def _monitor_refiner_is_cooling(self, step_index: int) -> bool:
        """Mirror Agent-Planner's dual step/trigger cooldown exactly."""

        if self._last_refinement_step is None or self._refiner is None:
            return False
        interval = int(self._refiner.config.min_steps_between_refinements or 0)
        if interval <= 0:
            return False
        return (
            step_index - self._last_refinement_step < interval
            and self._monitor_triggers_since_refinement < interval
        )

    def _record_monitor_refiner_cooldown(
        self, *, step_index: int, accounting_step_id: str
    ) -> None:
        self._monitor_triggers_since_refinement += 1
        self.overhead.record(
            component="monitor",
            operation="refiner_cooldown",
            latency_seconds=0.0,
            model=self._refiner_model_name(),
            step_id=accounting_step_id,
            outcome="cooling",
            metadata={"liveplan_step_index": step_index},
        )

    @staticmethod
    def _monitor_message(result: Any) -> Optional[str]:
        messages = result.get_all_messages()
        return "\n".join(messages) if messages else None

    @staticmethod
    def _refiner_message(output: Any, *, fallback: Optional[str] = None) -> Optional[str]:
        """Format guidance exactly as Agent-Planner's SWE-agent integration."""

        if not output.raw_response:
            return fallback
        body = output.new_plan or output.raw_response
        return f"[Plan Refiner Guidance]\n\n{body}"

    def _note_refinement(self, step_index: int, output: Any) -> None:
        self._last_refinement_step = step_index
        self._monitor_triggers_since_refinement = 0
        if output.new_plan:
            self._last_refinement_guidance = output.new_plan

    # -- pre-action ------------------------------------------------------

    def before_action(
        self,
        step_index: int,
        command: str,
        thought: str,
        *,
        accounting_step_id: Optional[str] = None,
    ) -> BeforeActionResult:
        accounting_step_id = accounting_step_id or str(step_index)
        # A deferred trigger belongs only to the proposal that created it. If
        # native execution failed before `after_action`, upstream LivePlan
        # drops that call-local state; discard it before inspecting a retry.
        self._pending_refiner_trigger = None
        if self.variant == "refiner_only":
            # No monitor object exists in this variant (plan item 4); the
            # rule engine never runs, so nothing can block execution here.
            return BeforeActionResult(execute=True, message=None, is_blocking=False)

        assert self._monitor is not None
        event = ActionEvent(step_index=step_index, command=command)
        started = perf_counter()
        try:
            result = self._monitor.check_step_pre_emptively(event, thought=thought)
        except Exception:
            self.overhead.record(
                component="monitor",
                operation="rule_check",
                latency_seconds=perf_counter() - started,
                step_id=accounting_step_id,
                outcome="failed",
            )
            raise
        checked_phase = (
            str(result.current_phase)
            if result is not None and result.current_phase is not None
            else self.current_phase
        )
        self.overhead.record(
            component="monitor",
            operation="rule_check",
            latency_seconds=perf_counter() - started,
            phase=checked_phase,
            step_id=accounting_step_id,
            metadata={"liveplan_step_index": step_index},
            outcome="matched" if result is not None else "clear",
        )
        if result is None:
            return BeforeActionResult(
                execute=True,
                message=None,
                is_blocking=False,
                classified_phase=checked_phase,
            )

        blocked = result.should_block_and_refine()
        rule_ids = tuple(
            str(match.rule_id)
            for match in result.rule_matches
            if getattr(match, "rule_id", None)
        )

        if self.variant == "monitor":
            message = self._monitor_message(result)
            return BeforeActionResult(
                execute=not blocked,
                message=message,
                is_blocking=blocked,
                classified_phase=checked_phase,
                rule_ids=rule_ids,
            )

        # variant == "refiner"
        assert self._refiner is not None

        if not blocked:
            # The action has not executed yet, so `self._committed` is
            # missing the very step this trigger fired on -- calling the
            # refiner here would hand it a stale trajectory. Defer the
            # actual refiner call to `after_action`, once the step's
            # observation is known and committed (see plan item 2).
            self._pending_refiner_trigger = result.rule_matches
            return BeforeActionResult(
                execute=True,
                message=None,
                is_blocking=False,
                classified_phase=checked_phase,
                rule_ids=rule_ids,
            )

        # Blocked: the proposed action never executes, so `self._committed`
        # (without this step) is exactly the right trajectory to refine on.
        monitor_message = self._monitor_message(result)
        if self._monitor_refiner_is_cooling(step_index):
            self._record_monitor_refiner_cooldown(
                step_index=step_index, accounting_step_id=accounting_step_id
            )
            return BeforeActionResult(
                execute=False,
                message=monitor_message,
                is_blocking=True,
                classified_phase=checked_phase,
                rule_ids=rule_ids,
            )

        refiner_input = RefinerInput(
            issue_description=self.issue_description,
            trajectory=PlanRefiner.build_trajectory_steps(self._committed),
            state_summary=PlanRefiner.build_state_summary_from_monitor(
                self._monitor,
                result.rule_matches,
                exclude_rule_patterns=["plan_compliance"],
            ),
            last_refinement_guidance=self._last_refinement_guidance,
            last_refinement_step=self._last_refinement_step,
        )
        output = self._invoke_refiner(
            refiner_input,
            step_index=step_index,
            accounting_step_id=accounting_step_id,
        )
        text = self._refiner_message(output, fallback=monitor_message)
        self._note_refinement(step_index, output)
        if output.refined_plan_phases:
            self._monitor.update_intended_plan(output.refined_plan_phases)

        return BeforeActionResult(
            execute=not blocked,
            message=text,
            is_blocking=blocked,
            classified_phase=checked_phase,
            rule_ids=rule_ids,
        )

    # -- post-action -------------------------------------------------------

    def after_action(
        self,
        step_index: int,
        command: str,
        thought: str,
        observation: str,
        executed: bool,
        *,
        accounting_step_id: Optional[str] = None,
    ) -> Optional[str]:
        accounting_step_id = accounting_step_id or str(step_index)
        if executed:
            self._committed.append((ActionEvent(step_index=step_index, command=command), thought, observation))
            self._executed_step_count += 1

        if self.variant == "refiner" and self._pending_refiner_trigger is not None:
            if not executed:
                # The proposed action was discarded after all (e.g. some
                # other mechanism vetoed it) -- drop the stale trigger
                # rather than refining on a trajectory that never gained
                # the step it was queued for.
                self._pending_refiner_trigger = None
                return None
            trigger = self._pending_refiner_trigger
            self._pending_refiner_trigger = None
            assert self._refiner is not None
            monitor_message = "\n".join(
                str(match.message)
                for match in trigger
                if getattr(match, "message", None)
            ) or None
            if self._monitor_refiner_is_cooling(step_index):
                self._record_monitor_refiner_cooldown(
                    step_index=step_index, accounting_step_id=accounting_step_id
                )
                return monitor_message
            refiner_input = RefinerInput(
                issue_description=self.issue_description,
                trajectory=PlanRefiner.build_trajectory_steps(self._committed),
                state_summary=PlanRefiner.build_state_summary_from_monitor(
                    self._monitor,
                    trigger,
                    exclude_rule_patterns=["plan_compliance"],
                ),
                last_refinement_guidance=self._last_refinement_guidance,
                last_refinement_step=self._last_refinement_step,
            )
            output = self._invoke_refiner(
                refiner_input,
                step_index=step_index,
                accounting_step_id=accounting_step_id,
            )
            text = self._refiner_message(output, fallback=monitor_message)
            self._note_refinement(step_index, output)
            if output.refined_plan_phases:
                self._monitor.update_intended_plan(output.refined_plan_phases)
            return text

        if self.variant != "refiner_only":
            return None
        if not executed:
            return None
        if self._refiner_only_interval <= 0:
            return None
        # Upstream checks the zero-based native trajectory index after the
        # action runs: the first call is at index N, then every N native
        # trajectory entries thereafter.
        if step_index < self._refiner_only_interval:
            return None
        if (
            self._last_refinement_step is not None
            and step_index - self._last_refinement_step < self._refiner_only_interval
        ):
            return None

        assert self._refiner is not None
        from plan_refiner.types import StateSummary

        refiner_input = RefinerInput(
            issue_description=self.issue_description,
            trajectory=PlanRefiner.build_trajectory_steps(self._committed),
            state_summary=StateSummary(),
            last_refinement_guidance=self._last_refinement_guidance,
            last_refinement_step=self._last_refinement_step,
        )
        output = self._invoke_refiner(
            refiner_input,
            step_index=step_index,
            accounting_step_id=accounting_step_id,
        )
        self._last_refinement_step = step_index
        if output.is_cooling_period:
            return output.cooling_period_message

        text = self._refiner_message(output)
        if output.new_plan:
            self._last_refinement_guidance = output.new_plan
        return text

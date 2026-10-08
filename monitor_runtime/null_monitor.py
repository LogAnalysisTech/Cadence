from __future__ import annotations

from typing import Sequence

from .finding_lifecycle import FindingLifecycleConfig
from .overhead import OverheadLedger
from .types import MonitorDecision, TrajectoryEvent, AgentStep


class NullPhaseBoundaryMonitor:
    """Stand-in installed when the realtime monitor is disabled.

    Matches the subset of `PhaseBoundaryMonitor`'s interface every vendored
    agent loop touches (`inspect_before_execution`, `overhead`, `finding_lifecycle`,
    `reset_run_state`, `open_findings`) so a loop can always install *some*
    monitor and never needs an `is None` branch in explicitly instrumented
    control runs. Every proposed step executes unmodified and no capability
    selection, review, or intervention happens. True vanilla runs install no
    integration at all.
    """

    is_null_monitor = True
    requires_agent_phase_reporting = False

    def __init__(self) -> None:
        self.overhead = OverheadLedger()
        self.finding_lifecycle = FindingLifecycleConfig(enabled=False)
        self.open_findings: tuple = ()

    @property
    def submission_audit_required(self) -> bool:
        return False

    def inspect_before_execution(
        self,
        history: Sequence[TrajectoryEvent],
        proposed_step: AgentStep,
    ) -> MonitorDecision:
        return MonitorDecision(execute_proposed_step=True, reason="monitor disabled")

    def reset_run_state(self) -> None:
        self.overhead.reset()

"""Select a method and its native-agent adapter without mixing implementations."""

import os

from monitor_runtime.settings import load_environment
from typing import Any


def create_monitor(*, issue_description: str = "") -> Any:
    load_environment()
    if os.getenv("MONITOR_BASELINE_VARIANT", "monitor").strip().lower() == "adaptive_interval":
        from cadence.factory import create_cadence_monitor

        return create_cadence_monitor(issue_description=issue_description)
    from monitor_baseline.factory import create_liveplan_monitor

    return create_liveplan_monitor(issue_description=issue_description)


def create_mini_swe_agent_integration(monitor: Any) -> Any:
    if getattr(monitor, "variant", None) == "adaptive_interval":
        from cadence.adapters.mini_swe_agent import MiniSWEAgentCadenceIntegration

        return MiniSWEAgentCadenceIntegration(monitor)
    from monitor_baseline.adapters.mini_swe_agent import MiniSWEAgentLivePlanIntegration

    return MiniSWEAgentLivePlanIntegration(monitor)


def create_moatless_integration(monitor: Any) -> Any:
    if getattr(monitor, "variant", None) == "adaptive_interval":
        from cadence.adapters.moatless import MoatlessCadenceIntegration

        return MoatlessCadenceIntegration(monitor)
    from monitor_baseline.adapters.moatless import MoatlessLivePlanIntegration

    return MoatlessLivePlanIntegration(monitor)

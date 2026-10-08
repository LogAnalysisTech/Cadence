from __future__ import annotations

import os

from monitor_runtime.settings import load_environment

from .bridge import LivePlanBoundaryMonitor, monitor_baseline_variant


def monitor_baseline_enabled(env_var: str = "MONITOR_BASELINE_ENABLED") -> bool:
    return os.getenv(env_var, "").strip().lower() in {"1", "true", "yes", "on"}


def create_liveplan_monitor(*, issue_description: str = "") -> LivePlanBoundaryMonitor:
    """Build RuleOnly, Fixed-Interval or Full-LivePlan; propagate failures."""
    load_environment()
    variant = monitor_baseline_variant()
    rules_config = os.getenv("MONITOR_BASELINE_RULES_CONFIG") or None
    # Leaving the config unset selects the upstream prompt for each variant:
    # refiner-only.yaml for fixed-cadence advice, default.yaml for full LivePlan.
    refiner_config = os.getenv("MONITOR_BASELINE_REFINER_CONFIG") or None
    refiner_model = os.getenv("MONITOR_BASELINE_REFINER_MODEL") or None
    return LivePlanBoundaryMonitor(
        variant=variant,
        rules_config=rules_config,
        refiner_config=refiner_config,
        refiner_model=refiner_model,
        issue_description=issue_description,
    )

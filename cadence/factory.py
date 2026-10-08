import os

from monitor_runtime.settings import load_environment

from .monitor import CadenceMonitor


def create_cadence_monitor(*, issue_description: str = "") -> CadenceMonitor:
    """Build Cadence using the historical experiment environment keys."""
    load_environment()
    return CadenceMonitor(
        refiner_model=os.getenv("MONITOR_BASELINE_REFINER_MODEL") or None,
        issue_description=issue_description,
        fixed_interval=int(os.getenv("MONITOR_BASELINE_ADAPTIVE_FIXED_INTERVAL", "0")),
        base_interval=int(os.getenv("MONITOR_BASELINE_ADAPTIVE_BASE_INTERVAL", "5")),
        growth_step=int(os.getenv("MONITOR_BASELINE_ADAPTIVE_GROWTH_STEP", "2")),
        execution_mode=os.getenv("MONITOR_BASELINE_ADAPTIVE_EXECUTION", "severity"),
    )

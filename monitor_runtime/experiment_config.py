from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Iterable


VALID_MONITOR_MODES = {"vanilla", "liveplan"}
# "adaptive_interval" is not an upstream LivePlan variant -- it selects
# monitor_baseline.CadenceMonitor (see monitor_baseline/factory.py),
# an experimental variant of Fixed-Interval where the refiner's own verdict
# decides block/advise and drives the check cadence.
VALID_LIVEPLAN_VARIANTS = {"monitor", "refiner", "refiner_only", "adaptive_interval"}
VALID_ABLATIONS: set[str] = set()

_MODE_ALIASES = {
    "off": "vanilla",
    "none": "vanilla",
    "native": "vanilla",
    "baseline": "liveplan",
}

_MONITOR_ENV_KEYS = {
    "REALTIME_MONITOR_ENABLED",
    "MONITOR_BASELINE_ENABLED",
    "REALTIME_MONITOR_AGENT_VARIANT",
    "REALTIME_MONITOR_SELECTOR_MODEL",
    "REALTIME_MONITOR_REVIEWER_MODEL",
    "REALTIME_MONITOR_PLAYBOOK",
    "REALTIME_MONITOR_PHASE_POLICY_ENABLED",
    "REALTIME_MONITOR_PHASE_POLICY_STARTUP_MODE",
    "REALTIME_MONITOR_PHASE_POLICY_MODEL",
    "REALTIME_MONITOR_PHASE_POLICY_ROOT",
    "REALTIME_MONITOR_EXPERIMENT_ID",
    "REALTIME_MONITOR_FINDING_LIFECYCLE_ENABLED",
    "REALTIME_MONITOR_CAPABILITY_PLAYBOOK_ENABLED",
    "MONITOR_BASELINE_VARIANT",
    "MONITOR_BASELINE_RULES_CONFIG",
    "MONITOR_BASELINE_REFINER_CONFIG",
    "MONITOR_BASELINE_REFINER_MODEL",
    "MONITOR_EXPERIMENT_CONFIG",
    "MONITOR_BASELINE_ADAPTIVE_FIXED_INTERVAL",
    "MONITOR_BASELINE_ADAPTIVE_BASE_INTERVAL",
    "MONITOR_BASELINE_ADAPTIVE_GROWTH_STEP",
    "MONITOR_BASELINE_ADAPTIVE_EXECUTION",
    "PLAN_REFINER_RETRY_ATTEMPTS",
}

def normalize_monitor_mode(value: str) -> str:
    if value.strip().lower() in {"realtime", "monitor", "ours"}:
        raise ValueError("The legacy realtime monitor was removed; use liveplan/adaptive_interval for Cadence")
    mode = _MODE_ALIASES.get(value.strip().lower(), value.strip().lower())
    if mode not in VALID_MONITOR_MODES:
        raise ValueError(
            f"monitor_mode must be one of {sorted(VALID_MONITOR_MODES)}, got {value!r}"
        )
    return mode


def parse_ablations(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    raw = value.split(",") if isinstance(value, str) else value
    normalized = tuple(sorted({item.strip().lower().replace("-", "_") for item in raw if item.strip()}))
    unknown = set(normalized) - VALID_ABLATIONS
    if unknown:
        raise ValueError(
            f"Unknown ablation(s) {sorted(unknown)}; valid values are {sorted(VALID_ABLATIONS)}"
        )
    return normalized


@dataclass(frozen=True)
class ExperimentConfig:
    """Serializable configuration for Vanilla, Cadence and LivePlan baselines.

    Agent and monitor models are deliberately separate. A vanilla condition
    clears every monitor environment variable and installs no integration.
    """

    monitor_mode: str = "vanilla"
    agent_model: str = "gpt-5-mini"
    selector_model: str = "gpt-5-mini"
    reviewer_model: str = "gpt-5.2"
    phase_policy_model: str | None = None
    liveplan_variant: str = "monitor"
    liveplan_refiner_model: str | None = None
    liveplan_refiner_config: str = ""
    adaptive_interval_fixed: int = 0
    adaptive_interval_base: int = 5
    adaptive_interval_growth: int = 2
    adaptive_interval_execution: str = "severity"
    ablations: tuple[str, ...] = ()
    phase_policy_startup_mode: str = "cold"

    def __post_init__(self) -> None:
        object.__setattr__(self, "monitor_mode", normalize_monitor_mode(self.monitor_mode))
        object.__setattr__(self, "ablations", parse_ablations(self.ablations))
        variant = self.liveplan_variant.strip().lower()
        if variant not in VALID_LIVEPLAN_VARIANTS:
            raise ValueError(
                f"liveplan_variant must be one of {sorted(VALID_LIVEPLAN_VARIANTS)}"
            )
        object.__setattr__(self, "liveplan_variant", variant)
        if type(self.adaptive_interval_fixed) is not int or self.adaptive_interval_fixed < 0:
            raise ValueError("adaptive_interval_fixed must be a nonnegative integer")
        if self.adaptive_interval_fixed and (self.monitor_mode != "liveplan" or variant != "adaptive_interval"):
            raise ValueError("adaptive_interval_fixed requires liveplan/adaptive_interval")
        if type(self.adaptive_interval_base) is not int or not 1 <= self.adaptive_interval_base <= 20:
            raise ValueError("adaptive_interval_base must be an integer between 1 and 20")
        if type(self.adaptive_interval_growth) is not int or self.adaptive_interval_growth < 1:
            raise ValueError("adaptive_interval_growth must be a positive integer")
        if (self.adaptive_interval_base, self.adaptive_interval_growth) != (5, 2):
            if self.monitor_mode != "liveplan" or variant != "adaptive_interval":
                raise ValueError("adaptive interval overrides require liveplan/adaptive_interval")
            if self.adaptive_interval_fixed:
                raise ValueError("base/growth overrides cannot be combined with fixed scheduling")
        if self.adaptive_interval_execution not in {"severity", "always-advisory", "always-replacement"}:
            raise ValueError("adaptive_interval_execution must be severity, always-advisory, or always-replacement")
        if self.adaptive_interval_execution != "severity":
            if self.monitor_mode != "liveplan" or variant != "adaptive_interval":
                raise ValueError("execution overrides require liveplan/adaptive_interval")
        if self.phase_policy_startup_mode not in {"cold", "warm"}:
            raise ValueError("phase_policy_startup_mode must be cold or warm")

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["ablations"] = list(self.ablations)
        return data

    def apply_environment(
        self,
        *,
        agent_variant: str,
        run_id: str,
        playbook_path: str = "/root/failure_analysis/capability_failure_playbook_compact.md",
        phase_policy_root: str = "/artifacts/phase_policies",
    ) -> None:
        """Apply a clean, mutually exclusive experiment condition."""

        for key in _MONITOR_ENV_KEYS:
            os.environ.pop(key, None)

        os.environ["REALTIME_MONITOR_ENABLED"] = "0"
        os.environ["MONITOR_BASELINE_ENABLED"] = "0"
        os.environ["MONITOR_EXPERIMENT_CONFIG"] = json.dumps(
            self.as_dict(), sort_keys=True
        )

        if self.monitor_mode == "vanilla":
            return

        if self.monitor_mode == "liveplan":
            os.environ["MONITOR_BASELINE_ENABLED"] = "1"
            os.environ["MONITOR_BASELINE_VARIANT"] = self.liveplan_variant
            os.environ["MONITOR_BASELINE_ADAPTIVE_FIXED_INTERVAL"] = str(self.adaptive_interval_fixed)
            os.environ["MONITOR_BASELINE_ADAPTIVE_BASE_INTERVAL"] = str(self.adaptive_interval_base)
            os.environ["MONITOR_BASELINE_ADAPTIVE_GROWTH_STEP"] = str(self.adaptive_interval_growth)
            os.environ["MONITOR_BASELINE_ADAPTIVE_EXECUTION"] = self.adaptive_interval_execution
            # Fixed-interval/full-LivePlan calls are infrastructure, not agent
            # decisions.  A short provider burst must not silently turn a
            # labeled baseline run into one with missing scheduled advice.
            os.environ["PLAN_REFINER_RETRY_ATTEMPTS"] = "8"
            if self.liveplan_refiner_config:
                os.environ["MONITOR_BASELINE_REFINER_CONFIG"] = self.liveplan_refiner_config
            if self.liveplan_refiner_model:
                os.environ["MONITOR_BASELINE_REFINER_MODEL"] = self.liveplan_refiner_model
            return


def experiment_config_from_cli(
    *,
    monitor_mode: str,
    agent_model: str,
    selector_model: str,
    reviewer_model: str,
    phase_policy_model: str = "",
    liveplan_variant: str = "monitor",
    liveplan_refiner_model: str = "",
    liveplan_refiner_config: str = "",
    adaptive_interval_fixed: int = 0,
    adaptive_interval_base: int = 5,
    adaptive_interval_growth: int = 2,
    adaptive_interval_execution: str = "severity",
    ablations: str = "",
    phase_policy_startup_mode: str = "cold",
) -> ExperimentConfig:
    return ExperimentConfig(
        monitor_mode=monitor_mode,
        agent_model=agent_model,
        selector_model=selector_model,
        reviewer_model=reviewer_model,
        phase_policy_model=phase_policy_model or None,
        liveplan_variant=liveplan_variant,
        liveplan_refiner_model=liveplan_refiner_model or None,
        liveplan_refiner_config=liveplan_refiner_config,
        adaptive_interval_fixed=adaptive_interval_fixed,
        adaptive_interval_base=adaptive_interval_base,
        adaptive_interval_growth=adaptive_interval_growth,
        adaptive_interval_execution=adaptive_interval_execution,
        ablations=parse_ablations(ablations),
        phase_policy_startup_mode=phase_policy_startup_mode,
    )


def experiment_config_from_payload(payload: dict[str, object]) -> ExperimentConfig:
    data = dict(payload)
    data["ablations"] = tuple(data.get("ablations") or ())
    return ExperimentConfig(**data)

from __future__ import annotations

from dataclasses import dataclass
import os


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


@dataclass(frozen=True)
class FindingLifecycleConfig:
    """Toggle one-review memory without changing the initial decision."""

    enabled: bool = True

    @classmethod
    def from_env(cls) -> "FindingLifecycleConfig":
        return cls(
            enabled=env_flag("REALTIME_MONITOR_FINDING_LIFECYCLE_ENABLED", True),
        )

    def as_dict(self) -> dict[str, bool]:
        return {"finding_lifecycle_enabled": self.enabled}

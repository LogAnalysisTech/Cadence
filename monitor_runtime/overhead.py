from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from time import perf_counter, time
from typing import Any, Literal, Mapping


Component = Literal["agent", "monitor"]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def extract_token_usage(value: Any) -> dict[str, int]:
    """Normalize OpenAI, Anthropic, LiteLLM, and native agent usage shapes."""

    if value is None:
        data: Mapping[str, Any] = {}
    elif not isinstance(value, Mapping) and hasattr(value, "prompt_tokens"):
        data = {
            "prompt_tokens": getattr(value, "prompt_tokens", 0),
            "completion_tokens": getattr(value, "completion_tokens", 0),
            "cache_read_tokens": getattr(value, "cache_read_tokens", 0),
            "cache_write_tokens": getattr(value, "cache_write_tokens", 0),
            "reasoning_tokens": getattr(value, "reasoning_tokens", 0),
        }
    else:
        data = _as_mapping(value)
    if "usage" in data:
        data = _as_mapping(data["usage"])
    elif "extra" in data:
        extra = _as_mapping(data["extra"])
        response = _as_mapping(extra.get("response"))
        data = _as_mapping(response.get("usage"))

    prompt_details = _as_mapping(data.get("prompt_tokens_details"))
    completion_details = _as_mapping(data.get("completion_tokens_details"))

    def integer(*keys: str) -> int:
        for key in keys:
            raw = data.get(key)
            if raw is not None:
                try:
                    return max(0, int(raw))
                except (TypeError, ValueError):
                    pass
        return 0

    usage = {
        "input_tokens": integer("prompt_tokens", "input_tokens", "tokens_sent"),
        "output_tokens": integer("completion_tokens", "output_tokens", "tokens_received"),
        "cache_read_tokens": integer("cache_read_tokens", "prompt_cache_hit_tokens", "cache_read_input_tokens"),
        "cache_write_tokens": integer("cache_write_tokens", "cache_creation_input_tokens"),
        "reasoning_tokens": integer("reasoning_tokens"),
    }
    if not usage["cache_read_tokens"]:
        usage["cache_read_tokens"] = int(prompt_details.get("cached_tokens") or 0)
    if not usage["reasoning_tokens"]:
        usage["reasoning_tokens"] = int(completion_details.get("reasoning_tokens") or 0)
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


@dataclass
class OverheadEvent:
    event_id: str
    component: Component
    operation: str
    latency_seconds: float
    tokens: dict[str, int]
    timestamp: float
    phase: str | None = None
    step_id: str | None = None
    model: str | None = None
    causal_context: str = "normal"
    outcome: str = "completed"
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "event_id": self.event_id,
            "component": self.component,
            "operation": self.operation,
            "latency_seconds": self.latency_seconds,
            "tokens": dict(self.tokens),
            "timestamp": self.timestamp,
            "causal_context": self.causal_context,
            "outcome": self.outcome,
        }
        if self.phase is not None:
            result["phase"] = self.phase
        if self.step_id is not None:
            result["step_id"] = self.step_id
        if self.model is not None:
            result["model"] = self.model
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


class OverheadLedger:
    """Run-scoped event ledger for direct agent and monitor time/token usage.

    Local phase comparison and bookkeeping are intentionally excluded. Agent
    work induced by an intervention remains agent cost and is linked through
    ``causal_context`` rather than reassigned to the monitor.
    """

    schema_version = "realtime-overhead-1.0"

    def __init__(self) -> None:
        self._counter = count(1)
        self._events: list[OverheadEvent] = []
        self._started_at = time()
        self._started_perf = perf_counter()

    def reset(self) -> None:
        self.__init__()

    def record(
        self,
        *,
        component: Component,
        operation: str,
        latency_seconds: float,
        usage: Any = None,
        phase: str | None = None,
        step_id: str | None = None,
        model: str | None = None,
        causal_context: str = "normal",
        outcome: str = "completed",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        event_id = f"overhead-{next(self._counter)}"
        self._events.append(
            OverheadEvent(
                event_id=event_id,
                component=component,
                operation=operation,
                latency_seconds=max(0.0, float(latency_seconds)),
                tokens=extract_token_usage(usage),
                timestamp=time(),
                phase=phase,
                step_id=step_id,
                model=model,
                causal_context=causal_context,
                outcome=outcome,
                metadata=metadata,
            )
        )
        return event_id

    def update(self, event_id: str | None, **changes: Any) -> None:
        if not event_id:
            return
        event = next((item for item in reversed(self._events) if item.event_id == event_id), None)
        if event is None:
            return
        for key, value in changes.items():
            if hasattr(event, key):
                setattr(event, key, value)

    @property
    def events(self) -> tuple[OverheadEvent, ...]:
        return tuple(self._events)

    def summary(self) -> dict[str, Any]:
        by_component: dict[str, dict[str, Any]] = defaultdict(self._empty_totals)
        by_operation: dict[str, dict[str, Any]] = defaultdict(self._empty_totals)
        for event in self._events:
            self._add_event(by_component[event.component], event)
            self._add_event(by_operation[event.operation], event)

        agent = by_component["agent"]
        monitor = by_component["monitor"]
        recorded_seconds = agent["latency_seconds"] + monitor["latency_seconds"]
        recorded_tokens = agent["tokens"]["total_tokens"] + monitor["tokens"]["total_tokens"]
        elapsed = max(0.0, perf_counter() - self._started_perf)
        return {
            "task_elapsed_seconds": elapsed,
            "recorded_component_seconds": recorded_seconds,
            "recorded_total_tokens": recorded_tokens,
            "monitor_time_ratio_of_task": monitor["latency_seconds"] / elapsed if elapsed else 0.0,
            "monitor_time_ratio_of_recorded": monitor["latency_seconds"] / recorded_seconds if recorded_seconds else 0.0,
            "monitor_token_ratio": monitor["tokens"]["total_tokens"] / recorded_tokens if recorded_tokens else 0.0,
            "by_component": dict(by_component),
            "by_operation": dict(by_operation),
        }

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "started_at": self._started_at,
            "accounting_policy": {
                "metrics": ["time", "tokens"],
                "local_bookkeeping_included": False,
                "intervention_recovery_attribution": "agent",
                "currency_cost_included": False,
            },
            "summary": self.summary(),
            "events": [event.as_dict() for event in self._events],
        }

    @staticmethod
    def _empty_totals() -> dict[str, Any]:
        return {
            "event_count": 0,
            "latency_seconds": 0.0,
            "tokens": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
            },
        }

    @staticmethod
    def _add_event(total: dict[str, Any], event: OverheadEvent) -> None:
        total["event_count"] += 1
        total["latency_seconds"] += event.latency_seconds
        for key, value in event.tokens.items():
            total["tokens"][key] += value

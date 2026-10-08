from __future__ import annotations

from monitor_runtime.overhead import OverheadLedger, extract_token_usage


def test_extract_token_usage_normalizes_provider_shapes() -> None:
    usage = extract_token_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 7},
            }
        }
    )

    assert usage == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_tokens": 40,
        "cache_write_tokens": 0,
        "reasoning_tokens": 7,
        "total_tokens": 120,
    }


def test_extract_token_usage_reads_mini_message_response() -> None:
    usage = extract_token_usage(
        {
            "extra": {
                "response": {
                    "usage": {"input_tokens": 9, "output_tokens": 4},
                }
            }
        }
    )

    assert usage["total_tokens"] == 13


def test_ledger_separates_components_and_attributes_recovery_to_agent() -> None:
    ledger = OverheadLedger()
    proposal = ledger.record(
        component="agent",
        operation="agent_generation",
        latency_seconds=2.0,
        usage={"input_tokens": 80, "output_tokens": 20},
        outcome="proposed",
    )
    ledger.update(proposal, outcome="discarded")
    ledger.record(
        component="monitor",
        operation="phase_reviewer",
        latency_seconds=1.0,
        usage={"input_tokens": 30, "output_tokens": 10},
    )
    ledger.record(
        component="agent",
        operation="agent_generation",
        latency_seconds=3.0,
        usage={"input_tokens": 100, "output_tokens": 25},
        causal_context="intervention_recovery",
        outcome="accepted",
    )

    payload = ledger.payload()
    summary = payload["summary"]
    assert summary["by_component"]["agent"]["tokens"]["total_tokens"] == 225
    assert summary["by_component"]["monitor"]["tokens"]["total_tokens"] == 40
    assert summary["monitor_time_ratio_of_recorded"] == 1 / 6
    assert payload["events"][0]["outcome"] == "discarded"
    assert payload["events"][2]["causal_context"] == "intervention_recovery"
    assert payload["accounting_policy"]["currency_cost_included"] is False

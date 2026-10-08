from __future__ import annotations

from monitor_runtime.provider_error_detection import (
    classify_provider_error,
    scan_log_text,
    scan_provider_errors,
)


def test_ignores_hugging_face_rate_limit_advisory() -> None:
    line = "Warning: unauthenticated requests to the HF Hub. Set HF_TOKEN for higher rate limits."

    assert classify_provider_error(line) is None


def test_classifies_provider_failures() -> None:
    assert classify_provider_error("litellm.RateLimitError: status 429") == "rate_limit"
    assert classify_provider_error("HTTP 402: insufficient credits") == "credit_or_quota"
    assert classify_provider_error("AuthenticationError: invalid API key") == "authentication"
    assert classify_provider_error("Modal concurrency limit exceeded") == "concurrency"


def test_repeated_rate_limits_are_sustained() -> None:
    result = scan_provider_errors(
        [
            "RateLimitError: rate limit exceeded",
            "HTTP 429 Too Many Requests",
            "OpenRouter requests per minute exceeded",
        ]
    )

    assert result["counts"]["rate_limit"] == 3
    assert result["rate_limit_status"] == "sustained"


def test_scans_only_after_latest_marker() -> None:
    result = scan_log_text(
        "429 Too Many Requests\nRUN START\nhealthy\nRUN START\nRateLimitError\n",
        after_marker="RUN START",
    )

    assert result["marker_found"] is True
    assert result["counts"]["rate_limit"] == 1
    assert result["rate_limit_status"] == "isolated"


def test_missing_marker_does_not_scan_historical_log() -> None:
    result = scan_log_text("429 Too Many Requests", after_marker="new run")

    assert result["marker_found"] is False
    assert result["counts"]["rate_limit"] == 0

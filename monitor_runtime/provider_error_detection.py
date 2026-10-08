from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


ERROR_KINDS = (
    "rate_limit",
    "credit_or_quota",
    "authentication",
    "concurrency",
)

_IGNORED_WARNING_PATTERNS = (
    re.compile(r"HF Hub.*higher rate limits", re.IGNORECASE),
    re.compile(r"Hugging Face.*higher rate limits", re.IGNORECASE),
)
_ERROR_PATTERNS = {
    "rate_limit": (
        re.compile(r"\b429\b"),
        re.compile(r"RateLimitError", re.IGNORECASE),
        re.compile(r"rate limit (?:exceeded|reached)", re.IGNORECASE),
        re.compile(r"too many requests", re.IGNORECASE),
        re.compile(r"(?:requests|tokens) per minute", re.IGNORECASE),
    ),
    "credit_or_quota": (
        re.compile(r"\b402\b"),
        re.compile(r"insufficient (?:credit|funds)", re.IGNORECASE),
        re.compile(r"quota (?:exhausted|exceeded)", re.IGNORECASE),
    ),
    "authentication": (
        re.compile(r"AuthenticationError", re.IGNORECASE),
        re.compile(r"authentication fail", re.IGNORECASE),
        re.compile(r"invalid api key", re.IGNORECASE),
        re.compile(r"account associated .* deactivated", re.IGNORECASE),
    ),
    "concurrency": (
        re.compile(r"concurren(?:cy|t).*(?:limit|error|exceed)", re.IGNORECASE),
        re.compile(r"too many concurrent", re.IGNORECASE),
    ),
}


def classify_provider_error(line: str) -> str | None:
    """Classify a genuine provider failure while ignoring advisory warnings."""

    if any(pattern.search(line) for pattern in _IGNORED_WARNING_PATTERNS):
        return None
    for kind in ERROR_KINDS:
        if any(pattern.search(line) for pattern in _ERROR_PATTERNS[kind]):
            return kind
    return None


def scan_provider_errors(
    lines: Iterable[str],
    *,
    sustained_rate_limit_threshold: int = 3,
    example_limit: int = 3,
) -> dict[str, object]:
    """Return provider-error counts and flag repeated rate limiting."""

    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {kind: [] for kind in ERROR_KINDS}
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        kind = classify_provider_error(line)
        if kind is None:
            continue
        counts[kind] += 1
        if len(examples[kind]) < example_limit:
            examples[kind].append(line)

    rate_limit_count = counts["rate_limit"]
    if rate_limit_count == 0:
        rate_limit_status = "clear"
    elif rate_limit_count < sustained_rate_limit_threshold:
        rate_limit_status = "isolated"
    else:
        rate_limit_status = "sustained"

    return {
        "counts": {kind: counts[kind] for kind in ERROR_KINDS},
        "rate_limit_status": rate_limit_status,
        "critical_exhaustion": bool(
            counts["credit_or_quota"] or counts["authentication"]
        ),
        "examples": {kind: values for kind, values in examples.items() if values},
    }


def scan_log_text(
    text: str,
    *,
    after_marker: str = "",
    sustained_rate_limit_threshold: int = 3,
) -> dict[str, object]:
    """Scan only content after the last marker when one is supplied."""

    marker_found = not after_marker
    if after_marker:
        marker_index = text.rfind(after_marker)
        marker_found = marker_index >= 0
        if marker_found:
            text = text[marker_index + len(after_marker) :]
        else:
            text = ""
    result = scan_provider_errors(
        text.splitlines(),
        sustained_rate_limit_threshold=sustained_rate_limit_threshold,
    )
    result["marker_found"] = marker_found
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect provider failures in a run log")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--after-marker", default="")
    parser.add_argument("--sustained-threshold", type=int, default=3)
    args = parser.parse_args()
    result = scan_log_text(
        args.log_path.read_text(encoding="utf-8", errors="replace"),
        after_marker=args.after_marker,
        sustained_rate_limit_threshold=args.sustained_threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

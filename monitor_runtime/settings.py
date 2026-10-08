"""Local configuration loading and the allowlist forwarded to Modal Secrets."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "VOYAGE_API_KEY")
BASE_URLS = ("OPENAI_BASE_URL", "OPENROUTER_BASE_URL", "ANTHROPIC_BASE_URL")


def load_environment(
    env_file: Path | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Fill missing variables from the repo .env, independent of working directory.

    Do not interpolate values: keys containing dollar signs must remain literal.
    Existing variables (including empty values) always take precedence.
    """
    target = os.environ if environ is None else environ
    path = ROOT / ".env" if env_file is None else env_file
    if path.is_file():
        for key, value in dotenv_values(path, interpolate=False).items():
            if value is not None:
                target.setdefault(key, value)
    return target


@dataclass(frozen=True)
class Settings:
    api_route: str
    values: Mapping[str, str] = field(repr=False)

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> Settings:
        route = environ.get("MODEL_API_ROUTE", "").strip().lower()
        if not route:
            legacy = environ.get("REALTIME_MONITOR_USE_OPENROUTER", "1")
            route = "direct" if legacy.strip().lower() in {"0", "false", "no", "off"} else "openrouter"
        if route not in {"openrouter", "direct"}:
            raise ValueError("MODEL_API_ROUTE must be openrouter or direct")
        values = {key: environ[key] for key in (*API_KEYS, *BASE_URLS) if environ.get(key)}
        for key in BASE_URLS:
            if key not in values:
                continue
            url = urlsplit(values[key])
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError(f"{key} must be an HTTP(S) API base URL without credentials, query or fragment")
            values[key] = values[key].rstrip("/")
        values.setdefault("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)
        return cls(api_route=route, values=values)

    def validate(self, *, require_voyage: bool = False, require_monitor: bool = False) -> None:
        """Fail before submitting a job; imports and CLI help need no credentials."""
        if (self.api_route == "openrouter" or require_monitor) and not self.values.get("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter routing and LLM monitor refinement")
        if self.api_route == "direct" and not any(self.values.get(key) for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")):
            raise RuntimeError("Direct routing requires OPENAI_API_KEY or ANTHROPIC_API_KEY")
        if require_voyage and not self.values.get("VOYAGE_API_KEY"):
            raise RuntimeError("VOYAGE_API_KEY is required for Moatless code search")

    def api_environment(self) -> dict[str, str]:
        """Only model/search settings cross into the remote container."""
        values = dict(self.values)
        values["MODEL_API_ROUTE"] = self.api_route
        if self.api_route == "openrouter" and values.get("OPENROUTER_API_KEY"):
            values["OPENAI_API_KEY"] = values["OPENROUTER_API_KEY"]
            values["OPENAI_BASE_URL"] = values["OPENROUTER_BASE_URL"]
        return values


def get_settings() -> Settings:
    return Settings.from_environment(load_environment())

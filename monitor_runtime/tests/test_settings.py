from pathlib import Path
import subprocess
import sys

import pytest

from monitor_runtime.settings import OPENROUTER_BASE_URL, Settings, load_environment


def test_dotenv_only_fills_missing_values(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text('OPENAI_API_KEY=file-key\nVOYAGE_API_KEY=file-voyage\nLITERAL="${UNSET}$value"\n')
    env = {"OPENAI_API_KEY": "process-key", "VOYAGE_API_KEY": ""}
    load_environment(path, environ=env)
    assert env == {"OPENAI_API_KEY": "process-key", "VOYAGE_API_KEY": "", "LITERAL": "${UNSET}$value"}


def test_missing_dotenv_is_optional(tmp_path: Path) -> None:
    assert load_environment(tmp_path / "missing", environ={}) == {}


def test_openrouter_routing_and_secret_allowlist() -> None:
    settings = Settings.from_environment({
        "OPENROUTER_API_KEY": "router-key", "OPENAI_API_KEY": "direct-key",
        "VOYAGE_API_KEY": "voyage-key", "MODAL_TOKEN_SECRET": "modal-secret",
        "UNRELATED_SECRET": "other-secret",
    })
    settings.validate(require_voyage=True, require_monitor=True)
    env = settings.api_environment()
    assert env["OPENAI_API_KEY"] == "router-key"
    assert env["OPENAI_BASE_URL"] == OPENROUTER_BASE_URL
    assert env["VOYAGE_API_KEY"] == "voyage-key"
    assert "MODAL_TOKEN_SECRET" not in env
    assert "UNRELATED_SECRET" not in env
    assert "router-key" not in repr(settings)


def test_custom_base_url_is_forwarded() -> None:
    env = Settings.from_environment({
        "OPENROUTER_API_KEY": "key", "OPENROUTER_BASE_URL": "http://localhost:8000/v1/",
    }).api_environment()
    assert env["OPENAI_BASE_URL"] == env["OPENROUTER_BASE_URL"] == "http://localhost:8000/v1"


def test_direct_route_preserves_provider_endpoint() -> None:
    settings = Settings.from_environment({
        "MODEL_API_ROUTE": "direct", "OPENAI_API_KEY": "direct-key",
        "OPENAI_BASE_URL": "https://example.test/v1/", "OPENROUTER_API_KEY": "monitor-key",
    })
    settings.validate(require_monitor=True)
    assert settings.api_environment()["OPENAI_API_KEY"] == "direct-key"
    assert settings.api_environment()["OPENAI_BASE_URL"] == "https://example.test/v1"


def test_legacy_routing_flag_and_explicit_route_priority() -> None:
    assert Settings.from_environment({"REALTIME_MONITOR_USE_OPENROUTER": "0"}).api_route == "direct"
    assert Settings.from_environment({"MODEL_API_ROUTE": "openrouter", "REALTIME_MONITOR_USE_OPENROUTER": "0"}).api_route == "openrouter"


@pytest.mark.parametrize("env,error", [
    ({}, "OPENROUTER_API_KEY"),
    ({"MODEL_API_ROUTE": "direct"}, "Direct routing"),
])
def test_credentials_checked_on_launch_not_import(env: dict[str, str], error: str) -> None:
    settings = Settings.from_environment(env)
    settings.api_environment()
    with pytest.raises(RuntimeError, match=error):
        settings.validate()


def test_monitor_and_voyage_have_separate_requirements() -> None:
    settings = Settings.from_environment({"MODEL_API_ROUTE": "direct", "OPENAI_API_KEY": "key"})
    settings.validate()
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        settings.validate(require_monitor=True)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        settings.validate(require_voyage=True)


@pytest.mark.parametrize("url", ["ftp://example.test", "https://user:secret@example.test", "https://example.test?key=secret"])
def test_invalid_urls_do_not_expose_values(url: str) -> None:
    with pytest.raises(ValueError) as error:
        Settings.from_environment({"OPENROUTER_BASE_URL": url})
    assert "secret" not in str(error.value)


def test_invalid_route_is_rejected() -> None:
    with pytest.raises(ValueError, match="MODEL_API_ROUTE"):
        Settings.from_environment({"MODEL_API_ROUTE": "typo"})


def test_refiner_client_uses_configured_endpoint(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    # A fresh interpreter exercises the real client without touching network or
    # changing the test process environment; no mock credentials are transmitted.
    result = subprocess.run(
        [sys.executable, "-c", "from plan_refiner.models import OpenRouterModel; "
         "m = OpenRouterModel(model_name='test'); "
         "assert m._api_url == 'https://example.test/v1/chat/completions'"],
        cwd=tmp_path,
        env={"PYTHONPATH": str(root / "monitor_baseline/vendor"),
             "OPENROUTER_API_KEY": "test-key", "OPENROUTER_BASE_URL": "https://example.test/v1/"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

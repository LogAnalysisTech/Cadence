from __future__ import annotations

import subprocess

from monitor_runtime.evaluation.swebench import (
    _evaluation_execution_failure,
    _run_evaluation_script,
    _write_remote_bytes,
)


class _LocalShellEnvironment:
    def execute(self, action: dict[str, str], timeout: int) -> dict[str, object]:
        completed = subprocess.run(
            action["command"],
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "returncode": completed.returncode,
            "output": completed.stdout + completed.stderr,
        }


def test_write_remote_bytes_chunks_large_payload(tmp_path) -> None:
    destination = tmp_path / "large.patch"
    payload = b"diff --git a/a b/a\n" * 10_000

    _write_remote_bytes(
        _LocalShellEnvironment(),
        remote_path=str(destination),
        data=payload,
    )

    assert destination.read_bytes() == payload


def test_evaluation_retains_failed_test_output_after_background_execution(tmp_path) -> None:
    script = tmp_path / "evaluation with spaces.sh"
    script.write_text("sleep 0.3\nprintf 'test failed\\n>>>>> End Test Output\\n'\nexit 1\n")
    result = _run_evaluation_script(
        _LocalShellEnvironment(), script_path=str(script), timeout=5, poll_interval=0.01,
    )
    assert result["returncode"] == 1
    assert "test failed" in result["output"]
    assert _evaluation_execution_failure(result) is None


def test_evaluation_timeout_preserves_output_and_is_not_completed(tmp_path) -> None:
    script = tmp_path / "timeout.sh"
    script.write_text("printf 'partial test output\\n'\nsleep 30\n")
    result = _run_evaluation_script(
        _LocalShellEnvironment(), script_path=str(script), timeout=0.1, poll_interval=0.01,
    )
    assert result["returncode"] == -1
    assert "partial test output" in result["output"]
    assert _evaluation_execution_failure(result) == "evaluation_execution_error"


def test_incomplete_output_is_not_a_completed_evaluation() -> None:
    assert _evaluation_execution_failure({"returncode": 0, "output": "setup failed"}) == "incomplete_test_output"

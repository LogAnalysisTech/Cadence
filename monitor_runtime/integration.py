from __future__ import annotations

from itertools import count
import shlex
from time import perf_counter
from typing import Any, Sequence

from .null_monitor import NullPhaseBoundaryMonitor
from .types import AgentStep, MonitorDecision, MonitorEvent, TrajectoryEvent


class NativePhaseMonitorIntegration:
    """Shared pre-action integration helper for native agent loops.

    Agent-specific adapters are responsible only for converting their native
    trajectory/action shapes to `AgentStep` and for injecting monitor messages
    back into their own prompt state.
    """

    def __init__(self, monitor: Any):
        self.monitor = monitor
        self._step_counter = count(1)
        self._pending_intervention_event_id: str | None = None
        self._pending_advisory_event_id: str | None = None
        self._pending_advisories: dict[str, tuple[MonitorEvent, tuple[str, ...]]] = {}
        self._inflight_advisories: dict[str, tuple[MonitorEvent, tuple[str, ...]]] = {}
        self._advisory_cancellations: list[dict[str, Any]] = []

    def next_step_id(self, prefix: str = "proposed") -> str:
        return f"{prefix}-{next(self._step_counter)}"

    def inspect_step(
        self,
        history: Sequence[TrajectoryEvent],
        proposed_step: AgentStep,
    ) -> MonitorDecision:
        return self.monitor.inspect_before_execution(history, proposed_step)

    def record_agent_generation(
        self,
        *,
        latency_seconds: float,
        usage: Any = None,
        phase: str | None = None,
        step_id: str | None = None,
        model: str | None = None,
        causal_context: str = "normal",
        outcome: str = "proposed",
    ) -> str:
        """Record native model work without attributing recovery work to the monitor."""

        return self.monitor.overhead.record(
            component="agent",
            operation="agent_generation",
            latency_seconds=latency_seconds,
            usage=usage,
            phase=phase,
            step_id=step_id,
            model=model,
            causal_context=causal_context,
            outcome=outcome,
        )

    def record_agent_action(
        self,
        *,
        latency_seconds: float,
        phase: str | None = None,
        step_id: str | None = None,
        causal_context: str = "normal",
        outcome: str = "completed",
        usage: Any = None,
        model: str | None = None,
        generation_event_id: str | None = None,
    ) -> str:
        return self.monitor.overhead.record(
            component="agent",
            operation="agent_action",
            latency_seconds=latency_seconds,
            phase=phase,
            step_id=step_id,
            causal_context=causal_context,
            outcome=outcome,
            usage=usage,
            model=model,
            metadata={"generation_event_id": generation_event_id}
            if generation_event_id
            else None,
        )

    def record_submission_audit(
        self, callback, *, phase: str | None = None, step_id: str | None = None
    ):
        """Time repository inspection as direct monitor work; it consumes no LLM tokens."""

        started = perf_counter()
        try:
            return callback()
        finally:
            self.monitor.overhead.record(
                component="monitor",
                operation="submission_audit",
                latency_seconds=perf_counter() - started,
                phase=phase,
                step_id=step_id,
            )

    def record_repository_identity(self, callback) -> dict[str, Any]:
        """Time the one-off capture of the immutable task repository revision."""

        started = perf_counter()
        try:
            return callback()
        finally:
            self.monitor.overhead.record(
                component="monitor",
                operation="repository_identity",
                latency_seconds=perf_counter() - started,
            )

    def mark_generation_outcome(self, event_id: str | None, outcome: str) -> None:
        self.monitor.overhead.update(event_id, outcome=outcome)

    def associate_generation(
        self, event_id: str | None, *, phase: str | None, step_id: str | None
    ) -> None:
        self.monitor.overhead.update(event_id, phase=phase, step_id=step_id)

    def overhead_payload(self) -> dict[str, Any]:
        payload = self.monitor.overhead.payload()
        metrics = getattr(self.monitor, "lifecycle_metrics", None)
        if isinstance(metrics, dict):
            payload["monitor_lifecycle"] = metrics
        payload["monitor_advisory_cancellations"] = list(self._advisory_cancellations)
        return payload

    def reset_run_state(self) -> None:
        """Reset monitor memory, accounting, and adapter-local task state."""

        self.monitor.reset_run_state()
        self._step_counter = count(1)
        self._pending_intervention_event_id = None
        self._pending_advisory_event_id = None
        self._pending_advisories = {}
        self._inflight_advisories = {}
        self._advisory_cancellations = []

    def note_intervention(self, event_id: str) -> None:
        self._pending_intervention_event_id = event_id

    def consume_generation_context(self) -> str:
        if self._pending_intervention_event_id is None:
            if self._pending_advisory_event_id is None:
                return "normal"
            self._pending_advisory_event_id = None
            return "advisory_influenced"
        self._pending_intervention_event_id = None
        return "intervention_recovery"

    def queue_advisory(self, step_id: str, decision: MonitorDecision) -> None:
        if decision.advisory_event is None:
            return
        self._pending_advisories[step_id] = (
            decision.advisory_event,
            decision.retire_finding_ids_after_advice,
        )

    def consume_advisory(self, step_id: str | None) -> MonitorEvent | None:
        if not step_id:
            return None
        pending = self._pending_advisories.pop(step_id, None)
        if pending is None:
            return None
        event, finding_ids = pending
        self._inflight_advisories[step_id] = (event, finding_ids)
        return event

    def confirm_advisory_delivery(self, step_id: str | None) -> None:
        """Commit lifecycle state only after guidance reaches agent-visible history."""

        if not step_id:
            return
        inflight = self._inflight_advisories.pop(step_id, None)
        if inflight is None:
            return
        event, finding_ids = inflight
        self.monitor.confirm_advisory_delivered(finding_ids, step_id=step_id)
        self._pending_advisory_event_id = event.event_id

    def cancel_advisory_after_action_failure(
        self, step_id: str | None, error: BaseException
    ) -> dict[str, Any] | None:
        """Cancel deferred advice when its action failed before observation delivery."""

        return self._cancel_advisory(
            step_id, error, reason="action_failed", count_action_failure=True
        )

    def cancel_advisory_after_delivery_failure(
        self, step_id: str | None, error: BaseException
    ) -> dict[str, Any] | None:
        """Cancel advice whose post-action injection failed without blaming the action."""

        return self._cancel_advisory(
            step_id,
            error,
            reason="advisory_delivery_failed",
            count_action_failure=False,
        )

    def cancel_advisory_after_terminal_control_flow(
        self, step_id: str | None, error: BaseException
    ) -> dict[str, Any] | None:
        """Cancel advice when an executed action terminates without another generation."""

        return self._cancel_advisory(
            step_id,
            error,
            reason="terminal_forfeit_after_execution",
            count_action_failure=False,
        )

    def _cancel_advisory(
        self,
        step_id: str | None,
        error: BaseException,
        *,
        reason: str,
        count_action_failure: bool,
    ) -> dict[str, Any] | None:

        if not step_id:
            return None
        pending = self._pending_advisories.pop(step_id, None)
        if pending is None:
            pending = self._inflight_advisories.pop(step_id, None)
        if pending is None:
            return None
        event, finding_ids = pending
        record = {
            "event_id": event.event_id,
            "step_id": step_id,
            "reason": reason,
            "error_type": type(error).__name__,
            "finding_ids_preserved": list(finding_ids),
        }
        self._advisory_cancellations.append(record)
        if count_action_failure:
            self.monitor.note_advisory_cancelled_action_failed()
        return record

    def confirm_blocking_guidance_delivery(
        self, decision: MonitorDecision, *, step_id: str | None
    ) -> None:
        """Retire old findings whose final reminder rode with a fresh BLOCK."""

        finding_ids = decision.retire_finding_ids_after_advice
        if finding_ids:
            self.monitor.confirm_blocking_guidance_delivered(
                finding_ids, step_id=step_id
            )

    @staticmethod
    def format_error_nudge_message(attempt: int, limit: int) -> str:
        """Deterministic corrective nudge injected one attempt before a hard format-error kill.

        This does not call the LLM reviewer: there is no valid proposed step to review yet,
        so the only useful action is a cheap, templated reminder of the required output shape.
        """

        return (
            f"Your last {attempt} response(s) failed to parse as a valid structured tool call. "
            f"This framework stops the task after {limit} consecutive failures, and you are one "
            "away from that limit. Before responding again: emit exactly one tool call that "
            "matches the required schema exactly, including the required `phase` argument, "
            "with no prose outside the tool call."
        )

    @staticmethod
    def intervention_prompt(event: MonitorEvent, action_contract: str) -> str:
        """Render the shared intervention followed by one native action contract."""

        return (
            f"{event.message}\n\n"
            "Protocol requirement: treat this as monitor feedback, not a new agent phase. "
            f"Your next response must be concise and {action_contract}. "
            "Make concrete progress toward the required evidence while retaining freedom "
            "to choose the safest implementation."
        )

    @staticmethod
    def advisory_prompt(event: MonitorEvent) -> str:
        return (
            f"{event.message}\n\n"
            "Protocol requirement: this is non-blocking monitor guidance about the "
            "completed action. Use it when choosing your next step; do not treat it as "
            "an executed action or as a new agent phase."
        )

    @staticmethod
    def decision_payload(decision: MonitorDecision) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decision": decision.reason,
            "checked_phase": decision.checked_phase,
            "execute_proposed_step": decision.execute_proposed_step,
            "disposition": decision.disposition,
        }
        if decision.selected_capabilities is not None:
            payload["selected_capabilities"] = list(
                decision.selected_capabilities.capability_ids
            )
        if decision.review is not None:
            payload["review_status"] = decision.review.status
            payload["enforcement_verdict"] = decision.review.verdict
            payload["creates_new_finding"] = decision.review.creates_new_finding
            payload["review_severity"] = decision.review.severity
            payload["overall_recommendation"] = decision.review.overall_recommendation
            payload["blocking_conditions"] = list(decision.review.blocking_conditions)
            payload["required_evidence"] = list(decision.review.required_evidence)
            payload["resolved_finding_ids"] = list(decision.review.resolved_finding_ids)
            payload["unresolved_finding_ids"] = list(
                decision.review.unresolved_finding_ids
            )
            payload["superseded_review_ids"] = list(
                decision.review.superseded_review_ids
            )
            payload["repeated_advisory_review_ids"] = list(
                decision.review.repeated_advisory_review_ids
            )
            payload["verification_evidence_assessment"] = dict(
                decision.review.verification_evidence_assessment
            )
            payload["related_finding_id"] = decision.review.related_finding_id
            payload["evidence_anchors"] = list(decision.review.evidence_anchors)
            payload["blocking_evidence_kind"] = decision.review.blocking_evidence_kind
            payload["blocking_evidence_step_ids"] = list(
                decision.review.blocking_evidence_step_ids
            )
            payload["submission_issue"] = decision.review.submission_issue
            prompt_cache = {
                key: decision.review.metadata[key]
                for key in (
                    "prompt_layout_version",
                    "static_prefix_chars",
                    "static_prefix_sha256",
                    "task_prefix_chars",
                    "task_prefix_sha256",
                    "capability_prefix_chars",
                    "capability_prefix_sha256",
                    "full_prompt_chars",
                    "dynamic_prompt_chars",
                    "selected_capabilities",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "observations_truncated",
                    "observation_original_chars",
                    "observation_retained_chars",
                    "observation_chars_saved",
                )
                if key in decision.review.metadata
            }
            if prompt_cache:
                payload["review_prompt_cache"] = prompt_cache
            if decision.review.metadata.get("block_contract_missing_fields"):
                payload["block_contract_missing_fields"] = list(
                    decision.review.metadata["block_contract_missing_fields"]
                )
        if decision.advisory_event is not None:
            payload["advisory_event_id"] = decision.advisory_event.event_id
            payload["retire_finding_ids_after_advice"] = list(
                decision.retire_finding_ids_after_advice
            )
        if decision.normalization_reason:
            payload["normalization_reason"] = decision.normalization_reason
        if decision.timing:
            payload["timing"] = dict(decision.timing)
        if decision.memory_snapshot:
            payload["review_memory"] = decision.memory_snapshot
        if decision.phase_policy:
            payload["phase_policy"] = dict(decision.phase_policy)
        return payload

    @staticmethod
    def event_record(event: MonitorEvent) -> dict[str, Any]:
        return {
            "event_type": event.event_type,
            "event_id": event.event_id,
            "phase": event.phase,
            "message": event.message,
            "intervention_for_phase": event.intervention_for_phase,
            "discarded_proposed_step_id": event.discarded_proposed_step_id,
            "selected_capabilities": list(event.selected_capabilities),
            "verdict": event.verdict,
            "metadata": event.metadata,
        }

    @staticmethod
    def stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            command = (
                value.get("command") or value.get("name") or value.get("tool_name")
            )
            if command is not None:
                return str(command)
        name = getattr(value, "name", None)
        if name is not None:
            return str(name)
        return str(value)

    @staticmethod
    def repository_identity(execute) -> dict[str, Any]:
        """Capture an immutable repository root and HEAD before agent actions run."""

        command = (
            "repo_root=$(git rev-parse --show-toplevel 2>/dev/null); root_rc=$?; "
            'if [ "$root_rc" -eq 0 ]; then '
            "base_commit=$(git -C \"$repo_root\" rev-parse --verify 'HEAD^{commit}' 2>/dev/null); "
            "commit_rc=$?; else base_commit=; commit_rc=1; fi; "
            "printf '__RAFDMON_REPO_ROOT__=%s\\n' \"$repo_root\"; "
            "printf '__RAFDMON_BASE_COMMIT__=%s\\n' \"$base_commit\"; "
            "printf '__RAFDMON_IDENTITY_RC__=%s\\n' \"$commit_rc\""
        )
        try:
            result = execute(command)
        except Exception as exc:
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        output = (
            str(result.get("output", "")) if isinstance(result, dict) else str(result)
        )
        normalized = output.replace("\r\n", "\n").replace("\r", "\n")
        values: dict[str, str] = {}
        for line in normalized.splitlines():
            stripped = line.strip()
            for marker, key in (
                ("__RAFDMON_REPO_ROOT__=", "repo_root"),
                ("__RAFDMON_BASE_COMMIT__=", "base_commit"),
                ("__RAFDMON_IDENTITY_RC__=", "returncode"),
            ):
                if stripped.startswith(marker):
                    values[key] = stripped.split("=", 1)[1]
        available = (
            values.get("returncode") == "0"
            and bool(values.get("repo_root"))
            and bool(values.get("base_commit"))
        )
        return {
            "available": available,
            "repo_root": values.get("repo_root"),
            "base_commit": values.get("base_commit"),
            "returncode": values.get("returncode"),
            **({} if available else {"error": output[-1000:]}),
        }

    @staticmethod
    def repository_submission_audit(
        execute,
        *,
        base_commit: str | None = None,
        excluded_patch_path: str | None = None,
        repo_root: str | None = None,
        patch_path: str = "patch.txt",
    ) -> dict[str, Any]:
        """Inspect the submitted diff and validate an existing patch artifact."""

        if repo_root:
            repository_setup = f"repo_root={shlex.quote(repo_root)}; repo_rc=0; "
        else:
            repository_setup = (
                "repo_root=$(git rev-parse --show-toplevel 2>/dev/null); repo_rc=$?; "
            )
        if patch_path.startswith("/"):
            artifact_setup = f"artifact_path={shlex.quote(patch_path)}; "
        else:
            artifact_setup = (
                f"artifact_rel={shlex.quote(patch_path)}; "
                'artifact_path="$repo_root/$artifact_rel"; '
            )

        if base_commit:
            # An alternate index mirrors `git add -A` and can remove a benchmark
            # test patch without changing the agent's worktree or staging area.
            diff_command = (
                f"base_ref={shlex.quote(base_commit)}; "
                f"excluded_patch={shlex.quote(excluded_patch_path or '')}; "
                'tmp_index=$(mktemp); rm -f "$tmp_index"; '
                'if [ "$repo_rc" -eq 0 ] '
                '&& GIT_INDEX_FILE="$tmp_index" git -C "$repo_root" read-tree "$base_ref" >/dev/null 2>&1 '
                '&& GIT_INDEX_FILE="$tmp_index" git -C "$repo_root" add -A -- >/dev/null 2>&1; then '
                'if [ -n "$excluded_patch" ] && [ -s "$excluded_patch" ]; then '
                'GIT_INDEX_FILE="$tmp_index" git -C "$repo_root" apply -R --cached < "$excluded_patch" '
                ">/dev/null 2>&1 || true; fi; "
                'GIT_INDEX_FILE="$tmp_index" git -C "$repo_root" diff --cached --quiet "$base_ref" --; diff_rc=$?; '
                "else diff_rc=2; fi; "
                "printf '__RAFDMON_DIFF_RC__=%s\\n' \"$diff_rc\"; "
                'if [ "$diff_rc" -eq 0 ] || [ "$diff_rc" -eq 1 ]; then '
                'GIT_INDEX_FILE="$tmp_index" git -C "$repo_root" diff --cached "$base_ref" '
                "--name-only --no-ext-diff; fi; "
                'rm -f "$tmp_index"; '
            )
        else:
            diff_command = (
                'if [ "$repo_rc" -eq 0 ]; then git -C "$repo_root" diff --quiet HEAD --; '
                "diff_rc=$?; else diff_rc=2; fi; "
                "printf '__RAFDMON_DIFF_RC__=%s\\n' \"$diff_rc\"; "
                'if [ "$diff_rc" -eq 0 ] || [ "$diff_rc" -eq 1 ]; then '
                'git -C "$repo_root" diff HEAD --name-only --no-ext-diff; fi; '
            )
        command = (
            repository_setup
            + artifact_setup
            + diff_command
            + 'if [ -e "$artifact_path" ]; then artifact_exists=1; else artifact_exists=0; fi; '
            'if [ -s "$artifact_path" ] && grep -q \'^diff --git \' "$artifact_path"; then artifact_nonempty=1; '
            "else artifact_nonempty=0; fi; "
            "printf '__RAFDMON_ARTIFACT_EXISTS__=%s\\n' \"$artifact_exists\"; "
            "printf '__RAFDMON_ARTIFACT_NONEMPTY__=%s\\n' \"$artifact_nonempty\"; "
            "apply_available=0; apply_rc=; apply_error=; "
            'if [ "$artifact_nonempty" -eq 1 ] && [ -n "${base_ref:-}" ]; then '
            'apply_index=$(mktemp); apply_log=$(mktemp); rm -f "$apply_index"; '
            'if GIT_INDEX_FILE="$apply_index" git -C "$repo_root" read-tree "$base_ref" >"$apply_log" 2>&1; then '
            'GIT_INDEX_FILE="$apply_index" git -C "$repo_root" apply --cached --check -- "$artifact_path" >"$apply_log" 2>&1; '
            "apply_rc=$?; apply_available=1; "
            'else apply_error="could not prepare initial-commit index"; fi; '
            'if [ -z "$apply_error" ] && [ -s "$apply_log" ]; then '
            "apply_error=$(tail -c 1000 \"$apply_log\" | tr '\\r\\n' '  '); fi; "
            'rm -f "$apply_index" "$apply_log"; fi; '
            "printf '__RAFDMON_APPLY_AVAILABLE__=%s\\n' \"$apply_available\"; "
            "printf '__RAFDMON_APPLY_RC__=%s\\n' \"$apply_rc\"; "
            "printf '__RAFDMON_APPLY_ERROR__=%s\\n' \"$apply_error\"; "
            "printf '__RAFDMON_STATUS__\\n'; "
            'git -C "$repo_root" status --porcelain --untracked-files=all'
        )
        try:
            result = execute(command)
        except Exception as exc:
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        if isinstance(result, dict):
            output = str(result.get("output", ""))
            returncode = result.get("returncode")
        else:
            output = str(result)
            returncode = 0
        # Persistent PTY-backed shells commonly return CRLF and may echo the
        # submitted command before its output. Parse complete marker lines
        # instead of assuming the diff marker is the first output line.
        normalized_output = output.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized_output.splitlines()
        diff_marker_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip().startswith("__RAFDMON_DIFF_RC__=")
            ),
            None,
        )
        status_marker_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip() == "__RAFDMON_STATUS__"
                and (diff_marker_index is None or index > diff_marker_index)
            ),
            None,
        )
        if diff_marker_index is None or status_marker_index is None:
            return {
                "available": False,
                "returncode": returncode,
                "error": output[-1000:],
            }
        first_line = lines[diff_marker_index].strip()
        detail_lines = lines[diff_marker_index + 1 : status_marker_index]
        status_text = "\n".join(lines[status_marker_index + 1 :])
        try:
            diff_rc = int(first_line.split("=", 1)[1])
        except (IndexError, ValueError):
            return {
                "available": False,
                "returncode": returncode,
                "error": output[-1000:],
            }
        artifact_exists = None
        artifact_nonempty = None
        apply_available = False
        apply_returncode = None
        apply_error = ""
        changed_files = []
        for line in detail_lines:
            if line.startswith("__RAFDMON_ARTIFACT_EXISTS__="):
                artifact_exists = line.endswith("=1")
            elif line.startswith("__RAFDMON_ARTIFACT_NONEMPTY__="):
                artifact_nonempty = line.endswith("=1")
            elif line.startswith("__RAFDMON_APPLY_AVAILABLE__="):
                apply_available = line.endswith("=1")
            elif line.startswith("__RAFDMON_APPLY_RC__="):
                raw_apply_rc = line.split("=", 1)[1].strip()
                if raw_apply_rc:
                    try:
                        apply_returncode = int(raw_apply_rc)
                    except ValueError:
                        apply_available = False
            elif line.startswith("__RAFDMON_APPLY_ERROR__="):
                apply_error = line.split("=", 1)[1].strip()
            elif line.strip():
                changed_files.append(line)
        repository_diff_nonempty = diff_rc == 1
        patch_nonempty = (
            artifact_nonempty if artifact_exists else repository_diff_nonempty
        )
        all_status_entries = [line for line in status_text.splitlines() if line.strip()]
        artifact_status_entries = [
            line for line in all_status_entries if line.strip() == "?? patch.txt"
        ]
        return {
            "available": diff_rc in {0, 1},
            "patch_nonempty": patch_nonempty,
            "repository_diff_nonempty": repository_diff_nonempty,
            "patch_artifact_exists": artifact_exists,
            "patch_artifact_nonempty": artifact_nonempty,
            "patch_applicability_available": apply_available,
            "patch_applicable": apply_returncode == 0 if apply_available else None,
            "patch_apply_returncode": apply_returncode,
            "patch_apply_error": apply_error,
            "changed_files": changed_files,
            "status_entries": [
                line
                for line in all_status_entries
                if line not in artifact_status_entries
            ],
            "submission_artifact_status_entries": artifact_status_entries,
            "base_commit": base_commit,
            "repo_root": repo_root,
            "patch_path": patch_path,
            "excluded_patch_path": excluded_patch_path,
            "returncode": returncode,
        }


class PassiveAgentTelemetry(NativePhaseMonitorIntegration):
    """Record native agent work without enabling any monitor behavior.

    Vanilla runs use this integration only for the shared overhead ledger. It
    never inspects an action, requests a phase, changes a prompt, or injects a
    message, so the native control flow remains the experimental baseline.
    """

    def __init__(self) -> None:
        super().__init__(NullPhaseBoundaryMonitor())

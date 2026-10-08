from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from monitor_runtime.settings import get_settings

if TYPE_CHECKING:
    from monitor_runtime.experiment_config import ExperimentConfig


def api_secret_dict() -> dict[str, str]:
    """Load the shared credential allowlist for Modal's encrypted Secret injection."""
    return get_settings().api_environment()


def validate_experiment_credentials(experiment: ExperimentConfig, *, require_voyage: bool = False) -> None:
    get_settings().validate(
        require_voyage=require_voyage,
        require_monitor=experiment.monitor_mode == "liveplan" and experiment.liveplan_variant != "monitor",
    )


def api_secret():
    """Expose configured model credentials to a Modal container."""

    import modal

    return modal.Secret.from_dict(api_secret_dict())


def modal_secret():
    """Forward the active Modal profile for nested SWE-ReX deployments."""

    import modal
    from modal import config as modal_config

    token_id = modal_config.config.get("token_id")
    token_secret = modal_config.config.get("token_secret")
    if not token_id or not token_secret:
        return modal.Secret.from_dict({})
    return modal.Secret.from_dict(
        {
            "MODAL_TOKEN_ID": token_id,
            "MODAL_TOKEN_SECRET": token_secret,
        }
    )


def configure_monitor(
    selector_model: str,
    reviewer_model: str,
    *,
    agent_variant: str,
    run_id: str = "local",
    enable_monitor: bool = True,
    phase_policy_enabled: bool = False,
    phase_policy_startup_mode: str = "cold",
    phase_policy_model: str | None = None,
    phase_policy_root: str = "/artifacts/phase_policies",
    finding_lifecycle_enabled: bool = True,
) -> None:
    """Accept monitor-off legacy calls; reject the removed monitor explicitly."""

    from monitor_runtime.experiment_config import ExperimentConfig

    if enable_monitor:
        raise ValueError("The legacy realtime monitor was removed; configure Cadence explicitly")
    configure_experiment(ExperimentConfig(monitor_mode="vanilla"), agent_variant=agent_variant, run_id=run_id)


def configure_experiment(
    config: "ExperimentConfig",
    *,
    agent_variant: str,
    run_id: str,
    phase_policy_root: str = "/artifacts/phase_policies",
) -> None:
    """Apply the unified experiment configuration inside a worker process."""

    import os

    config.apply_environment(
        agent_variant=agent_variant,
        run_id=run_id,
        phase_policy_root=phase_policy_root,
    )
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")


def runner_experiment_config(
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
    legacy_enable_monitor: bool = True,
    legacy_phase_policy_enabled: bool = True,
    legacy_finding_lifecycle_enabled: bool = True,
) -> "ExperimentConfig":
    """Resolve shared CLI flags, retaining compatibility with old runners."""

    from monitor_runtime.experiment_config import (
        experiment_config_from_cli,
        parse_ablations,
    )

    resolved_mode = monitor_mode.strip() or (
        "realtime" if legacy_enable_monitor else "vanilla"
    )
    resolved_ablations = parse_ablations(ablations)
    return experiment_config_from_cli(
        monitor_mode=resolved_mode,
        agent_model=agent_model,
        selector_model=selector_model,
        reviewer_model=reviewer_model,
        phase_policy_model=phase_policy_model,
        liveplan_variant=liveplan_variant,
        liveplan_refiner_model=liveplan_refiner_model,
        liveplan_refiner_config=liveplan_refiner_config,
        adaptive_interval_fixed=adaptive_interval_fixed,
        adaptive_interval_base=adaptive_interval_base,
        adaptive_interval_growth=adaptive_interval_growth,
        adaptive_interval_execution=adaptive_interval_execution,
        ablations=",".join(sorted(resolved_ablations)),
        phase_policy_startup_mode=phase_policy_startup_mode,
    )


def configure_runner_experiment(
    *,
    experiment_payload: dict[str, object] | None,
    agent_variant: str,
    run_id: str,
    agent_model: str,
    selector_model: str,
    reviewer_model: str,
    phase_policy_startup_mode: str,
    legacy_enable_monitor: bool,
    legacy_phase_policy_enabled: bool,
    legacy_finding_lifecycle_enabled: bool = True,
) -> "ExperimentConfig":
    """Configure a worker from the new payload or legacy positional flags."""

    if experiment_payload is None:
        config = runner_experiment_config(
            monitor_mode="",
            agent_model=agent_model,
            selector_model=selector_model,
            reviewer_model=reviewer_model,
            phase_policy_startup_mode=phase_policy_startup_mode,
            legacy_enable_monitor=legacy_enable_monitor,
            legacy_phase_policy_enabled=legacy_phase_policy_enabled,
            legacy_finding_lifecycle_enabled=legacy_finding_lifecycle_enabled,
        )
    else:
        from monitor_runtime.experiment_config import experiment_config_from_payload

        config = experiment_config_from_payload(experiment_payload)
    configure_experiment(config, agent_variant=agent_variant, run_id=run_id)
    return config


def materialize_phase_policy_snapshot(
    *,
    run_id: str,
    agent_variant: str,
    startup_mode: str,
    enabled: bool,
    store_root: str = "/artifacts/phase_policies",
    playbook_path: str = "/root/failure_analysis/capability_failure_playbook_compact.md",
) -> dict[str, object] | None:
    """Compatibility hook for runner payloads; legacy phase policies are removed."""

    if enabled:
        raise ValueError("The legacy realtime phase-policy monitor was removed")
    return None


def extract_monitor_artifacts(
    instance_id: str,
    traj_path: Path,
    instance_dir: Path,
    *,
    include_monitor_views: bool = True,
) -> None:
    """Materialize telemetry and, when enabled, monitor-specific trajectory views."""

    if not traj_path.exists():
        return
    instance_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    overhead_events = ((data.get("info") or {}).get("overhead") or {}).get("events") or []
    monitor_events_by_step: dict[str, list[dict]] = {}
    for event in overhead_events:
        step_id = event.get("step_id")
        if step_id and event.get("component") == "monitor":
            monitor_events_by_step.setdefault(step_id, []).append(event)

    def event_links(step_id: str | None) -> dict[str, object]:
        linked = monitor_events_by_step.get(step_id or "", [])
        review_events = [event for event in linked if event.get("operation") == "phase_reviewer"]
        check_events = [
            event
            for event in linked
            if event.get("operation") in {"phase_reviewer", "rule_check", "refiner_call"}
        ]
        return {
            "review_executed": bool(review_events),
            "review_event_ids": [event.get("event_id") for event in review_events],
            "check_executed": bool(check_events),
            "check_event_ids": [event.get("event_id") for event in check_events],
            "monitor_operations": [event.get("operation") for event in linked],
            "monitor_overhead_event_ids": [event.get("event_id") for event in linked],
        }

    def discarded_step_id(extra: dict) -> str | None:
        step_id = extra.get("discarded_proposed_step_id")
        if step_id:
            return str(step_id)
        event_id = str(extra.get("event_id") or "")
        if event_id.startswith("monitor_baseline-"):
            return event_id.removeprefix("monitor_baseline-")
        return None

    decisions = []
    interventions = []
    guidance = []
    phases = []
    for idx, message in enumerate(data.get("messages", [])):
        extra = message.get("extra") or {}
        if "phase_monitor" in extra:
            proposed_step_id = extra.get("overhead_step_id")
            decisions.append(
                {
                    "message_index": idx,
                    "trajectory_location": {"container": "messages", "index": idx},
                    "proposed_step_id": proposed_step_id,
                    **event_links(proposed_step_id),
                    "role": message.get("role"),
                    "phase": extra.get("phase") or extra.get("current_phase"),
                    "phase_monitor": extra.get("phase_monitor"),
                    "action": (extra.get("actions") or [None])[0],
                }
            )
        if extra.get("message_type") == "monitor_intervention":
            proposed_step_id = discarded_step_id(extra)
            interventions.append(
                {
                    "message_index": idx,
                    "trajectory_location": {"container": "messages", "index": idx},
                    "proposed_step_id": proposed_step_id,
                    **event_links(proposed_step_id),
                    "content": message.get("content", ""),
                    "extra": extra,
                }
            )
        if extra.get("message_type") == "monitor_guidance":
            proposed_step_id = extra.get("monitor_baseline_step_id") or (
                extra.get("metadata") or {}
            ).get("executed_step_id")
            guidance.append(
                {
                    "message_index": idx,
                    "trajectory_location": {"container": "messages", "index": idx},
                    "proposed_step_id": proposed_step_id,
                    **event_links(str(proposed_step_id) if proposed_step_id is not None else None),
                    "content": message.get("content", ""),
                    "extra": extra,
                }
            )
        phase = extra.get("phase") or extra.get("current_phase")
        if phase:
            phases.append({"message_index": idx, "phase": phase})
    if include_monitor_views:
        write_json(instance_dir / "monitor_decisions.json", decisions)
        write_json(instance_dir / "monitor_interventions.json", interventions)
        write_json(instance_dir / "monitor_guidance.json", guidance)
        write_json(instance_dir / "phase_trace.json", phases)
        timeline = sorted(
            [
                {
                    "event_kind": "inspection",
                    **decision,
                }
                for decision in decisions
            ]
            + [
                {
                    "event_kind": "intervention",
                    **intervention,
                }
                for intervention in interventions
            ]
            + [
                {
                    "event_kind": "guidance",
                    **item,
                }
                for item in guidance
            ],
            key=lambda item: item["message_index"],
        )
        write_json(instance_dir / "monitor_timeline.json", timeline)
    write_json(
        instance_dir / "overhead.json", (data.get("info") or {}).get("overhead") or {}
    )
    manifest = {
        "instance_id": instance_id,
        "trajectory": str(traj_path),
        "overhead": str(instance_dir / "overhead.json"),
        "patch": str(instance_dir / f"{instance_id}.patch"),
    }
    if include_monitor_views:
        manifest.update(
            {
                "monitor_decisions": str(instance_dir / "monitor_decisions.json"),
                "monitor_interventions": str(instance_dir / "monitor_interventions.json"),
                "monitor_guidance": str(instance_dir / "monitor_guidance.json"),
                "phase_trace": str(instance_dir / "phase_trace.json"),
                "monitor_timeline": str(instance_dir / "monitor_timeline.json"),
            }
        )
    write_json(
        instance_dir / "artifact_manifest.json",
        manifest,
    )


def summarize_instance(traj_path: Path) -> dict[str, object]:
    """Return compact execution and monitor metrics from one saved trajectory."""

    if not traj_path.exists():
        return {"trajectory_exists": False}
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    messages = data.get("messages", [])
    decisions = [
        message.get("extra", {}).get("phase_monitor")
        for message in messages
        if message.get("extra", {}).get("phase_monitor")
    ]
    interventions = [
        message
        for message in messages
        if message.get("extra", {}).get("message_type") == "monitor_intervention"
    ]
    guidance = [
        message
        for message in messages
        if message.get("extra", {}).get("message_type") == "monitor_guidance"
    ]
    review_timings = [
        decision["timing"]
        for decision in decisions
        if isinstance(decision, dict) and isinstance(decision.get("timing"), dict)
    ]
    review_timings.extend(
        timing
        for message in interventions
        if isinstance(
            timing := ((message.get("extra") or {}).get("metadata") or {}).get(
                "timing"
            ),
            dict,
        )
    )
    timestamps = []
    for message in messages:
        extra = message.get("extra") or {}
        timestamp = extra.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            timestamp = (extra.get("response") or {}).get("timestamp")
        if isinstance(timestamp, (int, float)):
            timestamps.append(float(timestamp))
    task_active_seconds = (
        max(timestamps) - min(timestamps) if len(timestamps) >= 2 else None
    )
    selector_seconds = sum(
        float(timing.get("selector_latency_seconds", 0)) for timing in review_timings
    )
    reviewer_seconds = sum(
        float(timing.get("reviewer_latency_seconds", 0)) for timing in review_timings
    )
    monitor_seconds = selector_seconds + reviewer_seconds
    info = data.get("info", {})
    exit_status = str(info.get("exit_status") or "")
    submission = info.get("submission") or ""
    autosubmitted = exit_status.lower().startswith("submitted (") or any(
        "autosubmitted" in str(message.get("content", "")).lower()
        for message in messages
    )
    submission_path = (
        "autosubmission"
        if autosubmitted
        else "explicit"
        if submission
        else "none"
    )
    overhead = info.get("overhead") or {}
    operation_summary = (overhead.get("summary") or {}).get("by_operation") or {}

    def operation_count(name: str) -> int:
        return int((operation_summary.get(name) or {}).get("event_count") or 0)

    phase_review_count = operation_count("phase_reviewer")
    if not phase_review_count and not operation_summary:
        # Compatibility for older trajectories that predate the overhead ledger.
        phase_review_count = sum(
            bool((decision or {}).get("selected_capabilities"))
            for decision in decisions
        )
    baseline_rule_checks = operation_count("rule_check")
    baseline_refiner_calls = operation_count("refiner_call")
    baseline_refiner_cooldowns = operation_count("refiner_cooldown")
    check_count = phase_review_count + baseline_rule_checks
    if not phase_review_count and not baseline_rule_checks:
        check_count = baseline_refiner_calls
    return {
        "trajectory_exists": True,
        "exit_status": info.get("exit_status"),
        "submission_chars": len(submission),
        "submission_path": submission_path,
        "api_calls": (info.get("model_stats") or {}).get("api_calls"),
        "instance_cost": (info.get("model_stats") or {}).get("instance_cost"),
        "actions_inspected": len(decisions) + len(interventions),
        "checks": check_count,
        "phase_reviews_executed": phase_review_count,
        "phase_policy_resolutions": operation_count("phase_policy_resolver"),
        "baseline_rule_checks": baseline_rule_checks,
        "baseline_refiner_calls": baseline_refiner_calls,
        "baseline_refiner_cooldowns": baseline_refiner_cooldowns,
        "agent_generations": operation_count("agent_generation"),
        "agent_actions": operation_count("agent_action"),
        "monitor_timing": {
            "timed_reviews": len(review_timings),
            "selector_seconds": selector_seconds,
            "reviewer_seconds": reviewer_seconds,
            "monitor_seconds": monitor_seconds,
            "task_active_seconds": task_active_seconds,
            "monitor_time_ratio": (
                monitor_seconds / task_active_seconds
                if task_active_seconds is not None and task_active_seconds > 0
                else None
            ),
        },
        "phase_monitor_decisions": len(decisions),
        "monitor_interventions": len(interventions),
        "monitor_guidance": len(guidance),
        "overhead_summary": overhead.get("summary", {}),
        "monitor_lifecycle": overhead.get("monitor_lifecycle", {}),
        "intervention_checked_phases": [
            (message.get("extra") or {}).get("intervention_for_phase")
            for message in interventions
        ],
    }


def write_json(path: Path, data: object) -> None:
    """Atomically publish JSON so volume readers never observe a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def redacted_config(config: dict) -> dict:
    """Serialize a runner config without exposing credential environment names."""

    text = json.dumps(config, default=str)
    for token in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
    ):
        text = text.replace(token, f"{token[:5]}...")
    return json.loads(text)

from __future__ import annotations

from typing import Any

def _describe_moatless_action(action: Any) -> str:
    """Render one action's real tool name and its complete argument values.

    `type(action).__name__` alone (e.g. "StringReplaceArgs") throws away the
    file path and old/new text a `StringReplace` actually carries, so a
    reviewer reading the trajectory back cannot tell what was searched for,
    what changed, or whether two edits are really the same one repeated.

    An earlier version of this function used each action's own `to_prompt()`
    instead -- but that is a hand-written, deliberately short human-readable
    summary, not a complete field dump, and is not guaranteed to mention
    everything that affects behavior. Confirmed empirically (Codex review,
    2026-09-06): `SemanticSearchArgs.to_prompt()` never mentions `category`,
    so re-running the same query with a different category renders
    identically to the reviewer; `GrepToolArgs`/`GlobArgs.to_prompt()` drop
    `max_results`; `ReadFileArgs.to_prompt()` silently omits the line range
    when only `start_line` is set (a documented, valid usage). Reading the
    model's own real field values via `model_dump()` instead keeps this
    correct as new action types are added, without trusting each one's prose
    summary to be complete. `thoughts`/`phase` are excluded because they are
    base `ActionArguments` fields already surfaced separately (as the
    trajectory step's own `thought`), not arguments of the action itself.
    """

    model_config = getattr(action, "model_config", None) or {}
    title = model_config.get("title") or type(action).__name__
    try:
        fields = action.model_dump(exclude={"thoughts", "phase"}, exclude_none=True)
    except Exception:
        return title
    args = ", ".join(f"{key}={value!r}" for key, value in fields.items())
    return f"{title}({args})" if args else title


def _moatless_task_context(node) -> str:
    trajectory = node.get_trajectory()
    if trajectory:
        return str(getattr(trajectory[0], "user_message", "") or "")
    return ""


def _executed_action_count(node) -> int:
    return sum(
        action_step.observation is not None
        for trajectory_node in node.get_trajectory()
        for action_step in trajectory_node.action_steps
    )


def _environment_observation(observation: str) -> str:
    """Exclude previously injected advice from future refiner trajectories."""

    marker = "\n\nExternal planning guidance for the next step:\n"
    return observation.split(marker, 1)[0]

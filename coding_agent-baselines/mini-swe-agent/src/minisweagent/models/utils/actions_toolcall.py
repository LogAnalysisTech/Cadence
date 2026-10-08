"""Parse actions & format observations with toolcalls"""

import json
import re
import time

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content


_PHASE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,3}$")
_RESERVED_PHASES = {"analysis", "reasoning", "thought", "thoughts", "working"}
_SUBMIT_TOKENS = ("complete_task_and_submit_final_output",)

# The default `instance_template` (config/default.yaml) spells out an explicit
# "## Recommended Workflow" with 6 numbered steps. When that suggested workflow is
# present, the phase label should be constrained to it instead of left free-form --
# leave this empty to fall back to an open vocabulary.
SUGGESTED_PHASES: tuple[str, ...] = (
    "exploring",
    "reproducing",
    "implementing",
    "verifying",
    "edge case testing",
    "submitting",
)


def _build_phase_description(suggested_phases: tuple[str, ...]) -> str:
    if not suggested_phases:
        return (
            "Required 1-4 word label for the current broad activity. Choose labels freely; "
            "reuse the exact label while the objective is unchanged and introduce a new label "
            "only for a genuine logical transition. Use a small finite vocabulary per task. "
            "Avoid command-, file-, or function-specific "
            "labels and generic labels such as thoughts, analysis, reasoning, or working. "
            "Examples such as exploring, reproducing, localizing fault, implementing, and "
            "verifying are illustrative, not a fixed taxonomy. Use submitting for submission."
        )
    choices = ", ".join(suggested_phases)
    return (
        "Required label for the current broad activity, matching the step of the "
        f"Recommended Workflow you are on. Choose exactly one of: {choices}. Reuse the "
        "same label while still on that step; switch only when you move to the next step."
    )


PHASE_DESCRIPTION = _build_phase_description(SUGGESTED_PHASES)

def build_bash_tool(phase_required: bool = True) -> dict:
    """Bash tool schema. `phase_required=False` drops the `phase` argument entirely,
    for a baseline run with no realtime-monitor-related agent output."""

    properties = {
        "command": {
            "type": "string",
            "description": "The bash command to execute",
        },
    }
    required = ["command"]
    if phase_required:
        properties["phase"] = {
            "type": "string",
            "description": PHASE_DESCRIPTION,
        }
        if SUGGESTED_PHASES:
            properties["phase"]["enum"] = list(SUGGESTED_PHASES)
        required.append("phase")
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


BASH_TOOL = build_bash_tool()


def validate_action_phase(value: object, command: object) -> str:
    """Validate the model-authored activity label and canonicalize submission."""
    command_text = command if isinstance(command, str) else ""
    if any(token in command_text.lower() for token in _SUBMIT_TOKENS):
        return "submitting"
    if not isinstance(value, str) or not _PHASE_RE.fullmatch(value.strip()):
        raise ValueError("'phase' must be a concise 1-4 word activity label.")
    phase = " ".join(value.strip().lower().split())
    if phase in _RESERVED_PHASES:
        raise ValueError(f"'{phase}' is a generic reasoning label, not an activity phase.")
    if SUGGESTED_PHASES and phase not in SUGGESTED_PHASES:
        raise ValueError(f"'phase' must be one of: {', '.join(SUGGESTED_PHASES)}.")
    return phase


def parse_toolcall_actions(
    tool_calls: list,
    *,
    format_error_template: str,
    template_kwargs: dict | None = None,
    phase_required: bool = True,
) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args.

    ``template_kwargs`` are extra variables exposed to ``format_error_template`` (e.g.
    ``{"finish_reason": ...}`` so a template can distinguish a real format mistake from a
    ``max_tokens`` truncation).
    """
    template_kwargs = {"realtime_monitor_phase_required": phase_required, **(template_kwargs or {})}
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                    has_tool_calls=False,
                    **template_kwargs,
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."
        if tool_call.function.name != "bash":
            error_msg += f"Unknown tool '{tool_call.function.name}'."
        if not isinstance(args, dict) or "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if phase_required:
            if not isinstance(args, dict) or "phase" not in args:
                error_msg += "Missing 'phase' argument in bash tool call."
            else:
                try:
                    args["phase"] = validate_action_phase(args["phase"], args.get("command"))
                except ValueError as exc:
                    error_msg += str(exc)
        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(format_error_template, undefined=StrictUndefined).render(
                        actions=[], error=error_msg.strip(), has_tool_calls=True, **template_kwargs
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        action = {"command": args["command"], "tool_call_id": tool_call.id}
        if phase_required:
            action["phase"] = args["phase"]
        actions.append(action)
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into tool result messages."""
    not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"  # human issued commands
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results

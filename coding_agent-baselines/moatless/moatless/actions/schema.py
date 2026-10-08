import importlib
import logging
import os
import pkgutil
import re
from abc import ABC
from typing import Any, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from moatless.artifacts.artifact import ArtifactChange
from moatless.completion.schema import ResponseSchema
from moatless.completion.stats import CompletionInvocation

logger = logging.getLogger(__name__)


_action_args: dict[str, type["ActionArguments"]] = {}


def _realtime_phase_reporting_enabled() -> bool:
    return os.getenv("REALTIME_MONITOR_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

_PHASE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,3}$")
_RESERVED_PHASES = {"analysis", "reasoning", "thought", "thoughts", "working"}

# The default `system_prompt` used by our SWE-bench flow (see
# .moatless/flows/swebench_claude.json, "## Core Decision Flow") spells out an explicit
# 6-step workflow. When that suggested workflow is present, the phase label should be
# constrained to it instead of left free-form -- leave this empty to fall back to an
# open vocabulary.
SUGGESTED_PHASES: tuple[str, ...] = (
    "understanding issue",
    "finding code",
    "reproducing",
    "implementing",
    "verifying",
    "submitting",
)


def _build_phase_description(suggested_phases: tuple[str, ...]) -> str:
    if not suggested_phases:
        return (
            "When realtime monitoring is enabled, provide a required 1-4 word label for the "
            "current broad activity. Choose labels freely; "
            "reuse the exact label while the objective is unchanged and introduce a new label "
            "only for a genuine logical transition. Use a small finite vocabulary per task. "
            "Avoid command-, file-, or function-specific "
            "labels and generic labels such as thoughts, analysis, reasoning, or working. "
            "Examples such as exploring, reproducing, localizing fault, implementing, and "
            "verifying are illustrative, not a fixed taxonomy. Use submitting for submission."
        )
    choices = ", ".join(suggested_phases)
    return (
        "When realtime monitoring is enabled, provide a required label for the current "
        "broad activity, matching the step of the Core Decision Flow you are on. Choose "
        f"exactly one of: {choices}. Reuse the same label while still on that step; switch "
        "only when you move to the next step."
    )


PHASE_DESCRIPTION = _build_phase_description(SUGGESTED_PHASES)


class ActionArguments(ResponseSchema, ABC):
    """Base structured action, extended with an optional monitor phase label."""

    thoughts: Optional[str] = Field(None, description="Your reasoning for the action.")
    phase: Optional[str] = Field(
        None,
        description=PHASE_DESCRIPTION,
        validation_alias=AliasChoices("phase", "current_phase"),
        json_schema_extra=({"enum": list(SUGGESTED_PHASES)} if SUGGESTED_PHASES else None),
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict[str, Any]:
        """Hide monitor-only schema surface from vanilla and LivePlan agents."""

        schema = super().model_json_schema(*args, **kwargs)
        if _realtime_phase_reporting_enabled():
            return schema
        schema.get("properties", {}).pop("phase", None)
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [name for name in required if name != "phase"]
        return schema

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """Exclude the monitor-only field from native trajectory payloads."""

        data = super().model_dump(*args, **kwargs)
        if not _realtime_phase_reporting_enabled():
            data.pop("phase", None)
        return data

    @property
    def current_phase(self) -> Optional[str]:
        """Backward-compatible accessor for trajectories created before `phase`."""
        return self.phase

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, value: Optional[str]) -> Optional[str]:
        """Normalize valid task-level labels while rejecting reasoning labels."""

        if value is None:
            return None
        if not isinstance(value, str) or not _PHASE_RE.fullmatch(value.strip()):
            raise ValueError("'phase' must be a concise 1-4 word activity label.")
        phase = " ".join(value.strip().lower().split())
        if phase in _RESERVED_PHASES:
            raise ValueError(f"'{phase}' is a generic reasoning label, not an activity phase.")
        if SUGGESTED_PHASES and phase not in SUGGESTED_PHASES:
            raise ValueError(f"'phase' must be one of: {', '.join(SUGGESTED_PHASES)}.")
        return phase

    def format_for_llm(self) -> str:
        """Format the action name for LLM consumption"""
        return str(self.name)

    @classmethod
    def format_name_for_llm(cls) -> str:
        """Format the class name for LLM consumption"""
        return str(cls.name)

    @classmethod
    def from_tool_call(cls, tool_args: dict[str, Any], tool_name: str | None = None):
        return cls(**tool_args)

    def equals(self, other: "ActionArguments") -> bool:
        return self.model_dump(exclude={"thoughts", "phase"}) == other.model_dump(exclude={"thoughts", "phase"})

    def to_prompt(self):
        prompt = f"Action: {self.name}\n"
        prompt += "\n".join([f"  {k}: {v}" for k, v in self.model_dump(exclude={"thoughts"}).items()])
        return prompt

    def short_summary(self) -> str:
        return ""

    @model_validator(mode="before")
    @classmethod
    def fix_null_fields(cls, data: Any) -> Any:
        """Allow thoughts to be null."""
        if isinstance(data, dict):
            data = data.copy()
            if str(cls.name).lower() in {"finish", "verifiedfinish", "respond", "reject"} and (
                "phase" in data or "current_phase" in data
            ):
                data["phase"] = "submitting"
                data.pop("current_phase", None)
            for key, value in data.items():
                if value == "null":
                    data[key] = None

        return data

    @classmethod
    def get_action_args(cls, action_name: str) -> type["ActionArguments"]:
        """
        Dynamically import and return the appropriate ActionArguments class for the given action.
        """
        if not _action_args:
            cls._load_action_args()

        action_args = _action_args.get(action_name)
        if action_args:
            return action_args

        raise ValueError(f"Unknown action: {action_name}")

    @classmethod
    def _load_action_args(cls):
        actions_package = importlib.import_module("moatless.actions")

        for _, module_name, _ in pkgutil.iter_modules(actions_package.__path__):
            full_module_name = f"moatless.actions.{module_name}"
            module = importlib.import_module(full_module_name)
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, ActionArguments) and obj != ActionArguments:
                    _action_args[name] = obj

    @classmethod
    def model_validate(cls, obj: Any) -> "ActionArguments":
        if isinstance(obj, dict):
            obj = obj.copy()
            action_args_class_path = obj.pop("action_args_class", None)

            if action_args_class_path:
                module_name, class_name = action_args_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                action_args_class = getattr(module, class_name)
                return action_args_class.model_validate(obj)
        return super().model_validate(obj)


class Observation(BaseModel):
    message: Optional[str] = Field(
        None,
        description="The message returned to the agent, will be displayed in message history.",
    )
    summary: Optional[str] = Field(
        None,
        description="Summary of the observation, will be displayed in summarised message history.",
    )
    terminal: bool = Field(False, description="Indicates if this action results in a terminal state")
    error: Optional[str] = Field(None, description="Error message if the action fails")
    properties: Optional[dict[str, Any]] = Field(default_factory=dict, description="Additional properties")
    execution_completion: Optional[CompletionInvocation] = Field(
        None, description="Completion created when executing the action"
    )
    artifact_changes: list[ArtifactChange] = Field(
        default_factory=list,
        description="Artifact changes created when executing the action",
    )

    @classmethod
    def create(
        cls,
        message: str,
        summary: Optional[str] = None,
        terminal: bool = False,
        properties: Optional[dict[str, Any]] = None,
        execution_completion: Optional[CompletionInvocation] = None,
        artifact_changes: Optional[list[ArtifactChange]] = None,
    ):
        return cls(
            message=message,
            terminal=terminal,
            summary=summary,
            properties=properties,
            execution_completion=execution_completion,
            artifact_changes=artifact_changes or [],
        )

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data["artifact_changes"] = [change.model_dump() for change in self.artifact_changes]
        if self.execution_completion:
            data["execution_completion"] = self.execution_completion.model_dump()
        else:
            data["execution_completion"] = None
        return data

    @classmethod
    def model_validate(cls, obj: Any):
        if isinstance(obj, dict):
            obj = obj.copy()
            obj["artifact_changes"] = [
                ArtifactChange.model_validate(change) for change in obj.get("artifact_changes", [])
            ]
            if obj.get("execution_completion"):
                obj["execution_completion"] = CompletionInvocation.model_validate(obj["execution_completion"])
        return super().model_validate(obj)


class RewardScaleEntry(BaseModel):
    min_value: int
    max_value: int
    description: str


class ActionProperty(BaseModel):
    type: str
    title: str
    description: str
    default: Optional[Any] = None


class ActionSchema(BaseModel):
    title: str
    description: str
    type: str = "object"
    action_class: str
    properties: dict[str, ActionProperty]

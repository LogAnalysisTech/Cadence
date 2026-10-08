"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import os
import time
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, Submitted, TimeExceeded
from minisweagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()
        self._realtime_phase_monitor = None
        self._agent_telemetry = _create_passive_agent_telemetry()
        self._maybe_install_realtime_phase_monitor()

    def _maybe_install_realtime_phase_monitor(self) -> None:
        """Install exactly the explicitly requested monitor condition.

        With both flags disabled this returns before importing monitor code,
        preserving the native mini-SWE-agent prompt, loop, and trajectory.
        """

        realtime_monitor_requested = os.getenv("REALTIME_MONITOR_ENABLED", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        monitor_baseline_requested = os.getenv("MONITOR_BASELINE_ENABLED", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if realtime_monitor_requested:
            raise RuntimeError(
                "The legacy realtime monitor was removed. Use MONITOR_BASELINE_ENABLED=1 "
                "and MONITOR_BASELINE_VARIANT=adaptive_interval for Cadence."
            )
        if not monitor_baseline_requested:
            return

        if monitor_baseline_requested:
            # Baseline installs fail closed: a silently-failed LivePlan install would
            # produce a mislabeled "vanilla" run masquerading as a monitor data point.
            from monitor_runtime.factory import create_mini_swe_agent_integration
            from monitor_runtime.factory import create_monitor

            self.set_realtime_phase_monitor(create_mini_swe_agent_integration(create_monitor()))
            return

    def set_realtime_phase_monitor(self, monitor) -> None:
        """Install or remove the optional native pre-action monitor."""

        self._realtime_phase_monitor = monitor
        self._agent_telemetry = monitor if monitor is not None else _create_passive_agent_telemetry()

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
                "realtime_monitor_phase_required": self._realtime_monitor_active(),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _realtime_monitor_active(self) -> bool:
        return self._realtime_phase_monitor is not None and getattr(
            self._realtime_phase_monitor.monitor, "requires_agent_phase_reporting", True
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self._agent_telemetry.reset_run_state()
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                self.n_consecutive_format_errors += 1
                limit = self.config.max_consecutive_format_errors
                if 0 < limit <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    messages = list(e.messages)
                    if (
                        self._realtime_monitor_active()
                        and 0 < limit
                        and self.n_consecutive_format_errors == limit - 1
                    ):
                        messages.append(
                            self.model.format_message(
                                role="user",
                                content=self._realtime_phase_monitor.format_error_nudge_message(
                                    self.n_consecutive_format_errors, limit
                                ),
                                extra={"message_type": "monitor_format_nudge"},
                            )
                        )
                    self.add_messages(*messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        query_started = time.perf_counter()
        causal_context = self._agent_telemetry.consume_generation_context()
        generation_step_id = f"generation-{self.n_calls}"
        try:
            message = self.model.query(self.messages)
        except Exception as exc:
            error_messages = getattr(exc, "messages", [])
            usage_source = error_messages[0] if error_messages else None
            self._agent_telemetry.record_agent_generation(
                latency_seconds=time.perf_counter() - query_started,
                usage=usage_source,
                step_id=generation_step_id,
                model=_mini_model_name(self.model),
                causal_context=causal_context,
                outcome="failed",
            )
            raise
        event_id = self._agent_telemetry.record_agent_generation(
            latency_seconds=time.perf_counter() - query_started,
            usage=message,
            phase=message.get("extra", {}).get("phase"),
            step_id=generation_step_id,
            model=_mini_model_name(self.model),
            causal_context=causal_context,
        )
        message.setdefault("extra", {})["overhead_event_id"] = event_id
        message["extra"]["overhead_step_id"] = generation_step_id
        message["extra"]["overhead_causal_context"] = causal_context
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = []
        actions = message.get("extra", {}).get("actions", [])
        batch_validator = getattr(self._realtime_phase_monitor, "validate_action_batch", None)
        if callable(batch_validator):
            # LivePlan's state machine assumes one committed action per model turn.
            # Validate before touching the environment so a rejected batch is atomic.
            batch_validator(self, message, actions)
        if not actions:
            self._agent_telemetry.mark_generation_outcome(
                message.get("extra", {}).get("overhead_event_id"),
                "accepted_no_action",
            )
        pending_guidance: list[str | dict[str, object]] = []
        for action_index, action in enumerate(actions):
            if self._realtime_phase_monitor is not None:
                # The action has been proposed but has not reached the environment.
                decision = self._realtime_phase_monitor.inspect_before_action(self, message, action, action_index)
                if not decision.execute_proposed_step:
                    # The adapter removed the proposal and inserted feedback for requery.
                    return []
            action_started = time.perf_counter()
            action_outcome = "failed"
            try:
                result = self.env.execute(action)
                outputs.append(result)
                action_outcome = "completed"
            except Submitted:
                # Submission is a successful terminal action represented by a
                # control-flow exception, not an environment execution failure.
                action_outcome = "submitted"
                raise
            except Exception as error:
                if hasattr(self._realtime_phase_monitor, "handle_action_failure"):
                    self._realtime_phase_monitor.handle_action_failure(
                        self, message, action, error
                    )
                raise
            finally:
                self._agent_telemetry.record_agent_action(
                    latency_seconds=time.perf_counter() - action_started,
                    phase=action.get("phase") or message.get("extra", {}).get("phase"),
                    step_id=message.get("extra", {}).get("overhead_step_id"),
                    causal_context=message.get("extra", {}).get("overhead_causal_context", "normal"),
                    outcome=action_outcome,
                    generation_event_id=message.get("extra", {}).get("overhead_event_id"),
                )
            if hasattr(self._realtime_phase_monitor, "handle_after_action"):
                guidance = self._realtime_phase_monitor.handle_after_action(self, message, action, result)
                if guidance:
                    pending_guidance.append(guidance)
        result_messages = self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )
        for guidance in pending_guidance:
            if isinstance(guidance, dict):
                guidance_content = str(guidance.get("content", ""))
                monitor_event = dict(guidance.get("extra", {}))
            else:
                guidance_content = guidance
                monitor_event = {}
            result_messages += self.add_messages(
                self.model.format_message(
                    role="user",
                    content=guidance_content,
                    extra={
                        "message_type": "monitor_guidance",
                        **monitor_event,
                    },
                )
            )
            if hasattr(self._realtime_phase_monitor, "confirm_advisory_delivery"):
                self._realtime_phase_monitor.confirm_advisory_delivery(
                    message.get("extra", {}).get("overhead_step_id")
                )
        return result_messages

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        agent_data["info"]["overhead"] = self._agent_telemetry.overhead_payload()
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data


def _mini_model_name(model) -> str:
    config = getattr(model, "config", None)
    return str(getattr(config, "model_name", None) or type(model).__name__)


def _create_passive_agent_telemetry():
    # Imported lazily so importing mini-SWE-agent does not initialize monitor LLM clients.
    from monitor_runtime.integration import PassiveAgentTelemetry

    return PassiveAgentTelemetry()

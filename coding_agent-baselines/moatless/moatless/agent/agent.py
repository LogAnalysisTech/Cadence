from datetime import datetime
import logging
import os
import traceback
import time
from collections.abc import Callable, Collection
from typing import Any, Awaitable, Optional, cast

from opentelemetry import trace
from pydantic import Field, PrivateAttr

from moatless.actions.action import Action
from moatless.actions.schema import (
    ActionArguments,
    Observation,
    PHASE_DESCRIPTION,
)
from moatless.agent.events import (
    ActionCreatedEvent,
    ActionExecutedEvent,
    AgentErrorEvent,
    AgentEvent,
    RunAgentEvent,
)
from moatless.completion import BaseCompletionModel
from moatless.completion.schema import ResponseSchema
from moatless.component import MoatlessComponent
from moatless.context_data import current_action_step
from moatless.events import BaseEvent
from moatless.exceptions import (
    CompletionError,
    CompletionRejectError,
    RejectError,
    RuntimeError,
)
from moatless.message_history.base import BaseMemory
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.node import ActionStep, Node, Thoughts
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("moatless.agent")


def _monitor_protocol_violation(
    structured_outputs: list,
    *,
    require_phase: bool = True,
    require_single_action: bool = False,
    terminal_action_names: Collection[str] = (),
) -> str | None:
    """Return a description of the first realtime-monitor protocol violation, or None."""

    if require_single_action and len(structured_outputs) != 1:
        return "Online monitor mode requires exactly one structured action per response."
    if not require_phase:
        return None
    missing_phase = [
        action.name for action in structured_outputs if isinstance(action, ActionArguments) and action.phase is None
    ]
    if missing_phase:
        return f"Missing required phase for action(s): {', '.join(missing_phase)}"
    phases = {
        action.phase
        for action in structured_outputs
        if isinstance(action, ActionArguments) and action.phase is not None
    }
    if len(phases) > 1:
        return (
            "All structured actions in one response must use the same phase so the "
            "realtime monitor can review the action group atomically."
        )
    terminal_positions = [
        index
        for index, action in enumerate(structured_outputs)
        if isinstance(action, ActionArguments)
        and (bool(getattr(action, "is_terminal", False)) or action.name in terminal_action_names)
    ]
    if any(index != len(structured_outputs) - 1 for index in terminal_positions):
        return (
            "A terminal action such as Finish must be the final action in its "
            "response so every accepted action in the group can execute."
        )
    return None


class ActionAgent(MoatlessComponent):
    agent_id: Optional[str] = Field(None, description="Agent ID")
    model_id: Optional[str] = Field(None, description="Model ID")
    description: Optional[str] = Field(None, description="Description of the agent")

    completion_model: Optional[BaseCompletionModel] = Field(
        None, description="Completion model to be used for generating completions"
    )

    system_prompt: str = Field(..., description="System prompt to be used for generating completions")
    actions: list[Action] = Field(default_factory=list)
    memory: BaseMemory = Field(
        default_factory=MessageHistoryGenerator,
        description="Message history generator to be used for generating completions",
    )
    shadow_mode: bool = Field(True, description="Set to true to not persist changes done by the agent.")

    _action_map: dict[type[ActionArguments], Action] = PrivateAttr(default_factory=dict)
    _workspace: Workspace | None = PrivateAttr(default=None)
    _on_event: Optional[Callable[[BaseEvent], Awaitable[None]]] = PrivateAttr(default=None)
    _realtime_phase_monitor: Any | None = PrivateAttr(default=None)
    _realtime_monitor_root: Any | None = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        monitor_requested = os.getenv("REALTIME_MONITOR_ENABLED", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if monitor_requested:
            self.system_prompt = (
                self.system_prompt.rstrip()
                + "\n\n# Phase Reporting\nEvery action must include its required `phase` field. "
                + PHASE_DESCRIPTION
                + " When emitting multiple actions in one response, every action must use "
                "the same phase; start a new response after observing the group before "
                "transitioning to another phase. If the group contains a terminal action "
                "such as Finish, it must be the final action in that response."
            )

        action_args: list[type[ActionArguments]] = []
        for action in self.actions:
            self._action_map[action.args_schema] = action

            if not action.hidden:
                action_args.append(action.args_schema)

        if not action_args:
            raise RuntimeError("No actions found")

        if self.completion_model:
            # Cast action_args to the expected type since ActionArguments inherits from ResponseSchema
            response_schemas = cast(list[type[ResponseSchema]], action_args)
            self.completion_model.initialize(response_schemas, self.system_prompt)
        self._maybe_install_realtime_phase_monitor()

    def _maybe_install_realtime_phase_monitor(self) -> None:
        """Install only an explicitly requested monitor condition."""

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
            from monitor_runtime.factory import create_moatless_integration
            from monitor_runtime.factory import create_monitor

            self.set_realtime_phase_monitor(create_moatless_integration(create_monitor()))
            return

    def set_realtime_phase_monitor(self, monitor: Any | None) -> None:
        """Install or remove the optional native pre-action monitor."""

        self._realtime_phase_monitor = monitor
        self._realtime_monitor_root = None

    def _realtime_monitor_active(self) -> bool:
        return self._realtime_phase_monitor is not None and getattr(
            self._realtime_phase_monitor.monitor, "requires_agent_phase_reporting", True
        )

    def _single_action_monitor_active(self) -> bool:
        return self._realtime_phase_monitor is not None and bool(
            getattr(self._realtime_phase_monitor, "requires_single_action", False)
        )

    async def run_simple(self, user_message: str) -> Observation | None:
        node = Node.create(user_message)
        child_node = node.create_child()
        await self.run(child_node)
        return child_node.observation

    @tracer.start_as_current_span("ActionAgent.run")
    async def run(self, node: Node):
        """Run the agent on a node to generate and execute an action."""
        if not self.completion_model:
            raise RuntimeError("Completion model not set")

        if self._realtime_phase_monitor is not None:
            root = node
            while root.parent is not None:
                root = root.parent
            if self._realtime_monitor_root is not root:
                self._realtime_phase_monitor.reset_run_state()
                self._realtime_monitor_root = root

        if node.is_executed():
            raise RuntimeError("Node already executed")

        try:
            while True:
                generated_actions = False
                if not node.action_steps:
                    await self._generate_actions(node)
                    generated_actions = True
                else:
                    logger.info(f"Node{node.node_id}: Action steps already generated. Skipping action generation.")

                if generated_actions and not node.monitor_events:
                    duplicate_node = node.find_node_with_same_action_steps()
                    if duplicate_node:
                        node.is_duplicate = True
                        logger.info(f"Node{node.node_id} is a duplicate to Node{duplicate_node.node_id}. Skipping")
                        return

                if not node.action_steps:
                    logger.warning(f"Node{node.node_id}: No action steps generated. Skipping execution.")
                    return

                action_names = [action_step.action.name for action_step in node.action_steps]
                logger.info(f"Node{node.node_id}: Execute actions: {action_names}")
                intervention_inserted = False
                group_inspector = getattr(self._realtime_phase_monitor, "inspect_before_action_group", None)
                group_checked = callable(group_inspector)
                if group_checked:
                    # Realtime Moatless treats one model response as an atomic proposal:
                    # review the complete action group before any native action executes.
                    decision = await group_inspector(self, node, list(node.action_steps))
                    if not decision.execute_proposed_step:
                        intervention_inserted = True

                if intervention_inserted:
                    # The adapter removed the whole rejected group. Regenerate this node
                    # from monitor feedback without executing any part of the proposal.
                    continue

                for i, action_step in enumerate(list(node.action_steps)):
                    current_action_step.set(i)  # Used in logging to identify the action step
                    if self._realtime_phase_monitor is not None and not group_checked:
                        # Inspect before `_execute`; a blocked proposal is removed by the adapter.
                        decision = await self._realtime_phase_monitor.inspect_before_action(self, node, action_step)
                        if not decision.execute_proposed_step:
                            current_action_step.set(None)
                            intervention_inserted = True
                            break
                    action_started = time.perf_counter()
                    try:
                        await self._execute(node, action_step)
                    except Exception as error:
                        if hasattr(self._realtime_phase_monitor, "handle_action_failure"):
                            await self._realtime_phase_monitor.handle_action_failure(self, node, action_step, error)
                        raise
                    else:
                        if not group_checked and hasattr(self._realtime_phase_monitor, "handle_after_action"):
                            try:
                                await self._realtime_phase_monitor.handle_after_action(self, node, action_step)
                            except Exception as monitor_error:
                                logger.exception(
                                    "Post-action monitor delivery failed after a successful action",
                                    exc_info=True,
                                )
                                if hasattr(
                                    self._realtime_phase_monitor,
                                    "handle_advisory_delivery_failure",
                                ):
                                    await self._realtime_phase_monitor.handle_advisory_delivery_failure(
                                        self, node, action_step, monitor_error
                                    )
                    finally:
                        if self._realtime_phase_monitor is not None:
                            self._realtime_phase_monitor.record_agent_action(
                                latency_seconds=time.perf_counter() - action_started,
                                phase=getattr(action_step.action, "phase", None),
                                step_id=getattr(node, "overhead_step_id", None),
                                causal_context=getattr(node, "overhead_causal_context", "normal"),
                                usage=getattr(getattr(action_step, "completion", None), "usage", None),
                                model=getattr(getattr(action_step, "completion", None), "model", None),
                                generation_event_id=getattr(node, "overhead_generation_event_id", None),
                            )
                            self._realtime_phase_monitor.persist_overhead(node)
                    current_action_step.set(None)
                    if node.terminal:
                        break

                if group_checked and hasattr(self._realtime_phase_monitor, "handle_after_action_group"):
                    try:
                        await self._realtime_phase_monitor.handle_after_action_group(
                            self, node, list(node.action_steps)
                        )
                    except Exception as monitor_error:
                        logger.exception(
                            "Post-action group monitor delivery failed after successful actions",
                            exc_info=True,
                        )
                        executed_steps = [step for step in node.action_steps if step.observation is not None]
                        if executed_steps and hasattr(
                            self._realtime_phase_monitor,
                            "handle_advisory_delivery_failure",
                        ):
                            await self._realtime_phase_monitor.handle_advisory_delivery_failure(
                                self, node, executed_steps[-1], monitor_error
                            )

                if intervention_inserted:
                    # Regenerate the node from the newly inserted monitor feedback.
                    continue
                break

        except Exception as e:
            logger.exception(f"Node{node.node_id}: Failed to run.")
            node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
            raise e

    async def _generate_actions(self, node: Node):
        node.possible_actions = [action.name for action in self.actions]

        # TODO: This is a hack to get the file context to work with the existing repository, remove when we got rid of the legacy file_context..
        if node.file_context and not node.file_context._repo and self._workspace:
            node.file_context._repo = self._workspace.repository

        try:
            await self._emit_event(RunAgentEvent(agent_id=self.agent_id, node_id=node.node_id))
            messages = await self.memory.generate_messages(node, self._workspace)
            logger.info(f"Node{node.node_id}: Build action with {len(messages)} messages")

            generation_started = time.perf_counter()
            completion_response = await self.completion_model.create_completion(messages=messages)
            measured_generation_seconds = time.perf_counter() - generation_started
            if completion_response.completion_invocation:
                node.completions["build_action"] = completion_response.completion_invocation
                if self._realtime_phase_monitor is not None:
                    invocation = completion_response.completion_invocation
                    node.overhead_causal_context = self._realtime_phase_monitor.consume_generation_context()
                    node.overhead_generation_event_id = self._realtime_phase_monitor.record_agent_generation(
                        latency_seconds=measured_generation_seconds,
                        usage=invocation.usage,
                        model=invocation.model,
                        causal_context=node.overhead_causal_context,
                    )
                    self._realtime_phase_monitor.persist_overhead(node)
            elif self._realtime_phase_monitor is not None:
                node.overhead_causal_context = self._realtime_phase_monitor.consume_generation_context()
                node.overhead_generation_event_id = self._realtime_phase_monitor.record_agent_generation(
                    latency_seconds=measured_generation_seconds,
                    causal_context=node.overhead_causal_context,
                )
                self._realtime_phase_monitor.persist_overhead(node)

            if completion_response.thought:
                node.thoughts = Thoughts(text=completion_response.thought)

            node.assistant_message = completion_response.text_response
            if not completion_response.structured_outputs:
                raise RejectError("No action found")

            if self._realtime_monitor_active() or self._single_action_monitor_active():
                require_phase = self._realtime_monitor_active()
                violation = _monitor_protocol_violation(
                    completion_response.structured_outputs,
                    require_phase=require_phase,
                    require_single_action=self._single_action_monitor_active(),
                    terminal_action_names=(
                        {action.name for action in self.actions if action.is_terminal} if require_phase else ()
                    ),
                )
                if violation is not None:
                    # Unlike the built-in FormatError retry loops in mini-SWE-agent and SWE-agent,
                    # this monitor-only protocol check has no existing retry budget of its own:
                    # give exactly one corrective nudge before giving up on the node.
                    logger.warning(f"Node{node.node_id}: {violation} Retrying once with a corrective nudge.")
                    retry_messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                self._realtime_phase_monitor.format_error_nudge_message(1, 2)
                                if require_phase
                                else (
                                    "LivePlan requires exactly one tool action per response. "
                                    "Choose the single next action, then wait for its observation "
                                    "and any monitor guidance before proposing another action."
                                )
                            ),
                        },
                    ]
                    retry_started = time.perf_counter()
                    completion_response = await self.completion_model.create_completion(messages=retry_messages)
                    measured_generation_seconds = time.perf_counter() - retry_started
                    if completion_response.completion_invocation:
                        node.completions["build_action"] = completion_response.completion_invocation
                        invocation = completion_response.completion_invocation
                        node.overhead_generation_event_id = self._realtime_phase_monitor.record_agent_generation(
                            latency_seconds=measured_generation_seconds,
                            usage=invocation.usage,
                            model=invocation.model,
                            causal_context=node.overhead_causal_context,
                        )
                        self._realtime_phase_monitor.persist_overhead(node)
                    if completion_response.thought:
                        node.thoughts = Thoughts(text=completion_response.thought)
                    node.assistant_message = completion_response.text_response
                    if not completion_response.structured_outputs:
                        raise RejectError("No action found after corrective nudge")
                    violation = _monitor_protocol_violation(
                        completion_response.structured_outputs,
                        require_phase=require_phase,
                        require_single_action=self._single_action_monitor_active(),
                        terminal_action_names=(
                            {action.name for action in self.actions if action.is_terminal} if require_phase else ()
                        ),
                    )
                    if violation is not None:
                        raise CompletionRejectError(violation, completion_response.completion_invocation)

            node.thinking_blocks = completion_response.thinking_blocks

            if completion_response.structured_outputs:
                node.action_steps = [
                    ActionStep(action=action)
                    for action in completion_response.structured_outputs
                    if isinstance(action, ActionArguments)
                ]
                for step in node.action_steps:
                    await self._emit_event(
                        ActionCreatedEvent(
                            agent_id=self.agent_id,
                            node_id=node.node_id,
                            action_name=step.action.name,
                        )
                    )

        except CompletionError as e:
            if e.completion_invocation:
                logger.error(f"Node{node.node_id}: Build action failed with completion error: {e}")
                node.completions["build_action"] = e.completion_invocation
                if self._realtime_phase_monitor is not None:
                    if not getattr(node, "overhead_generation_event_id", None):
                        invocation = e.completion_invocation
                        node.overhead_generation_event_id = self._realtime_phase_monitor.record_agent_generation(
                            latency_seconds=invocation.duration_sec,
                            usage=invocation.usage,
                            model=invocation.model,
                            causal_context=self._realtime_phase_monitor.consume_generation_context(),
                            outcome="failed",
                        )
                    else:
                        self._realtime_phase_monitor.mark_generation_outcome(
                            node.overhead_generation_event_id,
                            "failed",
                        )
                    self._realtime_phase_monitor.persist_overhead(node)
            else:
                logger.exception(f"Node{node.node_id}: Build action failed with error ")

            if isinstance(e, CompletionRejectError):
                node.error = f"Completion validation error: {str(e)}\n\n{traceback.format_exc()}"
                await self._emit_event(
                    AgentErrorEvent(
                        agent_id=self.agent_id,
                        node_id=node.node_id,
                        error=f"Completion validation error: {e.message}",
                    )
                )
                return
            else:
                node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
                await self._emit_event(
                    AgentErrorEvent(
                        agent_id=self.agent_id,
                        node_id=node.node_id,
                        error=f"{e.__class__.__name__}: {str(e)}",
                    )
                )
                raise e
        except Exception as e:
            node.terminal = True
            node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(f"Node{node.node_id}: Build action failed with error: {e}. Type {type(e)}")
            await self._emit_event(
                AgentErrorEvent(
                    agent_id=self.agent_id,
                    node_id=node.node_id,
                    error=f"{e.__class__.__name__}: {str(e)}",
                )
            )

            raise e

    @tracer.start_as_current_span("ActionAgent._execute_action_step")
    async def _execute_action_step(self, node: Node, action_step: ActionStep) -> Observation:
        action = self.action_map.get(type(action_step.action))
        if not action:
            logger.error(
                f"Node{node.node_id}: Action {type(action_step.action)} not found in action map. "
                f"Available actions: {list(self.action_map.keys())}"
            )
            raise RuntimeError(
                f"Action {type(action_step.action)} not found in action map with actions: {list(self.action_map.keys())}"
            )

        observation = await action.execute(action_step.action, file_context=node.file_context)
        if action.is_terminal:
            node.terminal = True
        return observation

    @tracer.start_as_current_span("ActionAgent._execute")
    async def _execute(self, node: Node, action_step: ActionStep):
        try:
            action_step.start_time = datetime.now()
            previous_context = node.file_context.clone() if node.file_context else None
            action_step.observation = await self._execute_action_step(node, action_step)

            if previous_context and node.file_context:
                action_step.observation.artifact_changes = node.file_context.get_artifact_changes(previous_context)

            await self._emit_event(
                ActionExecutedEvent(
                    agent_id=self.agent_id,
                    node_id=node.node_id,
                    action_name=action_step.action.name,
                )
            )

            if not action_step.observation:
                logger.warning(f"Node{node.node_id}: Action {action_step.action.name} returned no observation")
            else:
                if action_step.observation.terminal:
                    node.terminal = True
                if action_step.observation.execution_completion:
                    action_step.completion = action_step.observation.execution_completion

            action_step.end_time = datetime.now()

            logger.info(
                f"Node{node.node_id}: Executed action: {action_step.action.name}. "
                f"Terminal: {action_step.observation.terminal if node.observation else False}. "
            )

            logger.debug(
                f"Node{node.node_id}: Observation: {action_step.observation.message if node.observation else None}"
            )

        except CompletionError as e:
            if e.completion_invocation:
                action_step.completion = e.completion_invocation

            node.terminal = True

            if isinstance(e, CompletionRejectError):
                node.error = f"Completion validation error: {str(e)}\n\n{traceback.format_exc()}"  # TODO: Remove this when we got support in the UI to show action_step.error
                # action_step.observation.error = f"Completion validation error: {e.message}"
                return
            else:
                node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
                # action_step.observation.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"  # TODO: Remove this when we got support in the UI to show action_step.error
                raise e

        except Exception as e:
            logger.exception(f"Node{node.node_id}: Execution of action {action_step.action.name} failed.")
            node.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"  # TODO: Remove this when we got support in the UI to show action_step.error
            # action_step.observation.error = f"{e.__class__.__name__}: {str(e)}\n\n{traceback.format_exc()}"
            node.terminal = True
            raise e

    @classmethod
    def get_component_type(cls) -> str:
        return "agent"

    @classmethod
    def _get_package(cls) -> str:
        return "moatless.agent"

    @classmethod
    def _get_base_class(cls) -> type:
        return ActionAgent

    @property
    def action_map(self):
        return self._action_map

    @property
    def workspace(self) -> Workspace:
        if not self._workspace:
            raise RuntimeError("Workspace not set")
        return self._workspace

    # TODO: Replace this with initialize method
    @workspace.setter
    def workspace(self, workspace: Workspace):
        self._workspace = workspace
        for action in self.actions:
            action.workspace = workspace

    async def initialize(self, workspace: Workspace):
        if not self.completion_model:
            raise RuntimeError("Completion model not set")

        self._workspace = workspace

        try:
            self.workspace.repository.shadow_mode = self.shadow_mode
        except ValueError:
            # Repository not set, skip setting shadow_mode
            pass

        for action in self.actions:
            await action.initialize(workspace)

    async def _emit_event(self, event: AgentEvent):
        """Emit a pure agent event"""
        if self._on_event:
            await self._on_event(event)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        dump = super().model_dump(**kwargs)

        dump["actions"] = []
        for action in self.actions:
            dump["actions"].append(action.model_dump(**kwargs))

        dump["memory"] = self.memory.model_dump(**kwargs)

        if self.completion_model:
            dump["completion_model"] = self.completion_model.model_dump(**kwargs)

        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "ActionAgent":
        if isinstance(obj, dict):
            obj = obj.copy()

            obj["actions"] = [Action.from_dict(action_data) for action_data in obj.get("actions", [])]

            if "memory" in obj:
                obj["memory"] = BaseMemory.from_dict(obj["memory"])
            else:
                obj["memory"] = MessageHistoryGenerator()

            if "completion_model" in obj:
                obj["completion_model"] = BaseCompletionModel.from_dict(obj["completion_model"])

            instance = super().model_validate(obj)

            for action in instance.actions:
                if not isinstance(action, Action):
                    raise ValueError(f"Invalid action type: {type(action)}. Expected Action subclass.")
                if not hasattr(action, "args_schema"):
                    raise ValueError(f"Action {action.__class__.__name__} has no args_schema attribute")
                instance._action_map[action.args_schema] = action

            return instance

        return obj

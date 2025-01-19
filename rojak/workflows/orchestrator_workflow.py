import copy
from dataclasses import dataclass, field
from typing import Literal
from temporalio import workflow
from temporalio.exceptions import ActivityError
from rojak.types import ConversationMessage, ContextVariables
from collections import deque
import asyncio
from rojak.utils import debug_print
from rojak.workflows.agent_workflow import (
    AgentWorkflow,
    AgentWorkflowRunParams,
    ResumeRequest,
    ResumeResponse,
    ToolResponse,
    AgentTypes,
)
from rojak.agents import Agent, Interrupt


@dataclass
class OrchestratorParams:
    type: Literal["stateless", "persistent"]
    """Specify if it is stateless or persistent workflow."""

    context_variables: ContextVariables = field(default_factory=dict)
    """A dictionary of additional context variables, available to functions and Agent instructions."""

    max_turns: int | float = field(default=float("inf"))
    """The maximum number of conversational turns allowed."""

    debug: bool = False
    """If True, enables debug logging"""

    history_size: int = field(default=10)
    """The maximum number of messages retained in the list before older messages are removed."""

    messages: list[ConversationMessage] = field(default_factory=list)
    """List of conversation messages to initialise workflow with."""

    tasks: deque[tuple[str, "TaskParams"]] = field(default_factory=deque)
    """Tasks queue to initialise workflow with."""


@dataclass
class OrchestratorResponse:
    """The response object from containing the updated state."""

    messages: list[ConversationMessage]
    """The list of updated messages."""

    context_variables: ContextVariables
    """The dictionary of the updated context variables."""

    agent: AgentTypes | None = None
    """The last agent to be called."""

    interrupt: Interrupt | None = None
    """The object surfaced to the client when the interupt is triggered."""


@dataclass
class TaskParams:
    messages: list[ConversationMessage]
    """List of message object."""

    agent: AgentTypes
    """The agent to be called."""


@dataclass
class UpdateConfigParams:
    messages: list[ConversationMessage] | None = None
    """A list of message objects."""

    context_variables: ContextVariables | None = None
    """The dictionary of the updated context variables."""

    max_turns: int | float | None = None
    """The maximum number of conversational turns allowed."""

    history_size: int | None = None
    """The maximum number of messages retained in the list before older messages are removed."""

    debug: bool | None = None
    """If True, enables debug logging"""


@dataclass
class GetConfigResponse:
    messages: list[ConversationMessage]
    """A list of message objects."""

    context_variables: ContextVariables
    """The dictionary of the updated context variables."""

    max_turns: int | float
    """The maximum number of conversational turns allowed."""

    history_size: int
    """The maximum number of messages retained in the list before older messages are removed."""

    debug: bool
    """If True, enables debug logging"""


@workflow.defn
class OrchestratorWorkflow:
    @workflow.init
    def __init__(self, params: OrchestratorParams) -> None:
        self.lock = asyncio.Lock()  # Prevent concurrent update handler executions
        self.tasks: deque[tuple[str, TaskParams]] = params.tasks
        self.responses: dict[str, OrchestratorResponse | ResumeRequest] = {}
        self.latest_response: OrchestratorResponse | ResumeRequest | None = None
        self.max_turns = params.max_turns
        self.debug = params.debug
        self.context_variables = params.context_variables
        self.current_agent_workflow: AgentWorkflow | None = None
        self.task_id: str | None = None
        self.history_size = params.history_size
        self.type = params.type
        self.messages: list[ConversationMessage] = params.messages

    @workflow.run
    async def run(self, params: OrchestratorParams) -> OrchestratorResponse:
        while True:
            await workflow.wait_condition(lambda: bool(self.tasks))
            task_id, task = self.tasks.popleft()
            self.task_id = task_id
            self.messages += task.messages
            self.agent = task.agent  # Keep track of the last to be called

            message = self.messages[-1]
            debug_print(
                self.debug, workflow.now(), f"{message.role}: {message.content}"
            )

            active_agent = self.agent
            init_len = len(self.messages)
            past_message_state = copy.deepcopy(self.messages)

            try:
                while len(self.messages) - init_len < self.max_turns and active_agent:
                    active_agent = await self.process(active_agent)

                response = OrchestratorResponse(
                    messages=self.messages,
                    agent=self.agent,
                    context_variables=self.context_variables,
                )

                self.reply(self.task_id, response)

                await workflow.wait_condition(lambda: workflow.all_handlers_finished())

                if self.type == "stateless":
                    return self.responses[self.task_id]
                else:
                    if len(self.messages) > self.history_size:
                        messages = deque(self.messages[-self.history_size :])
                        while messages and messages[0].role == "tool":
                            messages.popleft()
                        self.messages = list(messages)

                    workflow_history_size = workflow.info().get_current_history_size()
                    workflow_history_length = (
                        workflow.info().get_current_history_length()
                    )
                    if (
                        workflow_history_length > 10_000
                        or workflow_history_size > 20_000_000
                    ):
                        debug_print(
                            self.debug,
                            workflow.now(),
                            "Continue as new due to prevent workflow event history from exceeding limit.",
                        )
                        workflow.continue_as_new(
                            args=[
                                OrchestratorParams(
                                    type=params.type,
                                    context_variables=self.context_variables,
                                    max_turns=self.max_turns,
                                    debug=self.debug,
                                    history_size=self.history_size,
                                    messages=self.messages,
                                    tasks=self.tasks,
                                )
                            ]
                        )
            except ActivityError as e:
                # Return messages to previous state and wait for new messages
                workflow.logger.error(f"Agent failed to complete. Error: {e}")
                self.messages = past_message_state
                active_agent = None
                self.pending = False
                continue

    def reply(self, task_id: str, response: OrchestratorResponse | ResumeRequest):
        """Return response back"""
        self.responses[task_id] = response
        self.latest_response = response

    async def process(self, active_agent: Agent) -> Agent | None:
        params = AgentWorkflowRunParams(
            agent=active_agent,
            messages=self.messages,
            context_variables=self.context_variables,
            debug=self.debug,
            orchestrator=self,
            task_id=self.task_id,
        )
        agent_workflow = AgentWorkflow(params)
        self.current_agent_workflow = agent_workflow
        response, updated_messages = await agent_workflow.run()

        self.messages = updated_messages

        if isinstance(response.output, ToolResponse):
            fn_result = response.output.output
            if fn_result.agent is not None:
                debug_print(
                    self.debug,
                    workflow.now(),
                    f"{active_agent.name}: Transferred to '{fn_result.agent.name}'.",
                )
                self.agent = active_agent = fn_result.agent
            if fn_result.context_variables is not None:
                self.context_variables = fn_result.context_variables
        elif isinstance(response.output, str):
            debug_print(
                self.debug,
                workflow.now(),
                f"\n{active_agent.name}: {response.output}",
            )
            active_agent = None

        return active_agent

    def resume(self, params: ResumeResponse):
        """Resumes an interrupted agent workflow for a specific tool ID."""
        if not self.current_agent_workflow:
            raise ValueError("Cannot resume: No active agent workflow available.")

        tool_id = params.tool_id
        if tool_id in self.current_agent_workflow.interrupted:
            self.current_agent_workflow.interrupted.remove(tool_id)
            self.current_agent_workflow.resumed[tool_id] = params
        else:
            raise KeyError(
                f"Cannot resume: Tool ID '{tool_id}' not found in the approval queue."
            )

    @workflow.query
    def get_messages(self) -> list[ConversationMessage]:
        return self.messages

    @workflow.update
    async def add_task(
        self,
        params: tuple[str, TaskParams | ResumeResponse],
    ) -> OrchestratorResponse | ResumeRequest:
        task_id, task = params
        async with self.lock:
            self.task_id = task_id
            if isinstance(task, TaskParams):
                self.tasks.append(params)
            else:
                self.resume(task)
            await workflow.wait_condition(lambda: task_id in self.responses)
            return self.responses[task_id]

    @workflow.query
    def get_result(self, task_id: str) -> OrchestratorResponse:
        return self.responses[task_id]

    @workflow.query
    def get_latest_result(self) -> OrchestratorResponse | ResumeRequest | None:
        return self.latest_response

    @workflow.query
    def get_config(self) -> GetConfigResponse:
        return GetConfigResponse(
            messages=self.messages,
            context_variables=self.context_variables,
            max_turns=self.max_turns,
            history_size=self.history_size,
            debug=self.debug,
        )

    @workflow.signal
    def update_config(self, params: UpdateConfigParams):
        if params.messages is not None:
            self.messages = params.messages
        if params.context_variables is not None:
            self.context_variables = params.context_variables
        if params.max_turns is not None:
            self.max_turns = params.max_turns
        if params.history_size is not None:
            self.history_size = params.history_size
        if params.debug is not None:
            self.debug = params.debug

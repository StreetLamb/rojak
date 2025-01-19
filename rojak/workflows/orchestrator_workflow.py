from dataclasses import dataclass, field
from temporalio import workflow
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
class OrchestratorBaseParams:
    context_variables: ContextVariables = field(default_factory=dict)
    """A dictionary of additional context variables, available to functions and Agent instructions."""

    max_turns: int | float = field(default=float("inf"))
    """The maximum number of conversational turns allowed."""

    debug: bool = False
    """If True, enables debug logging"""


@dataclass
class OrchestratorParams(OrchestratorBaseParams):
    history_size: int = field(default=10)
    """The maximum number of messages retained in the list before older messages are removed."""


@dataclass
class ShortOrchestratorParams(OrchestratorBaseParams): ...


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


class OrchestratorBaseWorkflow:
    def __init__(self, params: OrchestratorBaseParams):
        self.tasks: deque[tuple[str, TaskParams]] = deque()
        self.responses: dict[str, OrchestratorResponse | ResumeRequest] = {}
        self.max_turns = params.max_turns
        self.debug = params.debug
        self.context_variables = params.context_variables
        self.current_agent_workflow: AgentWorkflow | None = None
        self.task_id: str | None = None

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

    @workflow.query
    def get_messages(self) -> list[ConversationMessage]:
        return self.messages

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


@workflow.defn
class ShortOrchestratorWorkflow(OrchestratorBaseWorkflow):
    """Orchestrator for short-running workflows."""

    @workflow.init
    def __init__(self, params: ShortOrchestratorParams) -> None:
        super().__init__(params)
        self.lock = asyncio.Lock()  # Prevent concurrent update handler executions
        self.task_id: str | None = None

    @workflow.run
    async def run(self, _: ShortOrchestratorParams) -> OrchestratorResponse:
        while True:
            await workflow.wait_condition(lambda: bool(self.tasks))
            task_id, task = self.tasks.popleft()
            self.task_id = task_id
            self.messages = task.messages
            self.agent = task.agent  # Keep track of the last to be called
            self.context_variables = task

            message = self.messages[-1]
            debug_print(
                self.debug, workflow.now(), f"{message.role}: {message.content}"
            )

            active_agent = self.agent
            init_len = len(self.messages)

            while len(self.messages) - init_len < self.max_turns and active_agent:
                active_agent = await self.process(active_agent)

            self.responses[self.task_id] = OrchestratorResponse(
                messages=self.messages,
                agent=self.agent,
                context_variables=self.context_variables,
            )

            await workflow.wait_condition(lambda: workflow.all_handlers_finished())

            return self.responses[self.task_id]

    @workflow.update(unfinished_policy=workflow.HandlerUnfinishedPolicy.ABANDON)
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


# @workflow.defn
# class OrchestratorWorkflow(OrchestratorBaseWorkflow):
#     """Orchestrator for long-running workflows."""

#     @workflow.init
#     def __init__(self, params: OrchestratorParams) -> None:
#         super().__init__(params)
#         self.lock = asyncio.Lock()  # Prevent concurrent update handler executions
#         self.queue: deque[tuple[list[ConversationMessage], Agent]] = deque()
#         self.pending: bool = False
#         self.history_size: int = params.history_size
#         # Stores latest response
#         self.result: OrchestratorResponse = OrchestratorResponse([], None, {})

#     @workflow.run
#     async def run(self, params: OrchestratorParams) -> OrchestratorResponse:
#         while True:
#             await workflow.wait_condition(lambda: bool(self.queue))
#             messages, agent = self.queue.popleft()
#             past_message_state = copy.deepcopy(self.messages)
#             self.messages += messages

#             for message in messages:
#                 debug_print(
#                     self.debug, workflow.now(), f"{message.role}: {message.content}"
#                 )

#             active_agent = self.agent = agent
#             init_len = len(self.messages)

#             try:
#                 while len(self.messages) - init_len < self.max_turns and active_agent:
#                     active_agent = await self.process(active_agent)

#                 self.result = OrchestratorResponse(
#                     messages=self.messages,
#                     agent=self.agent,
#                     context_variables=self.context_variables,
#                 )
#                 self.pending = False

#                 # Wait for all handlers to finish before checking if messages exceed limits
#                 await workflow.wait_condition(lambda: workflow.all_handlers_finished())

#                 # Summarise chat and start new workflow if messages exceeds `history_size` limit
#                 if len(self.messages) > self.history_size:
#                     self.messages = self.messages[-self.history_size :]

#                 workflow_history_size = workflow.info().get_current_history_size()
#                 workflow_history_length = workflow.info().get_current_history_length()
#                 if (
#                     workflow_history_length > 10_000
#                     or workflow_history_size > 20_000_000
#                 ):
#                     debug_print(
#                         self.debug,
#                         workflow.now(),
#                         "Continue as new due to prevent workflow event history from exceeding limit.",
#                     )
#                     workflow.continue_as_new(
#                         args=[
#                             OrchestratorParams(
#                                 agent=self.agent,
#                                 history_size=self.history_size,
#                                 max_turns=self.max_turns,
#                                 context_variables=self.context_variables,
#                                 messages=self.messages,
#                                 debug=self.debug,
#                             )
#                         ]
#                     )
#             except (ChildWorkflowError, ActivityError) as e:
#                 # Return messages to previous state and wait for new messages
#                 match e:
#                     case ChildWorkflowError():
#                         workflow.logger.error(
#                             f"Failed to run agent workflow. Error: {e}"
#                         )
#                         self.messages = past_message_state
#                         print("Revert messages to previous state.")
#                     case ActivityError():
#                         workflow.logger.error(
#                             f"Failed to summarise messages. Error: {e}"
#                         )
#                     case _:
#                         workflow.logger.error(f"Unexpected error. Error: {e}")
#                 active_agent = None
#                 self.pending = False
#                 continue

#     @workflow.update(unfinished_policy=workflow.HandlerUnfinishedPolicy.ABANDON)
#     async def send_messages(
#         self,
#         params: SendMessagesParams,
#     ) -> OrchestratorResponse:
#         async with self.lock:
#             self.pending = True
#             self.queue.append((params.messages, params.agent))
#             await workflow.wait_condition(lambda: self.pending is False)
#             return self.result

#     @workflow.query
#     def get_result(self) -> OrchestratorResponse:
#         return self.result

#     @workflow.query
#     def get_config(self) -> GetConfigResponse:
#         return GetConfigResponse(
#             messages=self.messages,
#             context_variables=self.context_variables,
#             max_turns=self.max_turns,
#             history_size=self.history_size,
#             debug=self.debug,
#         )

#     @workflow.signal
#     def update_config(self, params: UpdateConfigParams):
#         if params.messages is not None:
#             self.messages = params.messages
#         if params.context_variables is not None:
#             self.context_variables = params.context_variables
#         if params.max_turns is not None:
#             self.max_turns = params.max_turns
#         if params.history_size is not None:
#             self.history_size = params.history_size
#         if params.debug is not None:
#             self.debug = params.debug

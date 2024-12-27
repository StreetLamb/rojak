import copy
from dataclasses import dataclass, field
from datetime import timedelta
from temporalio import workflow
from temporalio.exceptions import ActivityError, ChildWorkflowError
from rojak.types import (
    ConversationMessage,
    ContextVariables,
)
from collections import deque
import asyncio
from rojak.utils import create_retry_policy, debug_print
from rojak.workflows.agent_workflow import (
    AgentWorkflow,
    AgentWorkflowRunParams,
    ToolResponse,
    AgentTypes,
)
from rojak.agents import (
    Agent,
    AgentCallParams,
    AgentResponse,
)


@dataclass
class OrchestratorBaseParams:
    agent: AgentTypes
    """The initial agent to be called."""

    context_variables: ContextVariables = field(default_factory=dict)
    """A dictionary of additional context variables, available to functions and Agent instructions."""

    max_turns: int | float = field(default=float("inf"))
    """The maximum number of conversational turns allowed."""

    messages: list[ConversationMessage] = field(default_factory=list)
    """A list of message objects."""

    debug: bool = False
    """If True, enables debug logging"""


@dataclass
class OrchestratorParams(OrchestratorBaseParams):
    history_size: int = field(default=10)
    """The maximum number of messages retained in the list before older messages are removed."""


@dataclass
class ShortOrchestratorParams(OrchestratorBaseParams):
    context_variables: ContextVariables = field(default_factory=dict)
    """A dictionary of additional context variables, available to functions and Agent instructions."""


@dataclass
class OrchestratorResponse:
    """The response object from containing the updated state."""

    messages: list[ConversationMessage]
    """The list of updated messages."""

    agent: AgentTypes | None
    """The last agent to be called."""

    context_variables: ContextVariables
    """The dictionary of the updated context variables."""


@dataclass
class SendMessageParams:
    message: ConversationMessage
    """A message object."""

    agent: AgentTypes
    """The agent to be called."""


@dataclass
class UpdateConfigParams:
    context_variables: ContextVariables | None = None
    """The dictionary of the updated context variables."""

    max_turns: int | float | None = None
    """The maximum number of conversational turns allowed."""

    history_size: int | None = None
    """The maximum number of messages retained in the list before older messages are removed."""

    debug: bool | None = None
    """If True, enables debug logging"""


class OrchestratorBaseWorkflow:
    def __init__(self, params: OrchestratorBaseParams):
        self.messages: list[ConversationMessage] = params.messages
        self.max_turns = params.max_turns
        # Keep track of the last to be called
        self.agent = params.agent
        self.debug = params.debug
        self.context_variables = params.context_variables

    async def process(self, active_agent: Agent) -> Agent | None:
        response, updated_messages = await workflow.execute_child_workflow(
            AgentWorkflow.run,
            AgentWorkflowRunParams(
                agent=active_agent,
                messages=self.messages,
                context_variables=self.context_variables,
                debug=self.debug,
            ),
            execution_timeout=timedelta(minutes=1),
        )

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


@workflow.defn
class ShortOrchestratorWorkflow(OrchestratorBaseWorkflow):
    """Orchestrator for short-running workflows."""

    @workflow.init
    def __init__(self, params: ShortOrchestratorParams) -> None:
        super().__init__(params)

    @workflow.run
    async def run(self, params: ShortOrchestratorParams) -> OrchestratorResponse:
        message = self.messages[-1]
        debug_print(self.debug, workflow.now(), f"{message.role}: {message.content}")

        active_agent = self.agent
        init_len = len(self.messages)

        while len(self.messages) - init_len < self.max_turns and active_agent:
            active_agent = await self.process(active_agent)

        return OrchestratorResponse(
            messages=self.messages,
            agent=self.agent,
            context_variables=self.context_variables,
        )


@workflow.defn
class OrchestratorWorkflow(OrchestratorBaseWorkflow):
    """Orchestrator for long-running workflows."""

    @workflow.init
    def __init__(self, params: OrchestratorParams) -> None:
        super().__init__(params)
        self.lock = asyncio.Lock()  # Prevent concurrent update handler executions
        self.queue: deque[tuple[ConversationMessage, Agent]] = deque()
        self.pending: bool = False
        self.history_size: int = params.history_size
        # Stores latest response
        self.result: OrchestratorResponse = OrchestratorResponse([], None, {})

    @workflow.run
    async def run(self, params: OrchestratorParams) -> OrchestratorResponse:
        while True:
            await workflow.wait_condition(lambda: bool(self.queue))
            message, agent = self.queue.popleft()
            past_message_state = copy.deepcopy(self.messages)
            self.messages.append(message)

            debug_print(
                self.debug, workflow.now(), f"{message.role}: {message.content}"
            )
            active_agent = self.agent = agent
            init_len = len(self.messages)

            try:
                while len(self.messages) - init_len < self.max_turns and active_agent:
                    active_agent = await self.process(active_agent)

                self.result = OrchestratorResponse(
                    messages=self.messages,
                    agent=self.agent,
                    context_variables=self.context_variables,
                )
                self.pending = False

                # Summarise chat and start new workflow if messages exceeds `history_size` limit
                if len(self.messages) > self.history_size:
                    await workflow.wait_condition(
                        lambda: workflow.all_handlers_finished()
                    )
                    response = await self.summarise()
                    debug_print(
                        self.debug, workflow.now(), f"Summary: {response.output}"
                    )
                    assert isinstance(response.output, str)
                    debug_print(
                        self.debug,
                        workflow.now(),
                        "Continue as new due to messages exceeding history size.",
                    )
                    workflow.continue_as_new(
                        args=[
                            OrchestratorParams(
                                agent=self.agent,
                                history_size=self.history_size,
                                max_turns=self.max_turns,
                                context_variables=self.context_variables,
                                messages=[
                                    ConversationMessage(
                                        role="assistant",
                                        content=response.output,
                                    )
                                ],
                                debug=self.debug,
                            )
                        ]
                    )
            except (ChildWorkflowError, ActivityError) as e:
                # Return messages to previous state and wait for new messages
                match e:
                    case ChildWorkflowError():
                        workflow.logger.error(
                            f"Failed to run agent workflow. Error: {e}"
                        )
                        self.messages = past_message_state
                        print("Revert messages to previous state.")
                    case ActivityError():
                        workflow.logger.error(
                            f"Failed to summarise messages. Error: {e}"
                        )
                    case _:
                        workflow.logger.error(f"Unexpected error. Error: {e}")
                active_agent = None
                self.pending = False
                continue

    async def summarise(self) -> AgentResponse:
        """Use the last agent used to generate a summary of the conversation."""
        messages_string = "\n\n".join(
            f"{message.role}: {message.content}" for message in self.messages
        )
        summary_prompt = (
            "This is the conversation history between a user and a chatbot:\n\n"
            + f"{messages_string}\n\n Please produce a three sentence summary of "
            + "this conversation."
        )
        response: AgentResponse = await workflow.execute_activity(
            f"{self.agent.type}_call",
            AgentCallParams(
                messages=[ConversationMessage(role="user", content=summary_prompt)],
                model=self.agent.model,
            ),
            start_to_close_timeout=timedelta(
                seconds=self.agent.retry_options.timeout_in_seconds
            ),
            retry_policy=create_retry_policy(self.agent.retry_options.retry_policy),
            result_type=AgentResponse,
        )
        return response

    @workflow.update(unfinished_policy=workflow.HandlerUnfinishedPolicy.ABANDON)
    async def send_message(
        self,
        params: SendMessageParams,
    ) -> OrchestratorResponse:
        async with self.lock:
            self.pending = True
            self.queue.append((params.message, params.agent))
            await workflow.wait_condition(lambda: self.pending is False)
            return self.result

    @workflow.query
    def get_result(self) -> OrchestratorResponse:
        return self.result

    @workflow.query
    def get_config(self) -> dict:
        return {
            "context_variables": self.context_variables,
            "max_turns": self.max_turns,
            "history_size": self.history_size,
            "debug": self.debug,
        }

    @workflow.signal
    def update_config(self, params: UpdateConfigParams):
        if params.context_variables is not None:
            self.context_variables = params.context_variables
        if params.max_turns is not None:
            self.max_turns = params.max_turns
        if params.history_size is not None:
            self.history_size = params.history_size
        if params.debug is not None:
            self.debug = params.debug

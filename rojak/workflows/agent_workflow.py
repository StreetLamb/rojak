from dataclasses import dataclass
from datetime import timedelta
import json
from typing import TYPE_CHECKING, Union
from temporalio import workflow
from temporalio.exceptions import ActivityError
from rojak.retrievers import Retriever
from rojak.types import ContextVariables, ConversationMessage
from rojak.utils import create_retry_policy, debug_print
from rojak.agents import (
    AgentCallParams,
    ExecuteFunctionParams,
    ExecuteInstructionsParams,
    AgentInstructionOptions,
    AgentToolCall,
    AgentResponse,
    AgentExecuteFnResult,
    Agent,
    ResumeResponse,
    ResumeRequest,
)

if TYPE_CHECKING:
    from rojak.workflows.orchestrator_workflow import OrchestratorWorkflow

try:
    from rojak.agents import OpenAIAgent
except ImportError:
    OpenAIAgent = None

try:
    from rojak.agents import AnthropicAgent
except ImportError:
    AnthropicAgent = None

AgentTypes = (
    Union[*(agent for agent in (OpenAIAgent, AnthropicAgent) if agent is not None)]
    | Agent  # typing fallback
)


@dataclass
class AgentWorkflowRunParams:
    agent: AgentTypes
    """The agent to be called."""

    messages: list[ConversationMessage]
    """List of message objects."""

    context_variables: ContextVariables
    """A dictionary of additional context variables, available to functions and Agent instructions."""

    orchestrator: "OrchestratorWorkflow"
    """The parent orchestrator that called this workflow."""

    task_id: str
    """Unique identifier for this request."""

    debug: bool = False
    """If True, enables debug logging."""


@dataclass
class ToolResponse:
    tool_call_id: str
    """Unique identifier for the tool call the response is for."""

    output: AgentExecuteFnResult
    """Result from tool call function."""


@dataclass
class AgentWorkflowResponse:
    output: str | ToolResponse
    """Agent Workflow output."""

    sender: str
    """Indicate which agent the message originated from."""


class ToolRejectedError(Exception):
    """Error raised when a tool call is rejected."""

    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.__cause__ = cause


class AgentWorkflow:
    def __init__(self, params: AgentWorkflowRunParams):
        self.orchestrator = params.orchestrator
        self.task_id = params.task_id
        self.agent = params.agent
        self.retry_policy = create_retry_policy(self.agent.retry_options.retry_policy)
        self.start_to_close_timeout = timedelta(
            seconds=params.agent.retry_options.timeout_in_seconds
        )
        self.debug = params.debug
        self.messages = params.messages
        self.context_variables = params.context_variables

        # Handle interrupts
        self.interrupt_map = {
            interrupt.tool_name: interrupt for interrupt in params.agent.interrupts
        }
        self.interrupted: set[str] = set()  # tool call ids for approval
        self.resumed: dict[
            str, ResumeResponse
        ] = {}  # tool call ids that resumed, pending actions

    async def run(self) -> tuple[AgentWorkflowResponse, list[ConversationMessage]]:
        # process instructions
        instructions = self.agent.instructions
        try:
            if isinstance(instructions, AgentInstructionOptions):
                instructions: str = await workflow.execute_activity(
                    f"{self.agent.type}_execute_instructions",
                    ExecuteInstructionsParams(
                        instructions,
                        self.context_variables,
                    ),
                    result_type=str,
                    start_to_close_timeout=self.start_to_close_timeout,
                    retry_policy=self.retry_policy,
                )
        except ActivityError as e:
            workflow.logger.error(f"Failed to execute instructions: {e}")
            raise

        # augment instructions
        if isinstance(self.agent.retriever, Retriever):
            context_prompt = await self.retrieve_context(self.messages[-1])
            instructions += context_prompt

        # execute call model activity
        response: AgentResponse = await workflow.execute_activity(
            f"{self.agent.type}_call",
            AgentCallParams(
                messages=[
                    ConversationMessage(role="system", content=instructions),
                    *self.messages,
                ],
                model=self.agent.model,
                function_names=self.agent.functions,
                parallel_tool_calls=self.agent.parallel_tool_calls,
                tool_choice=self.agent.tool_choice,
            ),
            result_type=AgentResponse,
            start_to_close_timeout=self.start_to_close_timeout,
            retry_policy=self.retry_policy,
        )

        # dont use isinstance to check as response output type different for different llm providers
        if response.type == "tool":
            tool_calls = response.tool_calls
            self.messages.append(
                ConversationMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=tool_calls,
                    sender=self.agent.name,
                ),
            )
            debug_print(self.debug, workflow.now(), f"{self.agent.name}: {tool_calls}")

            # TODO: Figure out how to handle concurrent tool calls without race conditions in context_variables
            results: list[AgentWorkflowResponse] = []
            for tool_call in tool_calls:
                result = await self.handle_tool_call(tool_call, self.context_variables)
                assert isinstance(result.output, ToolResponse)
                self.context_variables = result.output.output.context_variables
                results.append(result)

            final_result: AgentWorkflowResponse | None = None
            for result in results:
                assert isinstance(result.output, ToolResponse)
                fn_result = result.output.output
                debug_print(
                    self.debug,
                    workflow.now(),
                    f"{self.agent.name}: {fn_result.output}",
                )
                self.messages.append(
                    ConversationMessage(
                        role="tool",
                        content=fn_result.output,
                        sender=self.agent.name,
                        tool_call_id=result.output.tool_call_id,
                    )
                )
                # Send the last tool call response back to orchestrator to be used for next call to agent
                # Check if any tool call response is an agent. If so, send it back to orchestrator for next call to agent
                # If there are multiple tool call returning an agent, the last one will be used.
                if fn_result.agent:
                    final_result = result

            if not final_result:
                final_result = results[-1]

        else:
            assert isinstance(response.content, str)
            self.messages.append(
                ConversationMessage(
                    role="assistant",
                    content=response.content,
                    sender=self.agent.name,
                )
            )
            debug_print(
                self.debug, workflow.now(), f"{self.agent.name}: {response.content}"
            )
            final_result = AgentWorkflowResponse(
                output=response.content, sender=self.agent.name
            )

        return final_result, self.messages

    async def retrieve_context(self, message: ConversationMessage) -> str:
        try:
            retriever_result: str = await workflow.execute_activity(
                f"{self.agent.retriever.type}_retrieve_and_combine_results",
                message.content,
                result_type=str,
                start_to_close_timeout=self.start_to_close_timeout,
                retry_policy=self.retry_policy,
            )
            debug_print(
                self.debug, workflow.now(), f"Retriever context: {retriever_result}"
            )
            context_prompt = f"\nHere is the context to use to answer the user's question:\n{retriever_result}"
            return context_prompt
        except ActivityError as e:
            workflow.logger.error(
                f"Failed to retrieve context from retriever. Context will not be added. Error: {e.cause}"
            )
            return ""

    async def handle_interrupt(self, tool_call: AgentToolCall):
        """
        Handles interrupts at a specified point ("before" or "after").
        """
        if tool_call.function.name in self.interrupt_map:
            self.interrupted.add(tool_call.id)
            interrupt = self.interrupt_map[tool_call.function.name]
            debug_print(
                self.debug,
                workflow.now(),
                f"Interrupt: {interrupt.question}",
            )
            self.orchestrator.reply(
                self.task_id,
                ResumeRequest(
                    tool_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    question=interrupt.question,
                    when=interrupt.when,
                    tool_arguments=tool_call.function.arguments,
                    task_id=self.task_id,
                ),
            )

            await workflow.wait_condition(lambda: tool_call.id in self.resumed)

            debug_print(
                self.debug,
                workflow.now(),
                f"Interrupt: Resuming tool call '{tool_call.function.name}'",
            )

            resume_params = self.resumed[tool_call.id]

            if resume_params.action == "reject":
                raise ToolRejectedError("Tool rejected.") from ValueError(
                    f"Rejected by user. Reason: '{resume_params.content}'"
                )
            del self.resumed[tool_call.id]

    async def handle_tool_call(
        self,
        tool_call: AgentToolCall,
        context_variables: ContextVariables,
    ) -> AgentWorkflowResponse:
        """
        Handles a tool call, checks for interrupts before and after execution,
        and handles approval/rejection.
        """
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        try:
            await self.handle_interrupt(tool_call)

            # Execute function in activity
            debug_print(
                self.debug,
                workflow.now(),
                f"Processing tool call: '{name}' with args: {args}",
            )
            result: AgentExecuteFnResult = await workflow.execute_activity(
                f"{self.agent.type}_execute_function",
                ExecuteFunctionParams(name, args, context_variables),
                result_type=AgentExecuteFnResult,
                start_to_close_timeout=self.start_to_close_timeout,
                retry_policy=self.retry_policy,
            )

            tool_response = ToolResponse(
                tool_call_id=tool_call.id,
                output=result,
            )
            return AgentWorkflowResponse(output=tool_response, sender=self.agent.name)
        except (ActivityError, ToolRejectedError) as e:
            # If error, let the model know by sending error message in tool response
            workflow.logger.error(
                f"Failed to process tool call '{name}'. "
                f"Error will be sent to agent to reassess. Error: {e}"
            )
            result = AgentExecuteFnResult(
                output=str(e.__cause__), context_variables=context_variables
            )
            tool_response = ToolResponse(tool_call_id=tool_call.id, output=result)
            return AgentWorkflowResponse(output=tool_response, sender=self.agent.name)

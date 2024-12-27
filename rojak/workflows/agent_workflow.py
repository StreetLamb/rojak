from dataclasses import dataclass
from datetime import timedelta
import json
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
    OpenAIAgent,
)

AgentTypes = OpenAIAgent


@dataclass
class AgentWorkflowRunParams:
    agent: AgentTypes
    """The agent to be called."""

    messages: list[ConversationMessage]
    """List of message objects."""

    context_variables: ContextVariables
    """A dictionary of additional context variables, available to functions and Agent instructions."""

    debug: bool = False
    """If True, enables debug logging"""


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


@workflow.defn
class AgentWorkflow:
    @workflow.init
    def __init__(self, params: AgentWorkflowRunParams):
        self.agent = params.agent

        self.retry_policy = create_retry_policy(params.agent.retry_options.retry_policy)
        self.start_to_close_timeout = timedelta(
            seconds=params.agent.retry_options.timeout_in_seconds
        )
        self.debug = params.debug

    @workflow.run
    async def run(
        self, params: AgentWorkflowRunParams
    ) -> tuple[AgentWorkflowResponse, list[ConversationMessage]]:
        # process instructions
        context_variables = params.context_variables
        instructions = params.agent.instructions

        try:
            if isinstance(instructions, AgentInstructionOptions):
                instructions: str = await workflow.execute_activity(
                    f"{params.agent.type}_execute_instructions",
                    ExecuteInstructionsParams(
                        instructions,
                        context_variables,
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
            context_prompt = await self.retrieve_context(params.messages[-1])
            instructions += context_prompt

        # execute call model activity
        response: AgentResponse = await workflow.execute_activity(
            f"{params.agent.type}_call",
            AgentCallParams(
                messages=[
                    ConversationMessage(role="system", content=instructions),
                    *params.messages,
                ],
                model=params.agent.model,
                function_names=params.agent.functions,
                parallel_tool_calls=params.agent.parallel_tool_calls,
                tool_choice=params.agent.tool_choice,
            ),
            result_type=AgentResponse,
            start_to_close_timeout=self.start_to_close_timeout,
            retry_policy=self.retry_policy,
        )

        # dont use isinstance to check as response output type different for different llm providers
        if response.type == "tool":
            tool_calls: list[dict] = response.output
            params.messages.append(
                ConversationMessage(
                    role="assistant",
                    tool_calls=tool_calls,
                    sender=params.agent.name,
                ),
            )
            debug_print(
                self.debug, workflow.now(), f"{params.agent.name}: {tool_calls}"
            )

            # TODO: Figure out how to handle concurrent tool calls without race conditions in context_variables
            results = []
            for tool_call in tool_calls:
                result = await self.handle_tool_call(tool_call, context_variables)
                assert isinstance(result.output, ToolResponse)
                context_variables = result.output.output.context_variables
                results.append(result)

            final_result: AgentWorkflowResponse | None = None
            for result in results:
                assert isinstance(result.output, ToolResponse)
                fn_result = result.output.output
                debug_print(
                    self.debug,
                    workflow.now(),
                    f"{params.agent.name}: {fn_result.output}",
                )
                params.messages.append(
                    ConversationMessage(
                        role="tool",
                        content=fn_result.output,
                        sender=params.agent.name,
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
            assert isinstance(response.output, str)
            params.messages.append(
                ConversationMessage(
                    role="assistant",
                    content=response.output,
                    sender=params.agent.name,
                )
            )
            debug_print(
                self.debug, workflow.now(), f"{params.agent.name}: {response.output}"
            )
            final_result = AgentWorkflowResponse(
                output=response.output, sender=params.agent.name
            )

        return final_result, params.messages

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

    async def handle_tool_call(
        self,
        tool_call: AgentToolCall,
        context_variables: ContextVariables,
    ) -> AgentWorkflowResponse:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        try:
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
        except ActivityError as e:
            # If error, let the model know by sending error message in tool response
            workflow.logger.error(
                f"Failed to process tool call '{name}'. "
                f"Error will be sent to agent to reassess. Error: {e}"
            )
            result = AgentExecuteFnResult(output=str(e.cause))
            tool_response = ToolResponse(tool_call_id=tool_call.id, output=result)
            return AgentWorkflowResponse(output=tool_response, sender=self.agent.name)

from dataclasses import dataclass, field
import json
import os
from typing import Literal
from temporalio import activity
from anthropic import Anthropic, AnthropicBedrock, NotGiven
from anthropic.types import (
    Message,
    TextBlock,
    ToolUseBlock,
    MessageParam,
    ToolResultBlockParam,
    TextBlockParam,
)
from rojak.agents.agent import (
    Agent,
    AgentActivities,
    AgentCallParams,
    AgentExecuteFnResult,
    AgentOptions,
    AgentResponse,
    AgentToolCall,
    ExecuteFunctionParams,
    ExecuteInstructionsParams,
    ToolCallFunction,
)
from rojak.types.types import ConversationMessage
from rojak.utils import function_to_json_anthropic, mcp_to_anthropic_tool


@dataclass
class AnthropicAgentOptions(AgentOptions):
    api_key: str | None = None
    client: Anthropic | AnthropicBedrock | None = None
    inference_config: dict[str, any] = field(
        default_factory=lambda: {
            "max_tokens": 1000,
            "temperature": 0.0,
            "top_p": 0.9,
            "stop_sequences": [],
        }
    )


@dataclass
class AnthropicAgent(Agent):
    model: str = "claude-3-5-haiku-20241022"

    type: Literal["anthropic"] = field(default="anthropic")
    """Type of agent. Must be `"anthropic"`"""

    inference_config: dict[str, any] | None = None
    """Inference configuration for Anthropic models"""


class AnthropicAgentActivities(AgentActivities):
    def __init__(self, options: AnthropicAgentOptions = AnthropicAgentOptions()):
        super().__init__(options)

        if options.client:
            self.client = options.client
        elif options.api_key:
            self.client = Anthropic(api_key=options.api_key)
        elif os.environ.get("ANTHROPIC_API_KEY"):
            self.client = Anthropic()
        else:
            raise ValueError("Anthropic API key is required")

        self.inference_config = options.inference_config

    @staticmethod
    def handle_model_response(response: Message) -> AgentResponse:
        """Convert model response to AgentResponse"""
        content = ""
        tool_calls = []
        for block in response.content:
            if isinstance(block, TextBlock):
                content += block.text
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    AgentToolCall(
                        id=block.id,
                        function=ToolCallFunction(
                            name=block.name,
                            arguments=json.dumps(block.input),
                        ),
                    )
                )

        if tool_calls:
            return AgentResponse(type="tool", content=content, tool_calls=tool_calls)
        else:
            return AgentResponse(type="text", tool_calls=tool_calls, content=content)

    @staticmethod
    def convert_messages(
        messages: list[ConversationMessage],
    ) -> tuple[list[MessageParam], str | None]:
        """Convert messages to be Anthropic compatible.

        1. System messages become the returned system string (only one).
        2. User/assistant text messages remain single messages.
        3. Assistant messages that contain tool calls become a single message
        with a list of tool-use blocks.
        4. Consecutive tool-result messages are combined into one user message,
        where each result is a separate 'tool_result' block.
        """
        converted_messages: list[MessageParam] = []
        system_message: str | None = None

        # Temporary storage for consecutive tool-result blocks
        tool_result_buffer: list[ToolResultBlockParam] = []

        def flush_tool_results():
            # If we have accumulated any tool results,
            # append them as a single user message and clear the buffer.
            nonlocal tool_result_buffer
            if tool_result_buffer:
                converted_messages.append(
                    MessageParam(role="user", content=tool_result_buffer)
                )
                tool_result_buffer = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content

            elif msg.tool_calls:
                # Whenever we hit an assistant message with tool calls,
                # first flush any pending tool results (belonging to 'tool' role messages).
                flush_tool_results()

                # Convert each tool call into a tool-use block.
                tool_blocks = []
                for tool_call_dict in msg.tool_calls:
                    tool_call = AgentToolCall(**tool_call_dict)
                    tool_blocks.append(
                        ToolUseBlock(
                            type="tool_use",
                            id=tool_call.id,
                            input=json.loads(tool_call.function.arguments),
                            name=tool_call.function.name,
                        ).model_dump()
                    )

                # Append as a single assistant message containing multiple tool calls.
                converted_messages.append(
                    MessageParam(role="assistant", content=tool_blocks)
                )

            elif msg.role == "tool":
                # We accumulate tool results into a buffer until we reach a non-tool message.
                tool_result_buffer.append(
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=msg.tool_call_id,
                        content=[TextBlockParam(type="text", text=msg.content)],
                    )
                )

            else:
                # For normal user/assistant messages, first flush any pending tool results.
                flush_tool_results()

                # Then add this user/assistant message as-is.
                if msg.role in ["user", "assistant"]:
                    converted_messages.append(
                        MessageParam(role=msg.role, content=msg.content)
                    )

        # If conversation ended with tool results, flush them.
        flush_tool_results()

        return converted_messages, system_message

    @activity.defn(name="anthropic_call")
    async def call(self, params: AgentCallParams) -> AgentResponse:
        # Create list of messages
        messages, system_message = self.convert_messages(params.messages)

        if params.inference_config:
            self.inference_config = {**self.inference_config, **params.inference_config}

        # Create tool call json
        functions = [
            self.function_map[name]
            for name in params.function_names
            if name in self.function_map
        ]
        tools = [function_to_json_anthropic(f) for f in functions]
        for tool in tools:
            fn_params = tool["input_schema"]["properties"]
            fn_params.pop("context_variables", None)
            required_list = tool["input_schema"]["required"]
            if "context_variables" in required_list:
                required_list.remove("context_variables")

        tools += [
            mcp_to_anthropic_tool(tool) for tool in self.mcp_result.tools.values()
        ]

        response: Message = self.client.messages.create(
            model=params.model,
            messages=messages,
            system=system_message or NotGiven(),
            tools=tools or NotGiven(),
            tool_choice=params.tool_choice
            if tools and params.tool_choice
            else NotGiven(),
            max_tokens=self.inference_config["max_tokens"],
            temperature=self.inference_config["temperature"],
            top_p=self.inference_config["top_p"],
            stop_sequences=self.inference_config["stop_sequences"],
        )

        return self.handle_model_response(response)

    @activity.defn(name="anthropic_execute_instructions")
    async def execute_instructions(self, params: ExecuteInstructionsParams) -> str:
        return await super().execute_instructions(params)

    @activity.defn(name="anthropic_execute_function")
    async def execute_function(
        self, params: ExecuteFunctionParams
    ) -> str | AnthropicAgent | AgentExecuteFnResult:
        return await super().execute_function(params)

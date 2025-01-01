from dataclasses import dataclass, field
import json
import os
from typing import Literal
from temporalio import activity
from anthropic import Anthropic, NotGiven
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
from rojak.utils.helpers import function_to_json_anthropic


@dataclass
class AnthropicAgentOptions(AgentOptions):
    api_key: str | None = None
    client: Anthropic | None = None
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
    def __init__(self, options: AnthropicAgentOptions):
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

        Args:
            messages (list[ConversationMessage]): List of messages to convert

        Returns:
            tuple[list[MessageParam], str | None ]: List of converted messages and the system message if any
        """
        converted_messages: list[MessageParam] = []
        system_message: str | None = None
        for message in messages:
            # tool call
            if message.tool_calls:
                tool_blocks = []
                for tool_call in message.tool_calls:
                    tool_call = AgentToolCall(**tool_call)
                    tool_blocks.append(
                        ToolUseBlock(
                            type="tool_use",
                            id=tool_call.id,
                            input=json.loads(tool_call.function.arguments),
                            name=tool_call.function.name,
                        ).model_dump()
                    )
                converted_messages.append(
                    MessageParam(role="assistant", content=tool_blocks)
                )
            elif message.tool_call_id:
                content = [
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=message.tool_call_id,
                        content=[TextBlockParam(type="text", text=message.content)],
                    )
                ]
                converted_messages.append(MessageParam(role="user", content=content))
            elif message.role in ["user", "assistant"]:
                converted_messages.append(
                    MessageParam(role=message.role, content=message.content)
                )
            elif message.role == "system":
                system_message = message.content
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

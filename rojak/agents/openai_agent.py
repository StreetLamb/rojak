from dataclasses import dataclass, field
from typing import Any, Literal
from openai import NotGiven, OpenAI
from temporalio import activity
from rojak.utils import function_to_json
from rojak.agents import (
    Agent,
    AgentActivities,
    AgentCallParams,
    AgentExecuteFnResult,
    AgentResponse,
    AgentOptions,
    AgentToolCall,
    ExecuteFunctionParams,
    ExecuteInstructionsParams,
)
from openai.types.chat import ChatCompletion
import os


@dataclass
class OpenAIAgentOptions(AgentOptions):
    api_key: str | None = None
    client: OpenAI | None = None
    inference_config: dict[str, Any] = field(
        default_factory=lambda: {
            "max_tokens": 1000,
            "temperature": 0.0,
            "top_p": 0.9,
            "stop_sequences": [],
        }
    )


@dataclass
class OpenAIAgent(Agent):
    model: str = "gpt-4o-mini"

    type: Literal["openai"] = field(default="openai")
    """Type of agent. Must be `"openai"`."""

    inference_config: dict[str, Any] | None = None
    """Inference configuration for OpenAI models"""


class OpenAIAgentActivities(AgentActivities):
    def __init__(self, options: OpenAIAgentOptions):
        super().__init__(options)

        if options.client:
            self.client = options.client
        elif options.api_key:
            self.client = OpenAI(api_key=options.api_key)
        elif os.environ.get("OPENAI_API_KEY"):
            self.client = OpenAI()
        else:
            raise ValueError("OpenAI API key is required")

        self.inference_config = options.inference_config

    @staticmethod
    def handle_model_response(response: ChatCompletion) -> AgentResponse:
        """Convert model response to AgentResponse"""
        message = response.choices[0].message
        if message.content:
            return AgentResponse(output=message.content, type="text")
        elif message.tool_calls:
            tool_calls = [
                AgentToolCall(**dict(tool_call)) for tool_call in message.tool_calls
            ]
            return AgentResponse(
                output=tool_calls,
                type="tool",
            )
        else:
            raise ValueError("Unknown message type")

    @activity.defn(name="openai_call")
    async def call(self, params: AgentCallParams) -> AgentResponse:
        # Create list of messages
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": msg.tool_calls,
                "tool_call_id": msg.tool_call_id,
            }
            for msg in params.messages
        ]

        # Update inference config if needed
        if params.inference_config:
            self.inference_config = {**self.inference_config, **params.inference_config}

        # Create tool call json
        functions = [
            self.function_map[name]
            for name in params.function_names
            if name in self.function_map
        ]
        tools = [function_to_json(f) for f in functions]
        for tool in tools:
            fn_params = tool["function"]["parameters"]
            fn_params["properties"].pop("context_variables", None)
            if "context_variables" in fn_params["required"]:
                fn_params["required"].remove("context_variables")

        response = self.client.chat.completions.create(
            model=params.model,
            messages=messages,
            tools=tools or None,
            tool_choice=params.tool_choice,
            parallel_tool_calls=params.parallel_tool_calls if tools else NotGiven(),
            max_tokens=self.inference_config["max_tokens"],
            temperature=self.inference_config["temperature"],
            top_p=self.inference_config["top_p"],
            stop=self.inference_config["stop_sequences"],
        )

        return self.handle_model_response(response)

    @activity.defn(name="openai_execute_instructions")
    async def execute_instructions(self, params: ExecuteInstructionsParams) -> str:
        return await super().execute_instructions(params)

    @activity.defn(name="openai_execute_function")
    async def execute_function(
        self, params: ExecuteFunctionParams
    ) -> str | OpenAIAgent | AgentExecuteFnResult:
        return await super().execute_function(params)

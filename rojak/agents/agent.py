from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
from rojak.mcp import MCPClient
from mcp.types import Tool, TextContent
from rojak.retrievers import Retriever
from rojak.types import (
    ContextVariables,
    ConversationMessage,
    RetryOptions,
)
from temporalio.exceptions import ApplicationError

from rojak.types.types import MCPServerConfig

AgentFunction = Callable[[], str]


@dataclass
class AgentOptions:
    all_functions: list[AgentFunction] = field(default_factory=list)
    """List of functions that an agent can execute."""

    mcp_servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    """List of MCP servers to connect to."""


@dataclass
class AgentCallParams:
    """Parameters for generating response from model."""

    messages: list[ConversationMessage]
    """List of message objects."""

    model: str
    """The LLM model to use."""

    function_names: list[str] = field(default_factory=list)
    """List of tool call function names that the agent can select."""

    inference_config: dict[str, Any] | None = None
    """Additional configurations for the inference."""

    parallel_tool_calls: bool = True
    """Whether model should perform multiple tool calls together."""

    tool_choice: Any | None = None
    """The tool choice for the agent, if any."""


@dataclass
class ExecuteFunctionParams:
    """Parameters for executing tool call function."""

    name: str
    """The name of the tool call function."""

    args: dict[str, Any]
    """The arguments for the tool call function."""

    context_variables: ContextVariables
    """A dictionary of additional context variables, available to functions and Agent instructions."""


@dataclass
class ExecuteInstructionsParams:
    instructions: "AgentInstructionOptions"
    """Options for the callable instructions."""

    context_variables: ContextVariables
    """A dictionary of additional context variables, available to functions and Agent instructions."""


@dataclass
class ToolCallFunction:
    arguments: str
    """String representation of the arguments for the tool call function."""

    name: str
    """The name of the tool call function."""


@dataclass
class AgentToolCall:
    id: str
    """Unique identifier of the tool call."""

    function: ToolCallFunction
    """Function that the model called."""

    type: str = "function"
    """The type of the tool."""

    def __post_init__(self):
        if isinstance(self.function, dict):
            self.function = ToolCallFunction(**self.function)


@dataclass
class AgentResponse:
    """Response object from generating response from model."""

    type: Literal["text", "tool"]
    """Specify if it is a natural language or a tool call response."""

    content: str | None = None
    """String output from the model"""

    tool_calls: list[AgentToolCall] | None = None
    """List of tool call objects."""

    def __post_init__(self):
        if self.tool_calls:
            self.tool_calls = [
                tool_call
                if isinstance(tool_call, AgentToolCall)
                else AgentToolCall(**tool_call)
                for tool_call in self.tool_calls
            ]


@dataclass
class AgentInstructionOptions:
    """Information of the callable instructions."""

    type: Literal["function"]
    """The type of the instruction. Only `function` is supported."""

    name: str
    """The name of the function."""


@dataclass
class Agent(ABC):
    model: str
    """The LLM model to use."""

    type: str
    """The prefix of the activity name."""

    name: str = "Agent"
    """The name of the agent."""

    instructions: str | AgentInstructionOptions = "You are a helpful assistant."
    """Instructions for the agent, can be a string or a callable returning a string."""

    functions: list[str] = field(default_factory=list)
    """A list of functions that the agent can call."""

    tool_choice: Any | None = None
    """The tool choice for the agent, if any."""

    parallel_tool_calls: bool = True
    """Whether model should perform multiple tool calls together."""

    retriever: Retriever | None = None
    """Specify which retriever to use."""

    retry_options: RetryOptions = field(default_factory=RetryOptions)
    """Options for timeout and retries."""


@dataclass
class AgentExecuteFnResult:
    """Result object from executing tool call function."""

    output: str = ""
    """String output to pass as message content."""

    agent: Agent | None = None
    """The agent to call next."""

    context_variables: ContextVariables = field(default_factory=dict)
    """A dictionary of additional context variables, available to functions and Agent instructions."""


class AgentActivities(ABC):
    """
    Abstract base class for Agent implementations.
    This class provides a common structure for different types of agents.
    """

    def __init__(self, options: AgentOptions):
        self.function_map = {f.__name__: f for f in options.all_functions}

        # Handle MCP servers
        self.connections: dict[str, tuple[MCPClient | list[Tool]]] = {}
        self.mcp_clients: dict[str, MCPClient] = {}
        self.mcp_tools: dict[str, Tool] = {}
        self.tool_client_mapping: dict[str, str] = {}

    @classmethod
    async def create(cls, options: AgentOptions) -> "AgentActivities":
        """Asynchronous factory method to initialize the agent."""
        self = cls(options)
        for server_name, config in options.mcp_servers.items():
            try:
                mcp_client = MCPClient()
                await mcp_client.connect_to_server(config)
                list_tools_result = await mcp_client.session.list_tools()
                self.mcp_clients[server_name] = mcp_client
                for tool in list_tools_result.tools:
                    self.mcp_tools[tool.name] = {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    self.tool_client_mapping[tool.name] = server_name
            except Exception as e:
                print(f"Unable to connect to MCP server. Skipping. Error: {e}")
        return self

    @abstractmethod
    async def call(self, params: AgentCallParams) -> AgentResponse:
        """Generate response from the LLM model.

        Args:
            params (AgentCallParams): Parameters for response generation.

        Returns:
            AgentResponse: Generated response from the model.
        """
        pass

    @abstractmethod
    async def execute_instructions(self, params: ExecuteInstructionsParams) -> str:
        """Execute the instruction callable.

        Args:
            params (ExecuteInstructionsParams): Parameters containing information for executing callable.

        Raises:
            ApplicationError: Error occurred while executing instructions.

        Returns:
            str: Instructions as a string.
        """
        instructions = params.instructions
        if instructions.name not in self.function_map:
            raise ApplicationError(
                f"Function {instructions.name} not found",
                type="FunctionNotFound",
                non_retryable=True,
            )

        fn = self.function_map[instructions.name]
        args = {}

        if "context_variables" in fn.__code__.co_varnames:
            args["context_variables"] = params.context_variables

        res = fn(**args)
        return str(res)

    def handle_function_result(
        self,
        result: str | Agent | AgentExecuteFnResult,
        context_variables: ContextVariables,
    ) -> AgentExecuteFnResult:
        match result:
            case str():
                return AgentExecuteFnResult(
                    output=result,
                    context_variables=context_variables,
                )
            case Agent():
                return AgentExecuteFnResult(
                    output=f"Transferred to '{result.name}'",
                    agent=result,
                    context_variables=context_variables,
                )
            case AgentExecuteFnResult():
                return result
            case _:
                try:
                    return AgentExecuteFnResult(
                        output=str(result), context_variables=context_variables
                    )
                except Exception as e:
                    raise TypeError(
                        f"Unknown function result type: {type(result)}. Error: {str(e)}"
                    )

    @abstractmethod
    async def execute_function(
        self, params: ExecuteFunctionParams
    ) -> str | Agent | AgentExecuteFnResult:
        """Execute the tool call function

        Args:
            params (ExecuteFunctionParams): Parameters for executing tool call function.

        Raises:
            ApplicationError: Error executing tool call function.

        Returns:
            str | Agent | AgentExecuteFnResult: Response from the tool call function.
        """
        if params.name in self.function_map:
            fn = self.function_map[params.name]

            if "context_variables" in fn.__code__.co_varnames:
                params.args["context_variables"] = params.context_variables

            result = fn(**params.args)
        elif params.name in self.mcp_tools:
            server_name = self.tool_client_mapping[params.name]
            client = self.mcp_clients[server_name]
            response = await client.session.call_tool(params.name, params.args)
            result = ""
            for content in response.content:
                if isinstance(content, TextContent):
                    result += f"{content}\n"
        else:
            raise ApplicationError(
                f"Function {params.name} not found",
                type="FunctionNotFound",
                non_retryable=True,
            )

        return self.handle_function_result(result, params.context_variables)

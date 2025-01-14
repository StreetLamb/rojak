from dataclasses import dataclass
from typing import Any, Literal, Sequence


@dataclass
class ConversationMessage:
    role: Literal["user", "assistant", "tool", "system"]
    """The role of the messages author."""

    content: str | None = None
    """The contents of the assistant message. Required unless tool_calls is specified."""

    tool_calls: list[Any] | None = None
    """The tool calls generated by the model, such as function calls."""

    tool_call_id: str | None = None
    """Unique identifier of the tool call."""

    sender: str | None = None
    """Indicate which agent the message originated from."""


ContextVariables = dict[str, Any]


@dataclass
class RetryPolicy:
    """Options for retrying agent activities."""

    initial_interval_in_seconds: int = 1
    """Backoff interval for the first retry. Default 1s."""

    backoff_coefficient: float = 2.0
    """Coefficient to multiply previous backoff interval by to get new
    interval. Default 2.0.
    """

    maximum_interval_in_seconds: int | None = None
    """Maximum backoff interval between retries. Default 100x
    :py:attr:`initial_interval`.
    """

    maximum_attempts: int = 0
    """Maximum number of attempts.
    
    If 0, the default, there is no maximum.
    """

    non_retryable_error_types: Sequence[str] | None = None
    """List of error types that are not retryable."""

    def __post_init__(self):
        # Validation taken from Go SDK's test suite
        if self.maximum_attempts == 1:
            # Ignore other validation if disabling retries
            return
        if self.initial_interval_in_seconds < 0:
            raise ValueError("Initial interval cannot be negative")
        if self.backoff_coefficient < 1:
            raise ValueError("Backoff coefficient cannot be less than 1")
        if self.maximum_interval_in_seconds:
            if self.maximum_interval_in_seconds < 0:
                raise ValueError("Maximum interval cannot be negative")
            if self.maximum_interval_in_seconds < self.initial_interval_in_seconds:
                raise ValueError(
                    "Maximum interval cannot be less than initial interval"
                )
        if self.maximum_attempts < 0:
            raise ValueError("Maximum attempts cannot be negative")


@dataclass
class RetryOptions:
    timeout_in_seconds: int = 60
    """Maximum time allowed for an agent to complete its tasks."""

    retry_policy: RetryPolicy | None = None
    """Options for retrying agent activities."""

    def __post_init__(self):
        if self.timeout_in_seconds < 1:
            raise ValueError("Timeout cannot be less than one second")


@dataclass
class MCPServerConfig:
    """Configuration options for the MCP server"""

    type: Literal["sse", "stdio"]
    """Connection type to MCP server."""

    command: str | None = None
    """(For `stdio` type) The command or executable to run to start the MCP server."""

    args: list[str] | None = None
    """(For `stdio` type) Command line arguments to pass to the `command`."""

    url: str | None = None
    """(For `websocket` or `sse` type) The URL to connect to the MCP server."""

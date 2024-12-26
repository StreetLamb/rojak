from .agent import (
    Agent,
    AgentActivities,
    AgentCallParams,
    AgentExecuteFnResult,
    AgentInstructionOptions,
    AgentResponse,
    AgentOptions,
    AgentToolCall,
    ExecuteFunctionParams,
    ExecuteInstructionsParams,
)

try:
    from .openai_agent import OpenAIAgent, OpenAIAgentOptions, OpenAIAgentActivities  # noqa: F401

    _OPENAI_AVAILABLE_ = True
except ImportError:
    _OPENAI_AVAILABLE_ = False

__all__ = [
    "Agent",
    "AgentActivities",
    "AgentOptions",
    "AgentCallParams",
    "AgentExecuteFnResult",
    "AgentInstructionOptions",
    "AgentResponse",
    "AgentToolCall",
    "ExecuteFunctionParams",
    "ExecuteInstructionsParams",
]

if _OPENAI_AVAILABLE_:
    __all__.extend(
        [
            "OpenAIAgent",
            "OpenAIAgentOptions",
            "OpenAIAgentActivities",
        ]
    )

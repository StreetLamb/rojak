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
    Interrupt,
    ResumeRequest,
    ResumeResponse,
)

try:
    from .openai_agent import OpenAIAgent, OpenAIAgentOptions, OpenAIAgentActivities  # noqa: F401

    _OPENAI_AVAILABLE_ = True
except ImportError:
    _OPENAI_AVAILABLE_ = False

try:
    from .anthropic_agent import (  # noqa: F401
        AnthropicAgent,
        AnthropicAgentOptions,
        AnthropicAgentActivities,
    )

    _ANTHROPIC_AVAILABLE_ = True
except ImportError:
    _ANTHROPIC_AVAILABLE_ = False

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
    "Interrupt",
    "ResumeRequest",
    "ResumeResponse",
]

if _OPENAI_AVAILABLE_:
    __all__.extend(
        [
            "OpenAIAgent",
            "OpenAIAgentOptions",
            "OpenAIAgentActivities",
        ]
    )

if _ANTHROPIC_AVAILABLE_:
    __all__.extend(
        [
            "AnthropicAgent",
            "AnthropicAgentOptions",
            "AnthropicAgentActivities",
        ]
    )

from unittest.mock import MagicMock
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage


def create_mock_response(message, function_calls=[], model="gpt-4o"):
    role = message.get("role", "assistant")
    content = message.get("content", "")

    tool_calls = [
        ToolUseBlock(
            type="tool_use",
            id="mock_tc_id",
            name=call.get("name", ""),
            input=call.get("args", {}),
        ).model_dump()
        for call in function_calls
    ]

    return Message(
        id="mock_cc_id",
        role=role,
        model=model,
        content=[TextBlock(text=content, type="text"), *tool_calls],
        type="message",
        usage=Usage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            input_tokens=0,
            output_tokens=0,
        ),
    )


class MockAnthropicClient:
    def __init__(self):
        self.messages = MagicMock()

    def set_response(self, response: Message):
        """
        Set the mock to return a specific response.
        :param response: A ChatCompletion response to return.
        """
        self.messages.create.return_value = response

    def set_sequential_responses(self, responses: list[Message]):
        """
        Set the mock to return different responses sequentially.
        :param responses: A list of ChatCompletion responses to return in order.
        """
        self.messages.create.side_effect = responses

    def assert_create_called_with(self, **kwargs):
        self.messages.create.assert_called_with(**kwargs)


# Initialize the mock client
client = MockAnthropicClient()

# Set a sequence of mock responses
client.set_sequential_responses(
    [
        create_mock_response(
            {"role": "assistant", "content": "First response"},
            [
                {
                    "name": "process_refund",
                    "args": {"item_id": "item_123", "reason": "too expensive"},
                }
            ],
        ),
        create_mock_response({"role": "assistant", "content": "Second"}),
    ]
)

# This should return the first mock response
first_response = client.messages.create()
print(first_response)  # Outputs: role='agent' content='First response'

# This should return the second mock response
second_response = client.messages.create()
print(second_response)  # Outputs: role='agent' content='Second response'

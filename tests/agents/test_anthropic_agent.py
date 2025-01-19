from unittest.mock import Mock
import uuid
import pytest
from temporalio.testing import WorkflowEnvironment

from rojak import Rojak
from rojak.agents import (
    AgentExecuteFnResult,
    AgentInstructionOptions,
    AnthropicAgentActivities,
    AnthropicAgentOptions,
    AnthropicAgent,
    Interrupt,
    ResumeRequest,
    ResumeResponse,
)
from rojak.client import RunResponse
from rojak.types import (
    ConversationMessage,
    RetryOptions,
    RetryPolicy,
)
from rojak.workflows import OrchestratorResponse, TaskParams

from tests.mock_anthropic_client import (
    MockAnthropicClient,
    create_mock_response,
)

DEFAULT_RESPONSE_CONTENT = "sample response content"
DEFAULT_RESPONSE_CONTENT_2 = "sample response content 2"


@pytest.fixture
def mock_anthropic_client():
    m = MockAnthropicClient()
    m.set_response(
        create_mock_response({"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT})
    )
    return m


@pytest.mark.asyncio
async def test_run_with_messages(mock_anthropic_client: MockAnthropicClient):
    """
    Test a single run call with user messages and verify assistant reply.
    """
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(client=mock_anthropic_client)
        )
        worker = await rojak.create_worker([anthropic_activities])

        async with worker:
            agent = AnthropicAgent(name="assistant")
            task = TaskParams(
                agent=agent,
                messages=[{"role": "user", "content": "Hello how are you?"}],
            )
            run_response: RunResponse = await rojak.run(
                id=str(uuid.uuid4()),
                type="stateless",
                task=task,
            )
            # Verify result is an OrchestratorResponse
            assert isinstance(run_response.result, OrchestratorResponse)
            response: OrchestratorResponse = run_response.result
            assert response.messages[-1].role == "assistant"
            assert response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_get_result(mock_anthropic_client: MockAnthropicClient):
    """
    Demonstrates calling run again with no new task to retrieve the final state.
    """
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(client=mock_anthropic_client)
        )
        worker = await rojak.create_worker([anthropic_activities])

        async with worker:
            agent = AnthropicAgent(name="assistant")
            workflow_id = str(uuid.uuid4())

            # First call with a user message
            task = TaskParams(
                agent=agent,
                messages=[{"role": "user", "content": "Hello how are you?"}],
            )
            run_response = await rojak.run(
                id=workflow_id,
                type="stateless",
                task=task,
            )
            assert isinstance(run_response.result, OrchestratorResponse)

            response = await rojak.get_result(
                id=run_response.id, task_id=run_response.task_id
            )

            assert isinstance(response, OrchestratorResponse)
            first_response: OrchestratorResponse = response
            assert first_response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_callable_instructions(mock_anthropic_client: MockAnthropicClient):
    """
    Test agent instructions invoked via a Python callable.
    """
    task_queue_name = str(uuid.uuid4())
    instruct_fn_mock = Mock()

    def instruct_fn(context_variables):
        res = f"My name is {context_variables.get('name')}"
        instruct_fn_mock(context_variables)
        instruct_fn_mock.return_value = res
        return res

    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(
                client=mock_anthropic_client, all_functions=[instruct_fn]
            )
        )
        worker = await rojak.create_worker([anthropic_activities])

        async with worker:
            agent = AnthropicAgent(
                name="assistant",
                instructions=AgentInstructionOptions(
                    type="function", name="instruct_fn"
                ),
            )
            context_variables = {"name": "John"}
            task = TaskParams(
                agent=agent,
                messages=[{"role": "user", "content": "Hello how are you?"}],
            )
            run_response = await rojak.run(
                id=str(uuid.uuid4()),
                type="stateless",
                task=task,
                context_variables=context_variables,
            )
            assert isinstance(run_response.result, OrchestratorResponse)

            instruct_fn_mock.assert_called_once_with(context_variables)
            assert instruct_fn_mock.return_value == "My name is John"


@pytest.mark.asyncio
async def test_failed_tool_call(mock_anthropic_client: MockAnthropicClient):
    """
    Context variable should be updated by first tool call only since 2nd tool call fails.
    """
    task_queue_name = str(uuid.uuid4())
    get_weather_mock = Mock()
    get_air_quality_mock = Mock()

    def get_weather(context_variables: dict):
        get_weather_mock()
        context_variables["seen"].append("get_weather")
        raise Exception("Something went wrong!")

    def get_air_quality(context_variables: dict):
        get_air_quality_mock()
        context_variables["seen"].append("get_air_quality")
        return AgentExecuteFnResult(
            output="Air quality is great!", context_variables=context_variables
        )

    # set mock to return a response that triggers function calls
    mock_anthropic_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "get_air_quality", "args": {}}],
            ),
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "get_weather", "args": {}}],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
        ]
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        agent = AnthropicAgent(
            name="Test Agent",
            functions=["get_weather", "get_air_quality"],
            retry_options=RetryOptions(retry_policy=RetryPolicy(maximum_attempts=5)),
        )
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(
                client=mock_anthropic_client,
                all_functions=[get_weather, get_air_quality],
            )
        )
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        worker = await rojak.create_worker([anthropic_activities])

        async with worker:
            context_variables = {"seen": ["test"]}
            task = TaskParams(
                agent=agent,
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather and air quality like in San Francisco?",
                    }
                ],
            )
            run_response = await rojak.run(
                id=str(uuid.uuid4()),
                type="stateless",
                task=task,
                context_variables=context_variables,
            )
            assert isinstance(run_response.result, OrchestratorResponse)
            resp: OrchestratorResponse = run_response.result

            get_weather_mock.assert_called()
            get_air_quality_mock.assert_called_once()
            assert resp.context_variables["seen"] == ["test", "get_air_quality"]
            assert resp.messages[-1].role == "assistant"
            assert resp.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_multiple_tool_calls(mock_anthropic_client: MockAnthropicClient):
    """
    Multiple tool calls returned in a single Anthropic response.
    """
    task_queue_name = str(uuid.uuid4())
    expected_location = "San Francisco"
    get_weather_mock = Mock()
    get_air_quality_mock = Mock()

    def get_weather(location: str, context_variables: dict):
        get_weather_mock(location=location)
        context_variables["seen"].append("get_weather")
        res = f"It's sunny today in {location}"
        return AgentExecuteFnResult(output=res, context_variables=context_variables)

    def get_air_quality(location: str, context_variables: dict):
        get_air_quality_mock(location=location)
        context_variables["seen"].append("get_air_quality")
        res = f"Air quality in {location} is good!"
        return AgentExecuteFnResult(output=res, context_variables=context_variables)

    # mock that triggers both function calls
    mock_anthropic_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "get_weather", "args": {"location": expected_location}},
                    {
                        "name": "get_air_quality",
                        "args": {"location": expected_location},
                    },
                ],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
        ]
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        agent = AnthropicAgent(
            name="Test Agent",
            functions=["get_weather", "get_air_quality"],
        )
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(
                client=mock_anthropic_client,
                all_functions=[get_weather, get_air_quality],
            )
        )
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        worker = await rojak.create_worker([anthropic_activities])

        async with worker:
            context_vars = {"location": expected_location, "seen": []}
            task = TaskParams(
                agent=agent,
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather and air quality like in San Francisco?",
                    }
                ],
            )
            run_response = await rojak.run(
                id=str(uuid.uuid4()),
                type="stateless",
                task=task,
                context_variables=context_vars,
            )
            assert isinstance(run_response.result, OrchestratorResponse)
            resp: OrchestratorResponse = run_response.result

            get_weather_mock.assert_called_once_with(location=expected_location)
            get_air_quality_mock.assert_called_once_with(location=expected_location)
            assert "get_weather" in resp.context_variables["seen"]
            assert "get_air_quality" in resp.context_variables["seen"]
            assert resp.messages[-1].role == "assistant"
            assert resp.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_handoff(mock_anthropic_client: MockAnthropicClient):
    """
    Agent A calls a function that returns agent B, verifying a handoff.
    """
    task_queue_name = str(uuid.uuid4())

    def transfer_to_agent2(context_variables: dict):
        return AgentExecuteFnResult(
            output="handoff to agent2",
            context_variables=context_variables,
            agent=agent2,
        )

    agent1 = AnthropicAgent(name="Test Agent 1", functions=["transfer_to_agent2"])
    agent2 = AnthropicAgent(name="Test Agent 2")

    mock_anthropic_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "transfer_to_agent2"}],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
        ]
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(
                client=mock_anthropic_client, all_functions=[transfer_to_agent2]
            )
        )
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        worker = await rojak.create_worker([anthropic_activities])

        async with worker:
            task = TaskParams(
                agent=agent1,
                messages=[{"role": "user", "content": "I want to talk to agent 2"}],
            )
            run_response = await rojak.run(
                id=str(uuid.uuid4()),
                type="stateless",
                task=task,
            )
            assert isinstance(run_response.result, OrchestratorResponse)
            resp: OrchestratorResponse = run_response.result
            assert resp.agent == agent2
            assert resp.messages[-1].role == "assistant"
            assert resp.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_send_multiple_messages(mock_anthropic_client: MockAnthropicClient):
    """
    Demonstrates sending multiple user messages by calling run repeatedly.
    """
    task_queue_name = str(uuid.uuid4())

    # Two distinct assistant responses in sequence
    mock_anthropic_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT},
            ),
            create_mock_response(
                message={"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT_2},
            ),
        ]
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(client=mock_anthropic_client)
        )
        worker = await rojak.create_worker([anthropic_activities])

        async with worker:
            agent = AnthropicAgent(name="assistant")
            workflow_id = str(uuid.uuid4())

            # First user message
            task_1 = TaskParams(
                agent=agent,
                messages=[{"role": "user", "content": "Hello how are you?"}],
            )
            run_response_1 = await rojak.run(
                id=workflow_id,
                type="persistent",
                task=task_1,
            )
            assert isinstance(run_response_1.result, OrchestratorResponse)
            resp_1: OrchestratorResponse = run_response_1.result
            assert resp_1.agent == agent
            assert resp_1.messages[-1].role == "assistant"
            assert resp_1.messages[-1].content == DEFAULT_RESPONSE_CONTENT

            # Second user message (same workflow_id)
            task_2 = TaskParams(
                agent=agent,
                messages=[{"role": "user", "content": "Hello again?"}],
            )
            run_response_2 = await rojak.run(
                id=workflow_id,
                task=task_2,
            )
            assert isinstance(run_response_2.result, OrchestratorResponse)
            resp_2: OrchestratorResponse = run_response_2.result
            assert resp_2.agent == agent
            assert resp_2.messages[-1].role == "assistant"
            assert resp_2.messages[-1].content == DEFAULT_RESPONSE_CONTENT_2


@pytest.mark.asyncio
async def test_persistent_state_across_calls(
    mock_anthropic_client: MockAnthropicClient,
):
    """
    Shows how we can accumulate context over multiple run() calls, effectively
    replacing 'session' tests from earlier versions of Rojak. The second call
    uses `rojak.get_result(...)` rather than `run(..., task=None)`.
    """
    task_queue_name = str(uuid.uuid4())

    def transfer_agent_b(context_variables: dict):
        context_variables["seen"] = True
        return AgentExecuteFnResult(
            output="Transferred to Agent B",
            context_variables=context_variables,
            agent=agent_b,
        )

    # The mock response triggers a function call, then a final assistant message
    mock_anthropic_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "transfer_agent_b", "args": {}}],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
        ]
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        agent_a = AnthropicAgent(name="Agent A", functions=["transfer_agent_b"])
        agent_b = AnthropicAgent(name="Agent B")

        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(
                client=mock_anthropic_client,
                all_functions=[transfer_agent_b],
            )
        )

        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        worker = await rojak.create_worker([anthropic_activities])

        async with worker:
            workflow_id = str(uuid.uuid4())

            # --- First call: start a persistent workflow with a user message ---
            task_1 = TaskParams(
                agent=agent_a,
                messages=[{"role": "user", "content": "I want to talk to agent B"}],
            )
            run_response_1: RunResponse = await rojak.run(
                id=workflow_id,
                type="persistent",
                task=task_1,
                context_variables={"seen": False},
            )
            assert isinstance(run_response_1.result, OrchestratorResponse)
            resp_1: OrchestratorResponse = run_response_1.result

            # Verify that the agent was handed off to agent_b and the context updated
            assert resp_1.context_variables["seen"] is True
            assert resp_1.agent == agent_b
            assert resp_1.messages[-1].content == DEFAULT_RESPONSE_CONTENT

            # --- Second call: retrieve the final state (no new task) ---
            # We need the task_id from run_response_1 to query that same orchestrator state.
            final_response: OrchestratorResponse = await rojak.get_result(
                id=workflow_id,
                task_id=run_response_1.task_id,
            )

            # Confirm the final state matches what we expect
            assert final_response.context_variables["seen"] is True
            assert final_response.agent == agent_b
            assert final_response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_interrupt_and_approve(mock_anthropic_client: MockAnthropicClient):
    """
    Demonstrates how the orchestrator interrupts a function call,
    returns a ResumeRequest, and how we "approve" the call
    by calling rojak.run(..., resume=ResumeResponse(...)).
    """
    task_queue_name = str(uuid.uuid4())

    def say_hello():
        say_hello_mock()
        return "Hello!"

    mock_anthropic_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "say_hello", "args": {}}],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT_2}
            ),
        ]
    )

    agent = AnthropicAgent(
        functions=["say_hello"],
        interrupts=[Interrupt("say_hello")],
    )

    say_hello_mock = Mock()

    openai_activities = AnthropicAgentActivities(
        AnthropicAgentOptions(
            client=mock_anthropic_client,
            all_functions=[say_hello],
        )
    )
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        worker = await rojak.create_worker([openai_activities])

        async with worker:
            workflow_id = str(uuid.uuid4())
            first_task = TaskParams(
                agent=agent,
                messages=[{"role": "user", "content": "Hello"}],
            )
            run_resp_1 = await rojak.run(
                id=workflow_id, type="persistent", task=first_task
            )

            # We expect a ResumeRequest because we have an interrupt
            assert isinstance(run_resp_1.result, ResumeRequest)
            resume_req = run_resp_1.result
            assert resume_req.tool_name == "say_hello"
            tool_id = resume_req.tool_id  # We'll need this to resume

            approve_resume = ResumeResponse(action="approve", tool_id=tool_id)
            run_resp_2 = await rojak.run(
                id=workflow_id,
                resume=approve_resume,
            )

            # This time, we should get an OrchestratorResponse
            assert isinstance(run_resp_2.result, OrchestratorResponse)
            orch_resp = run_resp_2.result

            say_hello_mock.assert_called_once()
            assert orch_resp.messages[-1].content == DEFAULT_RESPONSE_CONTENT_2


@pytest.mark.asyncio
async def test_interrupt_and_reject(mock_anthropic_client: MockAnthropicClient):
    """
    Demonstrates how we 'reject' an interrupted function call.
    The orchestrator will skip calling the function and continue.
    """
    task_queue_name = str(uuid.uuid4())

    def say_hello():
        say_hello_mock()
        return "Hello!"

    mock_anthropic_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "say_hello", "args": {}}],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT_2}
            ),
        ]
    )

    agent = AnthropicAgent(
        functions=["say_hello"],
        interrupts=[Interrupt("say_hello")],
    )

    say_hello_mock = Mock()

    openai_activities = AnthropicAgentActivities(
        AnthropicAgentOptions(
            client=mock_anthropic_client,
            all_functions=[say_hello],
        )
    )
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        worker = await rojak.create_worker([openai_activities])

        async with worker:
            workflow_id = str(uuid.uuid4())
            first_task = TaskParams(
                agent=agent,
                messages=[{"role": "user", "content": "Hello"}],
            )
            run_resp_1 = await rojak.run(
                id=workflow_id, type="persistent", task=first_task
            )

            # We expect a ResumeRequest because we have an interrupt
            assert isinstance(run_resp_1.result, ResumeRequest)
            resume_req = run_resp_1.result
            assert resume_req.tool_name == "say_hello"
            tool_id = resume_req.tool_id  # We'll need this to resume

            # Reject the function call
            reject_reason = "User does not want this."
            reject_resume = ResumeResponse(
                action="reject", tool_id=tool_id, content=reject_reason
            )
            run_resp_2 = await rojak.run(
                id=workflow_id,
                resume=reject_resume,
            )

            # This time, we should get an OrchestratorResponse
            assert isinstance(run_resp_2.result, OrchestratorResponse)
            orch_resp = run_resp_2.result

            # The function should not have been called
            say_hello_mock.assert_not_called()
            assert reject_reason in orch_resp.messages[-2].content
            assert orch_resp.messages[-1].content == DEFAULT_RESPONSE_CONTENT_2


#
# Below tests only concern the .convert_messages() utility
# in AnthropicAgentActivities; no workflow code is involved.
#


def test_convert_messages_with_parallel_tool_calls():
    conversation_messages = [
        ConversationMessage(
            **{
                "content": "Help provide the weather forecast.",
                "role": "system",
                "sender": None,
                "tool_call_id": None,
                "tool_calls": None,
            }
        ),
        ConversationMessage(
            **{
                "content": "What is the weather like in Malaysia and Singapore?",
                "role": "user",
                "sender": None,
                "tool_call_id": None,
                "tool_calls": None,
            }
        ),
        ConversationMessage(
            **{
                "content": (
                    "I'll help you check the weather for both Malaysia and Singapore. "
                    "I'll retrieve the current weather information for each location.\n\n"
                    "Let's start with Malaysia:"
                ),
                "role": "assistant",
                "sender": "Weather Assistant",
                "tool_call_id": None,
                "tool_calls": [
                    {
                        "function": {
                            "arguments": '{"location": "Kuala Lumpur"}',
                            "name": "get_weather",
                        },
                        "id": "toolu_01Qz54ujndhYL3cGXKY1UukD",
                        "type": "function",
                    },
                    {
                        "function": {
                            "arguments": '{"location": "Singapore"}',
                            "name": "get_weather",
                        },
                        "id": "toolu_01AUvuz1d7UoUrs7SzhpCqnF",
                        "type": "function",
                    },
                ],
            }
        ),
        ConversationMessage(
            **{
                "content": '{"location": "Kuala Lumpur", "temperature": "65", "time": "now"}',
                "role": "tool",
                "sender": "Weather Assistant",
                "tool_call_id": "toolu_01Qz54ujndhYL3cGXKY1UukD",
                "tool_calls": None,
            }
        ),
        ConversationMessage(
            **{
                "content": '{"location": "Singapore", "temperature": "65", "time": "now"}',
                "role": "tool",
                "sender": "Weather Assistant",
                "tool_call_id": "toolu_01AUvuz1d7UoUrs7SzhpCqnF",
                "tool_calls": None,
            }
        ),
    ]

    assert AnthropicAgentActivities.convert_messages(conversation_messages) == (
        [
            {
                "role": "user",
                "content": "What is the weather like in Malaysia and Singapore?",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "id": "toolu_01Qz54ujndhYL3cGXKY1UukD",
                        "input": {"location": "Kuala Lumpur"},
                        "name": "get_weather",
                        "type": "tool_use",
                    },
                    {
                        "id": "toolu_01AUvuz1d7UoUrs7SzhpCqnF",
                        "input": {"location": "Singapore"},
                        "name": "get_weather",
                        "type": "tool_use",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01Qz54ujndhYL3cGXKY1UukD",
                        "content": [
                            {
                                "type": "text",
                                "text": '{"location": "Kuala Lumpur", "temperature": "65", "time": "now"}',
                            },
                        ],
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01AUvuz1d7UoUrs7SzhpCqnF",
                        "content": [
                            {
                                "type": "text",
                                "text": '{"location": "Singapore", "temperature": "65", "time": "now"}',
                            },
                        ],
                    },
                ],
            },
        ],
        "Help provide the weather forecast.",
    )


def test_convert_messages_with_nonparallel_tool_call():
    conversation_messages = [
        ConversationMessage(
            **{
                "content": "Help provide the weather forecast.",
                "role": "system",
                "sender": None,
                "tool_call_id": None,
                "tool_calls": None,
            }
        ),
        ConversationMessage(
            **{
                "content": "What is the weather like in Malaysia and Singapore?",
                "role": "user",
                "sender": None,
                "tool_call_id": None,
                "tool_calls": None,
            }
        ),
        ConversationMessage(
            **{
                "content": "I'll help you check the weather for Malaysia",
                "role": "assistant",
                "sender": "Weather Assistant",
                "tool_call_id": None,
                "tool_calls": [
                    {
                        "function": {
                            "arguments": '{"location": "Kuala Lumpur"}',
                            "name": "get_weather",
                        },
                        "id": "toolu_01Qz54ujndhYL3cGXKY1UukD",
                        "type": "function",
                    },
                ],
            }
        ),
        ConversationMessage(
            **{
                "content": '{"location": "Kuala Lumpur", "temperature": "65", "time": "now"}',
                "role": "tool",
                "sender": "Weather Assistant",
                "tool_call_id": "toolu_01Qz54ujndhYL3cGXKY1UukD",
                "tool_calls": None,
            }
        ),
        ConversationMessage(
            **{
                "content": "I'll help you check the weather for Singapore.",
                "role": "assistant",
                "sender": "Weather Assistant",
                "tool_call_id": None,
                "tool_calls": [
                    {
                        "function": {
                            "arguments": '{"location": "Singapore"}',
                            "name": "get_weather",
                        },
                        "id": "toolu_01AUvuz1d7UoUrs7SzhpCqnF",
                        "type": "function",
                    },
                ],
            }
        ),
        ConversationMessage(
            **{
                "content": '{"location": "Singapore", "temperature": "65", "time": "now"}',
                "role": "tool",
                "sender": "Weather Assistant",
                "tool_call_id": "toolu_01AUvuz1d7UoUrs7SzhpCqnF",
                "tool_calls": None,
            }
        ),
    ]

    assert AnthropicAgentActivities.convert_messages(conversation_messages) == (
        [
            {
                "role": "user",
                "content": "What is the weather like in Malaysia and Singapore?",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "id": "toolu_01Qz54ujndhYL3cGXKY1UukD",
                        "input": {"location": "Kuala Lumpur"},
                        "name": "get_weather",
                        "type": "tool_use",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01Qz54ujndhYL3cGXKY1UukD",
                        "content": [
                            {
                                "type": "text",
                                "text": '{"location": "Kuala Lumpur", "temperature": "65", "time": "now"}',
                            }
                        ],
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "id": "toolu_01AUvuz1d7UoUrs7SzhpCqnF",
                        "input": {"location": "Singapore"},
                        "name": "get_weather",
                        "type": "tool_use",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01AUvuz1d7UoUrs7SzhpCqnF",
                        "content": [
                            {
                                "type": "text",
                                "text": '{"location": "Singapore", "temperature": "65", "time": "now"}',
                            }
                        ],
                    }
                ],
            },
        ],
        "Help provide the weather forecast.",
    )

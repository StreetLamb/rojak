import asyncio
from unittest.mock import Mock
import uuid
import pytest
from rojak import Rojak
from temporalio.testing import WorkflowEnvironment
from rojak.agents import (
    AgentExecuteFnResult,
    AgentInstructionOptions,
    AnthropicAgentActivities,
    AnthropicAgentOptions,
    AnthropicAgent,
)
from rojak.types.types import ConversationMessage
from rojak.workflows import OrchestratorResponse, UpdateConfigParams
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
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(client=mock_anthropic_client)
        )
        worker = await rojak.create_worker([anthropic_activities])
        async with worker:
            agent = AnthropicAgent(name="assistant")
            response: OrchestratorResponse = await rojak.run(
                id=str(uuid.uuid4()),
                agent=agent,
                messages=[{"role": "user", "content": "Hello how are you?"}],
            )
            assert response.messages[-1].role == "assistant"
            assert response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_get_run_result(mock_anthropic_client: MockAnthropicClient):
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(client=mock_anthropic_client)
        )
        worker = await rojak.create_worker([anthropic_activities])
        async with worker:
            agent = AnthropicAgent(name="assistant")
            id = str(uuid.uuid4())
            await rojak.run(
                id=id,
                agent=agent,
                messages=[{"role": "user", "content": "Hello how are you?"}],
            )

            response = await rojak.get_run_result(id)
            assert response.messages[-1].role == "assistant"
            assert response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_callable_instructions(mock_anthropic_client: MockAnthropicClient):
    task_queue_name = str(uuid.uuid4())

    instruct_fn_mock = Mock()

    def instruct_fn(context_variables):
        res = f"My name is {context_variables.get("name")}"
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
            await rojak.run(
                id=str(uuid.uuid4()),
                agent=agent,
                messages=[{"role": "user", "content": "Hello how are you?"}],
                context_variables=context_variables,
            )

            instruct_fn_mock.assert_called_once_with(context_variables)
            assert instruct_fn_mock.return_value == (
                f"My name is {context_variables.get('name')}"
            )


@pytest.mark.asyncio
async def test_failed_tool_call(mock_anthropic_client: MockAnthropicClient):
    """Context variable should be updated by first tool call only since 2nd tool call fails."""
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

    messages = [
        {
            "role": "user",
            "content": "What's the weather and air quality like in San Francisco?",
        }
    ]

    # set mock to return a response that triggers function call
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
            name="Test Agent", functions=["get_weather", "get_air_quality"]
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
            response = await rojak.run(
                id=str(uuid.uuid4()),
                agent=agent,
                messages=messages,
                context_variables=context_variables,
            )
            get_weather_mock.assert_called()
            get_air_quality_mock.assert_called_once()
            assert response.context_variables["seen"] == ["test", "get_air_quality"]
            assert response.messages[-1].role == "assistant"
            assert response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_multiple_tool_calls(mock_anthropic_client: MockAnthropicClient):
    task_queue_name = str(uuid.uuid4())

    expected_location = "San Francisco"

    # set up mock to record function calls
    get_weather_mock = Mock()
    get_air_quality_mock = Mock()

    def get_weather(location: str, context_variables: dict):
        res = f"It's sunny today in {context_variables.get("location")}"
        get_weather_mock(location=location)
        get_weather_mock.return_value = res
        context_variables["seen"].append("get_weather")
        return AgentExecuteFnResult(output=res, context_variables=context_variables)

    def get_air_quality(location: str, context_variables: dict):
        res = f"Air quality in {context_variables.get("location")} is good!"
        get_air_quality_mock(location=location)
        get_air_quality_mock.return_value = res
        context_variables["seen"].append("get_air_quality")
        return AgentExecuteFnResult(output=res, context_variables=context_variables)

    messages = [
        {
            "role": "user",
            "content": "What's the weather and air quality like in San Francisco?",
        }
    ]

    # set mock to return a response that triggers function call
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
            name="Test Agent", functions=["get_weather", "get_air_quality"]
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
            context_variables = {"location": expected_location, "seen": []}
            response = await rojak.run(
                id=str(uuid.uuid4()),
                agent=agent,
                messages=messages,
                context_variables=context_variables,
            )
            get_weather_mock.assert_called_once_with(location=expected_location)
            get_air_quality_mock.assert_called_once_with(location=expected_location)
            assert get_weather_mock.return_value == (
                f"It's sunny today in {context_variables.get("location")}"
            )
            assert "get_weather" in response.context_variables.get("seen")
            assert "get_air_quality" in response.context_variables.get("seen")
            assert response.messages[-1].role == "assistant"
            assert response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_handoff(mock_anthropic_client: MockAnthropicClient):
    task_queue_name = str(uuid.uuid4())

    def transfer_to_agent2():
        return agent2

    agent1 = AnthropicAgent(name="Test Agent 1", functions=["transfer_to_agent2"])
    agent2 = AnthropicAgent(name="Test Agent 2")

    # set mock to return a response that triggers the handoff
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
            response = await rojak.run(
                id=str(uuid.uuid4()),
                agent=agent1,
                messages=[{"role": "user", "content": "I want to talk to agent 2"}],
            )
            print(response.agent)
            print(agent2)
            assert response.agent == agent2
            assert response.messages[-1].role == "assistant"
            assert response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_create_session():
    task_queue_name = str(uuid.uuid4())

    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(client=mock_anthropic_client)
        )
        worker = await rojak.create_worker([anthropic_activities])
        session_id = str(uuid.uuid4())
        async with worker:
            agent = AnthropicAgent(name="assistant")
            session = await rojak.create_session(
                session_id=session_id,
                agent=agent,
            )
            assert session.workflow_handle.id == session_id


@pytest.mark.asyncio
async def test_create_duplicate_session():
    task_queue_name = str(uuid.uuid4())

    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(client=mock_anthropic_client)
        )
        worker = await rojak.create_worker([anthropic_activities])
        session_id = str(uuid.uuid4())
        async with worker:
            agent = AnthropicAgent(name="assistant")
            session = await rojak.create_session(
                session_id=session_id,
                agent=agent,
            )

            session2 = await rojak.create_session(
                session_id=session_id,
                agent=agent,
            )

            run_id_1 = (await session.workflow_handle.describe()).run_id
            run_id_2 = (await session2.workflow_handle.describe()).run_id

            assert run_id_1 == run_id_2


@pytest.mark.asyncio
async def test_send_multiple_messages(mock_anthropic_client: MockAnthropicClient):
    task_queue_name = str(uuid.uuid4())

    mock_anthropic_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT},
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT_2}
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
            session = await rojak.create_session(
                session_id=str(uuid.uuid4()),
                agent=agent,
            )

            response: OrchestratorResponse = await session.send_message(
                agent=agent,
                message={"role": "user", "content": "Hello how are you?"},
            )
            assert response.agent == agent
            assert response.messages[-1].role == "assistant"
            assert response.messages[-1].content == DEFAULT_RESPONSE_CONTENT

            response2: OrchestratorResponse = await session.send_message(
                agent=agent,
                message={"role": "user", "content": "Hello how are you?"},
            )
            assert response2.agent == agent
            assert response2.messages[-1].role == "assistant"
            assert response2.messages[-1].content == DEFAULT_RESPONSE_CONTENT_2


@pytest.mark.asyncio
async def test_session_result(mock_anthropic_client: MockAnthropicClient):
    task_queue_name = str(uuid.uuid4())

    def transfer_agent_b(context_variables: dict):
        context_variables["seen"] = True
        return AgentExecuteFnResult(
            output="Transferred to Agent B",
            context_variables=context_variables,
            agent=agent_b,
        )

    # set mock to return a response that triggers function call
    mock_anthropic_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {
                        "name": "transfer_agent_b",
                        "args": {},
                    },
                ],
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
                client=mock_anthropic_client, all_functions=[transfer_agent_b]
            )
        )
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        worker = await rojak.create_worker([anthropic_activities])
        async with worker:
            session_id = str(uuid.uuid4())
            session = await rojak.create_session(session_id, agent_a, {"seen": False})
            response = await session.send_message(
                {"role": "user", "content": "I want to talk to agent B"},
                agent_a,
            )
            assert response.context_variables["seen"] is True
            assert response.messages[-1].role == "assistant"
            assert response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_get_result(mock_anthropic_client: MockAnthropicClient):
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(client=mock_anthropic_client)
        )
        worker = await rojak.create_worker([anthropic_activities])
        async with worker:
            agent = AnthropicAgent(name="assistant")
            session = await rojak.create_session(
                session_id=str(uuid.uuid4()),
                agent=agent,
            )

            await session.send_message(
                agent=agent,
                message={"role": "user", "content": "Hello how are you?"},
            )

            response: OrchestratorResponse = await session.get_result()

            assert response.agent == agent
            assert response.messages[-1].role == "assistant"
            assert response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_get_config(mock_anthropic_client: MockAnthropicClient):
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(client=mock_anthropic_client)
        )
        worker = await rojak.create_worker([anthropic_activities])
        async with worker:
            agent = AnthropicAgent(name="assistant")
            session = await rojak.create_session(
                session_id=str(uuid.uuid4()),
                agent=agent,
            )

            await session.send_message(
                agent=agent,
                message={"role": "user", "content": "Hello how are you?"},
            )

            response: dict[str, any] = await session.get_config()
            expected_keys = {"debug", "history_size", "context_variables", "max_turns"}
            assert expected_keys.issubset(response.keys())


@pytest.mark.asyncio
async def test_update_config(mock_anthropic_client: MockAnthropicClient):
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        anthropic_activities = AnthropicAgentActivities(
            AnthropicAgentOptions(client=mock_anthropic_client)
        )
        worker = await rojak.create_worker([anthropic_activities])
        async with worker:
            agent = AnthropicAgent(name="assistant")
            session = await rojak.create_session(
                session_id=str(uuid.uuid4()),
                agent=agent,
            )

            await session.send_message(
                agent=agent,
                message={"role": "user", "content": "Hello how are you?"},
            )

            await session.update_config(
                UpdateConfigParams(
                    context_variables={"hello": "world"}, max_turns=100, debug=True
                )
            )

            response: dict[str, any] = await session.get_config()

            assert response["debug"] is True
            assert response["max_turns"] == 100
            assert response["context_variables"].get("hello") == "world"


@pytest.mark.asyncio
async def test_continue_as_new(mock_anthropic_client: MockAnthropicClient):
    task_queue_name = str(uuid.uuid4())

    mock_anthropic_client.set_sequential_responses(
        [
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT_2}
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
            configs = {
                "agent": agent,
                "max_turns": 30,
                "context_variables": {"hello": "world"},
                "history_size": 1,
                "debug": True,
            }
            session = await rojak.create_session(
                session_id=str(uuid.uuid4()),
                agent=configs["agent"],
                max_turns=configs["max_turns"],
                context_variables=configs["context_variables"],
                history_size=configs["history_size"],
                debug=configs["debug"],
            )

            await session.send_message(
                agent=agent,
                message={"role": "user", "content": "Hello how are you?"},
            )
            await asyncio.sleep(1)
            session = await rojak.get_session(session_id=session.workflow_handle.id)
            response = await session.get_config()
            assert response["max_turns"] == configs["max_turns"]
            assert response["context_variables"] == configs["context_variables"]
            assert response["history_size"] == configs["history_size"]
            assert response["debug"] == configs["debug"]


def test_convert_messages():
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
                "content": "I'll help you check the weather for both Malaysia and Singapore. I'll retrieve the current weather information for each location.\n\nLet's start with Malaysia:",
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
                    }
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
        ],
        "Help provide the weather forecast.",
    )

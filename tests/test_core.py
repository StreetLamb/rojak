from unittest.mock import Mock
import uuid
import pytest
from rojak import Rojak
from temporalio.testing import WorkflowEnvironment
from rojak.agents import (
    OpenAIAgent,
    OpenAIAgentActivities,
    OpenAIAgentOptions,
    AgentExecuteFnResult,
    AgentInstructionOptions,
)
from rojak.workflows import OrchestratorResponse, UpdateConfigParams
from tests.mock_client import MockOpenAIClient, create_mock_response

DEFAULT_RESPONSE_CONTENT = "sample response content"

DEFAULT_RESPONSE_CONTENT_2 = "sample response content 2"


@pytest.fixture
def mock_openai_client():
    m = MockOpenAIClient()
    m.set_response(
        create_mock_response({"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT})
    )
    return m


@pytest.mark.asyncio
async def test_run_with_messages(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(client=mock_openai_client)
        )
        worker = await rojak.create_worker([openai_activities])
        async with worker:
            agent = OpenAIAgent(name="assistant")
            response: OrchestratorResponse = await rojak.run(
                id=str(uuid.uuid4()),
                agent=agent,
                messages=[{"role": "user", "content": "Hello how are you?"}],
            )
            assert response.messages[-1].role == "assistant"
            assert response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_callable_instructions(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())

    instruct_fn_mock = Mock()

    def instruct_fn(context_variables):
        res = f"My name is {context_variables.get("name")}"
        instruct_fn_mock(context_variables)
        instruct_fn_mock.return_value = res
        return res

    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(client=mock_openai_client, all_functions=[instruct_fn])
        )
        worker = await rojak.create_worker([openai_activities])
        async with worker:
            agent = OpenAIAgent(
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
async def test_tool_call(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())

    expected_location = "San Francisco"

    # set up mock to record function calls
    get_weather_mock = Mock()

    def get_weather(location: str, context_variables: dict):
        res = f"It's sunny today in {context_variables.get("location")}"
        get_weather_mock(location=location, context_variables=context_variables.copy())
        get_weather_mock.return_value = res
        context_variables["status"] = "completed"
        return AgentExecuteFnResult(output=res, context_variables=context_variables)

    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]

    # set mock to return a response that triggers function call
    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "get_weather", "args": {"location": expected_location}}
                ],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
        ]
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        agent = OpenAIAgent(name="Test Agent", functions=["get_weather"])
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(client=mock_openai_client, all_functions=[get_weather])
        )
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        worker = await rojak.create_worker([openai_activities])
        async with worker:
            context_variables = {"location": expected_location}
            response = await rojak.run(
                id=str(uuid.uuid4()),
                agent=agent,
                messages=messages,
                context_variables=context_variables,
            )
            get_weather_mock.assert_called_once_with(
                location=expected_location, context_variables=context_variables
            )
            assert get_weather_mock.return_value == (
                f"It's sunny today in {context_variables.get("location")}"
            )
            assert response.context_variables.get("status") == "completed"
            assert response.messages[-1].role == "assistant"
            assert response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_handoff(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())

    def transfer_to_agent2():
        return agent2

    agent1 = OpenAIAgent(name="Test Agent 1", functions=["transfer_to_agent2"])
    agent2 = OpenAIAgent(name="Test Agent 2")

    # set mock to return a response that triggers the handoff
    mock_openai_client.set_sequential_responses(
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
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(
                client=mock_openai_client, all_functions=[transfer_to_agent2]
            )
        )
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        worker = await rojak.create_worker([openai_activities])
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
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(client=mock_openai_client)
        )
        worker = await rojak.create_worker([openai_activities])
        session_id = str(uuid.uuid4())
        async with worker:
            agent = OpenAIAgent(name="assistant")
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
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(client=mock_openai_client)
        )
        worker = await rojak.create_worker([openai_activities])
        session_id = str(uuid.uuid4())
        async with worker:
            agent = OpenAIAgent(name="assistant")
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
async def test_send_multiple_messages(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())

    mock_openai_client.set_sequential_responses(
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
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(client=mock_openai_client)
        )
        worker = await rojak.create_worker([openai_activities])
        async with worker:
            agent = OpenAIAgent(name="assistant")
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
async def test_get_result(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(client=mock_openai_client)
        )
        worker = await rojak.create_worker([openai_activities])
        async with worker:
            agent = OpenAIAgent(name="assistant")
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
async def test_get_config(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(client=mock_openai_client)
        )
        worker = await rojak.create_worker([openai_activities])
        async with worker:
            agent = OpenAIAgent(name="assistant")
            session = await rojak.create_session(
                session_id=str(uuid.uuid4()),
                agent=agent,
            )

            await session.send_message(
                agent=agent,
                message={"role": "user", "content": "Hello how are you?"},
            )

            response: dict[str, any] = await session.get_config()
            assert list(response.keys()) == ["debug", "history_size", "max_turns"]


@pytest.mark.asyncio
async def test_update_config(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(client=mock_openai_client)
        )
        worker = await rojak.create_worker([openai_activities])
        async with worker:
            agent = OpenAIAgent(name="assistant")
            session = await rojak.create_session(
                session_id=str(uuid.uuid4()),
                agent=agent,
            )

            await session.send_message(
                agent=agent,
                message={"role": "user", "content": "Hello how are you?"},
            )

            await session.update_config(UpdateConfigParams(max_turns=100, debug=True))

            response: dict[str, any] = await session.get_config()

            assert response["debug"] is True
            assert response["max_turns"] == 100

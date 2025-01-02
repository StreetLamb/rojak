import asyncio
from unittest.mock import patch, Mock
import uuid
import pytest
from rojak import Rojak
from temporalio.testing import WorkflowEnvironment
from rojak.agents import (
    OpenAIAgent,
    OpenAIAgentActivities,
    OpenAIAgentOptions,
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
async def test_max_turns(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())

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

    def transfer_to_agent2():
        return agent2

    agent1 = OpenAIAgent(name="Test Agent 1", functions=["transfer_to_agent2"])
    agent2 = OpenAIAgent(name="Test Agent 2")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(
                client=mock_openai_client,
                all_functions=[transfer_to_agent2],
            )
        )
        worker = await rojak.create_worker([openai_activities])
        async with worker:
            agent = agent1
            session = await rojak.create_session(
                session_id=str(uuid.uuid4()),
                agent=agent,
                max_turns=2,
            )

            response = await session.send_message(
                agent=agent,
                message={"role": "user", "content": "Hello how are you?"},
            )
            print(response.messages)
            # Should not reach agent 2.
            assert response.messages[-1].sender != "Test Agent 2"


@pytest.mark.asyncio
async def test_history_size(mock_openai_client: MockOpenAIClient):
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
                session_id=str(uuid.uuid4()), agent=agent, history_size=1
            )

            response = await session.send_message(
                agent=agent,
                message={"role": "user", "content": "Hello how are you?"},
            )

            assert len(response.messages) == 2

            config = await session.get_config()

            assert len(config.messages) == 1


@pytest.mark.asyncio
async def test_continue_as_new(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())

    mock_workflow_info = Mock()
    mock_workflow_info.get_current_history_size.return_value = 10_001
    mock_workflow_info.get_current_history_length.return_value = 20_000_001

    async with await WorkflowEnvironment.start_time_skipping() as env:
        with patch("temporalio.workflow.info", return_value=mock_workflow_info):
            rojak = Rojak(client=env.client, task_queue=task_queue_name)
            openai_activities = OpenAIAgentActivities(
                OpenAIAgentOptions(client=mock_openai_client)
            )
            worker = await rojak.create_worker([openai_activities])
            async with worker:
                agent = OpenAIAgent(name="assistant")
                configs = {
                    "agent": agent,
                    "max_turns": 30,
                    "context_variables": {"hello": "world"},
                    "history_size": 10,
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

                mock_workflow_info.get_current_history_size.assert_called_once()
                mock_workflow_info.get_current_history_length.assert_called_once()

                session = await rojak.get_session(session_id=session.workflow_handle.id)
                response = await session.get_config()
                assert response.max_turns == configs["max_turns"]
                assert response.context_variables == configs["context_variables"]
                assert response.history_size == configs["history_size"]
                assert response.debug == configs["debug"]
                assert len(response.messages) == 2


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

            await session.update_config(
                UpdateConfigParams(
                    context_variables={"hello": "world"},
                    max_turns=100,
                    debug=True,
                    messages=[{"role": "user", "content": "Hello"}],
                )
            )

            response = await session.get_config()

            assert response.debug is True
            assert response.max_turns == 100
            assert response.context_variables.get("hello") == "world"
            assert response.messages[-1].content == "Hello"

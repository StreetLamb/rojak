from unittest.mock import Mock
import uuid
import pytest
from temporalio.testing import WorkflowEnvironment

from rojak import Rojak
from rojak.agents import (
    OpenAIAgent,
    OpenAIAgentActivities,
    OpenAIAgentOptions,
    AgentExecuteFnResult,
    AgentInstructionOptions,
    Interrupt,
    ResumeRequest,
    ResumeResponse,
)
from rojak.client import RunResponse
from rojak.types import (
    RetryOptions,
    RetryPolicy,
)
from rojak.workflows import OrchestratorResponse, TaskParams
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
            task = TaskParams(
                agent=agent,
                messages=[{"role": "user", "content": "Hello how are you?"}],
            )

            run_response: RunResponse = await rojak.run(
                id=str(uuid.uuid4()),
                type="stateless",
                task=task,
            )

            # The final result can be an OrchestratorResponse or ResumeRequest
            assert isinstance(run_response.result, OrchestratorResponse)
            response: OrchestratorResponse = run_response.result
            assert response.messages[-1].role == "assistant"
            assert response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_get_result(mock_openai_client: MockOpenAIClient):
    """
    Demonstrates that we can re-run the same workflow with no new TaskParams
    and still retrieve the last state (OrchestratorResponse).
    """
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(client=mock_openai_client)
        )
        worker = await rojak.create_worker([openai_activities])

        async with worker:
            agent = OpenAIAgent(name="assistant")
            workflow_id = str(uuid.uuid4())

            # First run with an initial task
            task = TaskParams(
                agent=agent,
                messages=[{"role": "user", "content": "Hello how are you?"}],
            )
            run_response: RunResponse = await rojak.run(
                id=workflow_id,
                type="stateless",
                task=task,
            )

            response = await rojak.get_result(
                id=run_response.id, task_id=run_response.task_id
            )

            assert isinstance(response, OrchestratorResponse)
            first_response: OrchestratorResponse = response
            assert first_response.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_callable_instructions(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())

    instruct_fn_mock = Mock()

    def instruct_fn(context_variables):
        res = f"My name is {context_variables.get('name')}"
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
            task = TaskParams(
                agent=agent,
                messages=[{"role": "user", "content": "Hello how are you?"}],
            )
            run_response: RunResponse = await rojak.run(
                id=str(uuid.uuid4()),
                type="stateless",
                task=task,
                context_variables=context_variables,
            )

            assert isinstance(run_response.result, OrchestratorResponse)
            instruct_fn_mock.assert_called_once_with(context_variables)
            assert instruct_fn_mock.return_value == (
                f"My name is {context_variables.get('name')}"
            )


@pytest.mark.asyncio
async def test_failed_tool_call(mock_openai_client: MockOpenAIClient):
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

    # set mock to return a response that triggers function calls
    mock_openai_client.set_sequential_responses(
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
        agent = OpenAIAgent(
            name="Test Agent",
            functions=["get_weather", "get_air_quality"],
            retry_options=RetryOptions(retry_policy=RetryPolicy(maximum_attempts=5)),
        )
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(
                client=mock_openai_client, all_functions=[get_weather, get_air_quality]
            )
        )
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        worker = await rojak.create_worker([openai_activities])

        async with worker:
            context_vars = {"seen": ["test"]}
            task = TaskParams(agent=agent, messages=messages)
            run_response: RunResponse = await rojak.run(
                id=str(uuid.uuid4()),
                type="stateless",
                task=task,
                context_variables=context_vars,
            )

            assert isinstance(run_response.result, OrchestratorResponse)
            orchestrator_resp: OrchestratorResponse = run_response.result

            get_weather_mock.assert_called()
            get_air_quality_mock.assert_called_once()
            assert orchestrator_resp.context_variables["seen"] == [
                "test",
                "get_air_quality",
            ]
            assert orchestrator_resp.messages[-1].role == "assistant"
            assert orchestrator_resp.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_multiple_tool_calls(mock_openai_client: MockOpenAIClient):
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

    messages = [
        {
            "role": "user",
            "content": "What's the weather and air quality like in San Francisco?",
        }
    ]

    mock_openai_client.set_sequential_responses(
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
        agent = OpenAIAgent(
            name="Test Agent",
            functions=["get_weather", "get_air_quality"],
        )
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(
                client=mock_openai_client, all_functions=[get_weather, get_air_quality]
            )
        )
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        worker = await rojak.create_worker([openai_activities])

        async with worker:
            context_vars = {"location": expected_location, "seen": []}
            task = TaskParams(agent=agent, messages=messages)
            run_response: RunResponse = await rojak.run(
                id=str(uuid.uuid4()),
                type="stateless",
                task=task,
                context_variables=context_vars,
            )

            assert isinstance(run_response.result, OrchestratorResponse)
            orchestrator_resp: OrchestratorResponse = run_response.result

            get_weather_mock.assert_called_once_with(location=expected_location)
            get_air_quality_mock.assert_called_once_with(location=expected_location)
            assert "get_weather" in orchestrator_resp.context_variables["seen"]
            assert "get_air_quality" in orchestrator_resp.context_variables["seen"]
            assert orchestrator_resp.messages[-1].role == "assistant"
            assert orchestrator_resp.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_handoff(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())

    def transfer_to_agent2(context_variables: dict):
        # Transfer to another agent
        return AgentExecuteFnResult(
            output="Handoff to agent2",
            context_variables=context_variables,
            agent=agent2,
        )

    agent1 = OpenAIAgent(name="Test Agent 1", functions=["transfer_to_agent2"])
    agent2 = OpenAIAgent(name="Test Agent 2")

    # mock that triggers the handoff
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
            task = TaskParams(
                agent=agent1,
                messages=[{"role": "user", "content": "I want to talk to agent 2"}],
            )
            run_response: RunResponse = await rojak.run(
                id=str(uuid.uuid4()),
                type="stateless",
                task=task,
            )
            assert isinstance(run_response.result, OrchestratorResponse)
            orchestrator_resp: OrchestratorResponse = run_response.result
            assert orchestrator_resp.agent == agent2
            assert orchestrator_resp.messages[-1].role == "assistant"
            assert orchestrator_resp.messages[-1].content == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_send_multiple_messages(mock_openai_client: MockOpenAIClient):
    """
    Demonstrates sending multiple user messages in separate calls to the same workflow.
    """
    task_queue_name = str(uuid.uuid4())

    # We want two different assistant replies in sequence
    mock_openai_client.set_sequential_responses(
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
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(client=mock_openai_client)
        )
        worker = await rojak.create_worker([openai_activities])

        async with worker:
            agent = OpenAIAgent(name="assistant")
            workflow_id = str(uuid.uuid4())

            # First user message
            task_1 = TaskParams(
                agent=agent,
                messages=[{"role": "user", "content": "Hello how are you?"}],
            )
            run_response_1: RunResponse = await rojak.run(
                id=workflow_id,
                type="persistent",
                task=task_1,
            )
            assert isinstance(run_response_1.result, OrchestratorResponse)
            response_1: OrchestratorResponse = run_response_1.result
            assert response_1.messages[-1].role == "assistant"
            assert response_1.messages[-1].content == DEFAULT_RESPONSE_CONTENT

            # Second user message (same workflow_id)
            task_2 = TaskParams(
                agent=agent,
                messages=[{"role": "user", "content": "What's new today?"}],
            )
            run_response_2: RunResponse = await rojak.run(
                id=workflow_id,
                type="persistent",
                task=task_2,
            )
            assert isinstance(run_response_2.result, OrchestratorResponse)
            response_2: OrchestratorResponse = run_response_2.result
            assert response_2.messages[-1].role == "assistant"
            assert response_2.messages[-1].content == DEFAULT_RESPONSE_CONTENT_2


@pytest.mark.asyncio
async def test_result(mock_openai_client: MockOpenAIClient):
    task_queue_name = str(uuid.uuid4())

    def transfer_agent_b(context_variables: dict):
        context_variables["seen"] = True
        return AgentExecuteFnResult(
            output="Transferred to Agent B",
            context_variables=context_variables,
            agent=agent_b,
        )

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "transfer_agent_b", "args": {}}],
            ),
            create_mock_response(
                message={"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT_2},
            ),
        ]
    )

    agent_a = OpenAIAgent(name="Agent A", functions=["transfer_agent_b"])
    agent_b = OpenAIAgent(name="Agent B")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        openai_activities = OpenAIAgentActivities(
            OpenAIAgentOptions(
                client=mock_openai_client, all_functions=[transfer_agent_b]
            )
        )
        rojak = Rojak(client=env.client, task_queue=task_queue_name)
        worker = await rojak.create_worker([openai_activities])

        async with worker:
            context_vars = {"seen": False}
            task = TaskParams(
                agent=agent_a,
                messages=[{"role": "user", "content": "I want to talk to agent B"}],
            )
            run_response: RunResponse = await rojak.run(
                id=str(uuid.uuid4()),
                type="persistent",
                task=task,
                context_variables=context_vars,
            )

            assert isinstance(run_response.result, OrchestratorResponse)
            orchestrator_resp: OrchestratorResponse = run_response.result
            assert orchestrator_resp.context_variables["seen"] is True
            assert orchestrator_resp.agent == agent_b
            assert orchestrator_resp.messages[-1].role == "assistant"
            assert orchestrator_resp.messages[-1].content == DEFAULT_RESPONSE_CONTENT_2


@pytest.mark.asyncio
async def test_interrupt_and_approve(mock_openai_client: MockOpenAIClient):
    """
    Demonstrates how the orchestrator interrupts a function call,
    returns a ResumeRequest, and how we "approve" the call
    by calling rojak.run(..., resume=ResumeResponse(...)).
    """
    task_queue_name = str(uuid.uuid4())

    def say_hello():
        say_hello_mock()
        return "Hello!"

    mock_openai_client.set_sequential_responses(
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

    agent = OpenAIAgent(
        functions=["say_hello"],
        interrupts=[Interrupt("say_hello")],
    )

    say_hello_mock = Mock()

    openai_activities = OpenAIAgentActivities(
        OpenAIAgentOptions(
            client=mock_openai_client,
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
async def test_interrupt_and_reject(mock_openai_client: MockOpenAIClient):
    """
    Demonstrates how we 'reject' an interrupted function call.
    The orchestrator will skip calling the function and continue.
    """
    task_queue_name = str(uuid.uuid4())

    def say_hello():
        say_hello_mock()
        return "Hello!"

    mock_openai_client.set_sequential_responses(
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

    agent = OpenAIAgent(
        functions=["say_hello"],
        interrupts=[Interrupt("say_hello")],
    )

    say_hello_mock = Mock()

    openai_activities = OpenAIAgentActivities(
        OpenAIAgentOptions(
            client=mock_openai_client,
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

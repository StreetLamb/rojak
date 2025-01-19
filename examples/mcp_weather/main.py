import asyncio
import uuid
from temporalio.client import Client
from rojak.agents import OpenAIAgent, OpenAIAgentActivities
from rojak.client import Rojak
from rojak.types import MCPServerConfig, RetryOptions, RetryPolicy
from rojak.workflows import TaskParams, OrchestratorResponse


async def main():
    client = await Client.connect("localhost:7233")
    rojak = Rojak(client=client, task_queue="tasks")

    agent = OpenAIAgent(
        name="Weather Agent",
        retry_options=RetryOptions(retry_policy=RetryPolicy(maximum_attempts=5)),
    )
    openai_activities = OpenAIAgentActivities()
    worker = await rojak.create_worker(
        [openai_activities],
        mcp_servers={
            "weather": MCPServerConfig("stdio", "python", ["mcp_weather_server.py"])
        },
    )
    try:
        async with worker:
            response = await rojak.run(
                id=str(uuid.uuid4()),
                task=TaskParams(
                    agent=agent,
                    messages=[
                        {
                            "role": "user",
                            "content": "Weather like in San Francisco?",
                        }
                    ],
                ),
                type="stateless",
                debug=True,
            )

            assert isinstance(response.result, OrchestratorResponse)

            print(response.result.messages[-1].content)
    finally:
        await rojak.cleanup_mcp()


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import uuid
from temporalio.client import Client
from rojak.agents.anthropic_agent import (
    AnthropicAgent,
    AnthropicAgentActivities,
    AnthropicAgentOptions,
)
from rojak.client import Rojak
from rojak.types.types import MCPServerConfig, RetryOptions, RetryPolicy


async def main():
    task_queue_name = str(uuid.uuid4())

    client = await Client.connect("localhost:7233")

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco",
        }
    ]

    agent = AnthropicAgent(
        name="Test Agent",
        retry_options=RetryOptions(retry_policy=RetryPolicy(maximum_attempts=5)),
        tool_choice={"disable_parallel_tool_use": True, "type": "auto"},
    )
    anthropic_activities = await AnthropicAgentActivities.create(
        AnthropicAgentOptions(
            mcp_servers={
                "weather": MCPServerConfig(
                    "stdio",
                    "python",
                    [
                        "dummy_server.py",
                    ],
                ),
                "weather2": MCPServerConfig(
                    "stdio",
                    "python",
                    [
                        "dummy_server.py",
                    ],
                ),
            },
        )
    )
    rojak = Rojak(client=client, task_queue=task_queue_name)
    worker = await rojak.create_worker([anthropic_activities])
    async with worker:
        response = await rojak.run(
            id=str(uuid.uuid4()),
            agent=agent,
            messages=messages,
            debug=True,
        )
        print(response)
    for mcp_client in list(anthropic_activities.mcp_clients.values()):
        await mcp_client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

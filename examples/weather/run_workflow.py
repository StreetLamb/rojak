import asyncio
from temporalio import client
from rojak.agents import OpenAIAgent
from rojak.types import RetryOptions, RetryPolicy
from rojak import Rojak
from rojak.workflows import OrchestratorResponse, TaskParams


async def main() -> None:
    temporal_client = await client.Client.connect("localhost:7233")
    rojak = Rojak(temporal_client, task_queue="weather-tasks")

    weather_agent = OpenAIAgent(
        name="Weather Assistant",
        instructions="Help provide the weather forecast.",
        functions=["get_weather", "send_email"],
        retry_options=RetryOptions(
            retry_policy=RetryPolicy(maximum_attempts=5),
            timeout_in_seconds=20,
        ),
        parallel_tool_calls=False,
    )

    response = await rojak.run(
        "weather-session",
        task=TaskParams(
            agent=weather_agent,
            messages=[
                {
                    "role": "user",
                    "content": "What is the weather like in Malaysia and Singapore? Send an email to john@example.com",
                }
            ],
        ),
        max_turns=30,
        debug=True,
        type="persistent",
    )

    assert isinstance(response.result, OrchestratorResponse)

    print(response.result.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())

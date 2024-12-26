import asyncio
from temporalio import client
from rojak.agents import OpenAIAgent
from rojak.types import RetryOptions, RetryPolicy
from rojak import Rojak


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

    session = await rojak.create_session(
        session_id="weather-session",
        agent=weather_agent,
        max_turns=30,
        debug=True,
    )

    response = await session.send_message(
        message={
            "role": "user",
            "content": "What is the weather like in Malaysia and Singapore?",
        },
        agent=weather_agent,
    )
    print(response.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())

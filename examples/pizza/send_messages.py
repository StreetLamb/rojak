from rojak import Rojak
from temporalio.client import Client
import asyncio
from examples.pizza.agents import triage_agent


SESSION_ID = "session_1"


async def main():
    client = await Client.connect("localhost:7233")

    rojak = Rojak(client, task_queue="tasks")

    agent = triage_agent

    session = await rojak.create_session(
        SESSION_ID,
        agent,
        history_size=30,
        debug=True,
        context_variables={
            "name": "John",
            "cart": {},
            "preferences": "Loves healthy food, allergic to nuts.",
        },
    )

    response = await session.update_config({"debug": True})

    while True:
        prompt = input("Enter your message (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break

        response = await session.send_messages(
            [{"role": "user", "content": prompt}],
            agent,
        )

        agent = response.agent  # Update the agent for the next message

        print(response.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())

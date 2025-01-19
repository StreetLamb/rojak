from typing import Literal
from rojak import Rojak
from temporalio.client import Client
import asyncio
from examples.pizza.agents import triage_agent
from rojak.agents import ResumeRequest, ResumeResponse
from rojak.workflows import OrchestratorResponse, TaskParams


SESSION_ID = "session_1"


async def main():
    client = await Client.connect("localhost:7233")

    rojak = Rojak(client, task_queue="tasks")

    agent = triage_agent

    state: Literal["Resume", "Response"] = "Response"

    tool_id: str | None = None
    tool_name: str | None = None

    try:
        configs = await rojak.get_config(SESSION_ID)
        messages = configs.messages
        if messages[-1].tool_calls:
            state = "Resume"
            tool_id = messages[-1].tool_calls[-1]["id"]
            tool_name = messages[-1].tool_calls[-1]["name"]
    except Exception:
        pass

    while True:
        if state == "Response":
            prompt = input("Enter your message (or 'exit' to quit): ")
        else:
            prompt = input(
                f"Resume '{tool_name}'? Enter 'approve' or state why you reject: "
            )

        if prompt.lower() == "exit":
            break

        if state == "Response":
            response = await rojak.run(
                SESSION_ID,
                "long",
                task=TaskParams(
                    messages=[{"role": "user", "content": prompt}], agent=agent
                ),
                context_variables={
                    "name": "John",
                    "cart": {},
                    "preferences": "Loves healthy food, allergic to nuts.",
                },
                debug=True,
            )
        else:
            if prompt == "approve":
                resume = ResumeResponse(action=prompt, tool_id=tool_id)
            else:
                resume = ResumeResponse(
                    action="reject", tool_id=tool_id, content=prompt
                )

            response = await rojak.run(
                SESSION_ID,
                resume=resume,
            )

        if isinstance(response.result, OrchestratorResponse):
            state = "Response"
            agent = response.result.agent
            print(response.result.messages[-1].content)

        elif isinstance(response.result, ResumeRequest):
            print(response.result)
            state = "Resume"
            tool_id = response.result.tool_id
            tool_name = response.result.tool_name
        else:
            print(response)


if __name__ == "__main__":
    asyncio.run(main())

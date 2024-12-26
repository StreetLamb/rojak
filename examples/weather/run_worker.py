from temporalio.client import Client
import asyncio
from rojak import Rojak
from rojak.agents import (
    OpenAIAgentActivities,
    OpenAIAgentOptions,
    AgentExecuteFnResult,
)
import json
from rojak.retrievers import (
    QdrantRetrieverActivities,
    QdrantRetrieverOptions,
)


def get_weather(location, time="now"):
    """Get the current weather in a given location. Location MUST be a city."""
    return json.dumps({"location": location, "temperature": "65", "time": time})


def send_email(recipient, subject, body, context_variables):
    print("Sending email...")
    print(f"To: {recipient}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")
    context_variables["status"] = "sent"
    return AgentExecuteFnResult(
        agent=None, context_variables=context_variables, output="Sent!"
    )


async def main():
    temporal_client: Client = await Client.connect("localhost:7233")

    openai_activities = OpenAIAgentActivities(
        OpenAIAgentOptions(
            all_functions=[get_weather, send_email],
        )
    )

    qdrant_retriever = QdrantRetrieverActivities(
        QdrantRetrieverOptions(
            url="http://localhost:6333", collection_name="demo_collection"
        )
    )

    rojak = Rojak(temporal_client, task_queue="weather-tasks")
    worker = await rojak.create_worker([openai_activities], [qdrant_retriever])
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

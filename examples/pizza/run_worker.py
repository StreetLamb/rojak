# main.py
from temporalio.client import Client
from rojak import Rojak
from rojak.agents import OpenAIAgentActivities, OpenAIAgentOptions
import asyncio
from examples.pizza.functions import (
    to_food_order,
    to_payment,
    to_feedback,
    to_triage,
    get_menu,
    add_to_cart,
    remove_from_cart,
    get_cart,
    process_payment,
    get_receipt,
    provide_feedback,
    food_ordering_instructions,
)


async def main():
    # Create client connected to server at the given address
    client = await Client.connect("localhost:7233")

    openai_activities = OpenAIAgentActivities(
        OpenAIAgentOptions(
            all_functions=[
                to_food_order,
                to_payment,
                to_feedback,
                to_triage,
                get_menu,
                add_to_cart,
                remove_from_cart,
                get_cart,
                process_payment,
                get_receipt,
                provide_feedback,
                food_ordering_instructions,
            ]
        )
    )

    rojak = Rojak(client, task_queue="tasks")
    worker = await rojak.create_worker([openai_activities])
    await worker.run()


if __name__ == "__main__":
    print("Starting worker")
    print("Then run 'python send_messages.py' to start sending messages.")

    asyncio.run(main())

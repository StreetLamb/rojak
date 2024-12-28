# Rojak 

Highly durable and scalable multi-agent orchestration framework.

> [!WARNING]
>
> This project is a work in progress. Please be aware that significant changes may occur.

## Install

Install the core Rojak library:
```shell
pip install rojak
```

To create agents that interact with OpenAI, include the openai extra:
```shell
pip install rojak[openai]
```

To enable Qdrant retrievers, include the qdrant-client extra:
```shell
pip install rojak[qdrant-client]
```

## Usage


```python
# main.py
from temporalio.client import Client
from rojak import Rojak
from rojak.agents import OpenAIAgentActivities, OpenAIAgentOptions, OpenAIAgent
import asyncio


async def main():
    client = await Client.connect("localhost:7233")

    def transfer_to_agent_b():
        return agent_b

    openai_activities = OpenAIAgentActivities(
        OpenAIAgentOptions(
            api_key="...",
            all_functions=[transfer_to_agent_b]
        )
    )

    rojak = Rojak(client, task_queue="tasks")

    worker = await rojak.create_worker([openai_activities])

    async with worker:
        agent_a = OpenAIAgent(
            name="Agent A",
            instructions="You are a helpful agent.",
            functions=["transfer_to_agent_b"]
        )

        agent_b = OpenAIAgent(
            name="Agent B",
            instructions="Only speak in Haikus.",
        )

        response = await rojak.run(
            id="unique-id",
            agent=agent_a,
            messages=[{"role": "user", "content": "I want to talk to agent B."}]
        )

        print(response.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())

```

```
Agent B is here,  
Ready to chat and assist,  
What do you wish for?
```

## Table of Contents

- [Rojak](#rojak)
  - [Install](#install)
  - [Usage](#usage)
  - [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Examples](#examples)
- [Understanding Rojak’s Architecture](#understanding-rojaks-architecture)
- [Running Rojak](#running-rojak)
    - [Workers](#workers)
    - [`rojak.run()`](#rojakrun)
      - [Arguments](#arguments)
      - [`OrchestratorResponse` Fields](#orchestratorresponse-fields)
  - [Agents](#agents)
    - [`Agent` Abstract Class Fields](#agent-abstract-class-fields)
    - [Instructions](#instructions)
    - [Functions](#functions)
    - [Handoffs and Updating Context Variables](#handoffs-and-updating-context-variables)
    - [Function Schemas](#function-schemas)
    - [Retrievers](#retrievers)
  - [Sessions](#sessions)
    - [`rojak.create_session()`](#rojakcreate_session)
      - [Arguments](#arguments-1)
    - [`session.get_session()`](#sessionget_session)
      - [Arguments](#arguments-2)
    - [`session.send_message()`](#sessionsend_message)
      - [Arguments](#arguments-3)
    - [Other Session methods](#other-session-methods)

# Overview

Rojak simplifies the orchestration of reliable multi-agent systems by leveraging Temporal as its backbone. Designed to address the real-world challenges of agentic systems, such as network outages, unreliable endpoints, failures, and long-running processes, Rojak ensures reliability and scalability.

Much like OpenAI’s Swarm, Rojak employs two key concepts:
- **Agents**: These function like individual team members, each responsible for specific tasks and equipped with the necessary tools to accomplish them.
- **Handoffs**: These facilitate seamless transitions, allowing one Agent to pass responsibility or context to another effortlessly.


# Examples

Basic examples can be found in the `/examples` directory:

- [`weather`](examples/weather/): A straightforward example demonstrating tool calling and the use of `context_variables`.


# Understanding Rojak’s Architecture

![Rojak Diagram](assets/rojak_diagram.png)

Rojak is built on Temporal workflows and activities, structured into two main workflow types: the **Orchestrator Workflow** and the **Agent Workflow**.

- The **Orchestrator Workflow** is responsible for receiving the user’s query, executing the Agent Workflow, and passing along the user query and relevant agent information.

- The **Agent Workflow** handles tasks such as retrieving a response from the LLM model and executing tools or functions. These tasks, referred to as **Activities**, are the discrete units of work within the workflow.

**Activities** are method functions grouped by class, with each class representing actions for a specific provider. Base classes like `AgentActivities` and `RetrieverActivities` serve as templates, while concrete classes, such as `OpenAIAgentActivities` for OpenAI and `QdrantRetrieverActivities` for Qdrant Vector DB, implement provider-specific methods. This design ensures flexibility and seamless integration with various providers.

After completing its tasks, the Agent Workflow generates a result containing the agent’s response and the next agent (if any) to hand off to. This result is passed back to the Orchestrator Workflow, which then continues the process by executing the Agent Workflow for the specified agent in the result.

Every step in the workflows is tracked and recorded in the **Temporal Service**, which, in the event of failures, allows the workflow to resume from the previous step. This ensures that workflows are durable, reliable, and recoverable.

While the Temporal Service oversees the workflow, **Workers** are responsible for running the code. **Workers** poll the Temporal Service for tasks and execute them. If there are no running workers, the workflow will not progress. You can deploy not just one worker, but hundreds or even thousands, if necessary, to scale your system’s performance.

# Running Rojak

Ensure that a Temporal Service is running locally. You can find [instructions for setting it up here](https://learn.temporal.io/getting_started/python/dev_environment/#set-up-a-local-temporal-service-for-development-with-temporal-cli).

```shell
$ temporal server start-dev
```

Once the Temporal Service is running, connect the Temporal client to the Temporal Service and use it to instantiate a Rojak client.

```python
from temporalio.client import Client
from rojak import Rojak

temporal_client = await Client.connect("localhost:7233")
rojak = Rojak(temporal_client, task_queue="tasks")
```

### Workers

Workers are responsible for executing the tasks defined in workflows and activities. They poll the Temporal Service for tasks and run the corresponding activity or workflow logic.

To create and start a worker, you first need to define the activities it will handle. For example, if you’re using an OpenAI agent, you must provide the corresponding `OpenAIAgentActivities` configured with appropriate options through `OpenAIAgentOptions`.

Here’s how to create and start a worker:
```python
from rojak.agents import OpenAIAgentActivities, OpenAIAgentOptions

# Initialize Rojak client
rojak = Rojak(temporal_client, task_queue="tasks")

# Initialize an OpenAI agent
agent = OpenAIAgent(name="Agent")

# Define the activities for the OpenAI agent
openai_activities = OpenAIAgentActivities(
    OpenAIAgentOptions(api_key="...")
)

# Create a worker to handle tasks for the defined activities
worker = rojak.create_worker([openai_activities])

# Create a worker to handle tasks for the defined activities
await worker.run()

# Alternatively, use a context manager
# async with worker:
#     response = await rojak.run(...)
```

### `rojak.run()`

Rojak's `run()` function takes `messages` and return `messages` and save no state between calls. Importantly, however, it also handles Agent function execution, hand-offs, context variable references, and can take multiple turns before returning to the user.

At its core, Rojak's `rojak.run()` implements the following loop:

1. Get a completion from the current Agent
2. Execute tool calls and append results
3. Switch Agent if necessary
4. Update context variables, if necessary
5. If no new function calls, return


#### Arguments

| Argument              | Type    | Description                                                                                                                                            | Default        |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------- |
| **id**                | `str`   | Unique identifier of the run.                                                                                                                          | (required)     |
| **agent**             | `Agent` | The (initial) agent to be called.                                                                                                                      | (required)     |
| **messages**          | `List`  | A list of message objects, identical to [Chat Completions `messages`](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages) | (required)     |
| **context_variables** | `dict`  | A dictionary of additional context variables, available to functions and Agent instructions                                                            | `{}`           |
| **max_turns**         | `int`   | The maximum number of conversational turns allowed                                                                                                     | `float("inf")` |
| **debug**             | `bool`  | If `True`, enables debug logging                                                                                                                       |


Once `rojak.run()` is finished (after potentially multiple calls to agents and tools) it will return a Response containing all the relevant updated state. Specifically, the new messages, the last Agent to be called, and the most up-to-date context_variables. You can pass these values (plus new user messages) in to your next execution of `rojak.run()` to continue the interaction where it left off.


#### `OrchestratorResponse` Fields

| Field                 | Type    | Description                                                                                                                                                 |
| --------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **messages**          | `List`  | A list of message objects generated during the conversation. Message object contains a `sender` field indicating which `Agent` the message originated from. |
| **agent**             | `Agent` | The last agent to handle a message.                                                                                                                         |
| **context_variables** | `dict`  | The same as the input variables, plus any changes.                                                                                                          |


## Agents

An `Agent` simply encapsulates a set of `instructions` with a set of `functions` (plus some additional settings below), and has the capability to hand off execution to another `Agent`.

While it's tempting to personify an `Agent` as "someone who does X", it can also be used to represent a very specific workflow or step defined by a set of `instructions` and `functions` (e.g. a set of steps, a complex retrieval, single step of data transformation, etc). This allows `Agent`s to be composed into a network of "agents", "workflows", and "tasks", all represented by the same primitive.

Available built-in `Agent` classes:
- `OpenAIAgent` - For interacting with OpenAI models.


### `Agent` Abstract Class Fields

| Field                   | Type                               | Description                                                                              | Default                          |
| ----------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------- |
| **model**               | `str`                              | The model to be used by the agent.                                                       | (required)                       |
| **name**                | `str`                              | The name of the agent.                                                                   | `"Agent"`                        |
| **instructions**        | `str` or `AgentInstructionOptions` | Instructions for the agent, can be a string or a dict representing the function to call. | `"You are a helpful assistant."` |
| **functions**           | `List[str]`                        | A list of function names that the agent can call.                                        | `[]`                             |
| **tool_choice**         | `Any`                              | The tool choice for the agent, if any.                                                   | `None`                           |
| **parallel_tool_calls** | `bool`                             | Whether model should perform multiple tool calls together.                               | `True`                           |
| **retriever**           | `Retriever`                        | Specify which retriever to use.                                                          | `None`                           |
| **retry_options**       | `RetryOptions`                     | Options for timeout and retries.                                                         | `None`                           |


### Instructions

`Agent` `instructions` are directly converted into the `system` prompt of a conversation (as the first message). Only the `instructions` of the active `Agent` will be present at any given time (e.g. if there is an `Agent` handoff, the `system` prompt will change, but the chat history will not.)

```python
agent = OpenAIAgent(
   instructions="You are a helpful agent."
)
```

The `instructions` can either be a regular `str`, or a function that returns a `str`. The function can optionally receive a `context_variables` parameter, which will be populated by the `context_variables` passed into `rojak.run()`.

```python
def instructions_fn(context_variables):
    user_name = context_variables["user_name"]
    return f"Help the user, {user_name}, do whatever they want."

openai_activities = OpenAIAgentActivities(
    OpenAIAgentOptions(
        all_functions=[instructions_fn]  # Register the instructions function
    )
)

rojak = Rojak(temporal_client, task_queue="tasks")
worker = await rojak.create_worker([openai_activities])

async with worker:
    agent = OpenAIAgent(
        instructions={
            "type": "function", 
            "name": "instructions_fn"
        } # Specify to use the `instruction_fn`
    )
    response = await rojak.run(
        id=str(uuid.uuid4()),
        agent=agent,
        messages=[{"role":"user", "content": "Hi!"}],
        context_variables={"user_name":"John"}
    )
    print(response.messages[-1]["content"])
```

```
Hi John, how can I assist you today?
```

### Functions

- Rojak `Agent`s can call python functions directly.
- Function should usually return a `str` (values will be attempted to be cast as a `str`).
- If a function returns an `Agent`, execution will be transferred to that `Agent`.
- If a function defines a `context_variables` parameter, it will be populated by the `context_variables` passed into `rojak.run()`.
- If an `Agent` function call has an error (missing function, wrong argument, error) an error response will be appended to the chat so the `Agent` can recover gracefully.
- If multiple functions are called by the `Agent`, they will be executed in that order.

```python
def greet(context_variables, language):
   user_name = context_variables["user_name"]
   greeting = "Hola" if language.lower() == "spanish" else "Hello"
   print(f"{greeting}, {user_name}!")
   return "Done"

openai_activities = OpenAIAgentActivities(
    OpenAIAgentOptions(
        all_functions=[greet]  # Register the greet function
    )
)

rojak = Rojak(temporal_client, task_queue="tasks")
worker = await rojak.create_worker([openai_activities])

async with worker:
    agent = OpenAIAgent(
        functions=["greet"]
    )
    response = await rojak.run(
        id=str(uuid.uuid4()),
        agent=agent,
        messages=[{"role": "user", "content": "Usa greet() por favor."}],
        context_variables={"user_name": "John"}
    )
    print(response.messages[-1]["content"])
```

```
Hola, John!
```

### Handoffs and Updating Context Variables

An `Agent` can hand off to another `Agent` by returning it in a `function`.

```python
sales_agent = OpenAIAgent(name="Sales Agent")

def transfer_to_sales():
   return sales_agent

openai_activities = OpenAIAgentActivities(
    OpenAIAgentOptions(
        all_functions=[transfer_to_sales]  # Register the function
    )
)

rojak = Rojak(temporal_client, task_queue="tasks")
worker = await rojak.create_worker([openai_activities])

async with worker:
    agent = OpenAIAgent(functions=["transfer_to_sales"])
    response = rojak.run(
        id=str(uuid.uuid4()),
        agent=agent, 
        messages=[{"role":"user", "content":"Transfer me to sales."}])

    print(response.agent.name)
```

```
Sales Agent
```

It can also update the `context_variables` by returning a more complete `Result` object. This can also contain a `value` and an `agent`, in case you want a single function to return a value, update the agent, and update the context variables (or any subset of the three).

```python
from rojak.agents import AgentExecuteFnResult

def talk_to_sales():
   print("Hello, World!")
   return AgentExecuteFnResult(
       value="Done",
       agent=sales_agent,
       context_variables={"department": "sales"}
   )

openai_activities = OpenAIAgentActivities(
    OpenAIAgentOptions(
        all_functions=[talk_to_sales]  # Register the function
    )
)

rojak = Rojak(temporal_client, task_queue="tasks")
worker = await rojak.create_worker([openai_activities])

async with worker:
    agent = OpenAIAgent(functions=["talk_to_sales"])
    sales_agent = OpenAIAgent(name="Sales Agent")

    response = rojak.run(
        agent=agent,
        messages=[{"role": "user", "content": "Transfer me to sales"}],
        context_variables={"user_name": "John"}
    )
    print(response.agent.name)
    print(response.context_variables)
```

```
Sales Agent
{'department': 'sales', 'user_name': 'John'}
```

> [!NOTE]
> If an `Agent` calls multiple functions to hand-off to an `Agent`, only the last handoff function will be used.


### Function Schemas

Rojak can automatically converts functions into a JSON Schema. For example, when using `OpenAIAgent`, Rojak uses the `function_to_json()` utility function to convert functions into JSON schema that is passed into Chat Completions `tools`.

- Docstrings are turned into the function `description`.
- Parameters without default values are set to `required`.
- Type hints are mapped to the parameter's `type` (and default to `string`).
- Per-parameter descriptions are not explicitly supported, but should work similarly if just added in the docstring. (In the future docstring argument parsing may be added.)

```python
def greet(name, age: int, location: str = "New York"):
   """Greets the user. Make sure to get their name and age before calling.

   Args:
      name: Name of the user.
      age: Age of the user.
      location: Best place on earth.
   """
   print(f"Hello {name}, glad you are {age} in {location}!")
```

```json
{
   "type": "function",
   "function": {
      "name": "greet",
      "description": "Greets the user. Make sure to get their name and age before calling.\n\nArgs:\n   name: Name of the user.\n   age: Age of the user.\n   location: Best place on earth.",
      "parameters": {
         "type": "object",
         "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "location": {"type": "string"}
         },
         "required": ["name", "age"]
      }
   }
}
```

### Retrievers

`Retriever`s are used to fetch relevant information from a large corpus of data or a database in response to a query. They enhance the performance and accuracy of your agents by enabling access to and utilisation of external knowledge sources, making the system more robust and contextually aware.

When a `Retriever` is specified in an `Agent`:
- The agent will query the vector database to retrieve relevant data based on the input query.
- The retrieved data will be appended to the Agent’s instructions, providing the agent with additional context.
- If an error occurs while retrieving data (e.g., a database connection issue), the agent will gracefully proceed without appending any data.

Available built-in retrievers:
- `QdrantRetriever` - Interact with Qdrant service.

Below is an example configuration for an agent that interacts with a local Qdrant service to retrieve relevant data:
```python
from rojak import Rojak
from rojak.agents import OpenAIAgent, OpenAIAgentActivities
from rojak.retrievers import QdrantRetriever, QdrantRetrieverActivities, QdrantRetrieverOptions

# Configure Qdrant Retriever Activities
qdrant_activities = QdrantRetrieverActivities(
    QdrantRetrieverOptions(
        url="http://localhost:6333", 
        collection_name="demo_collection"
    )
)

openai_activities = OpenAIAgentActivities()

rojak = Rojak(temporal_client, task_queue="tasks")

# Create a worker to handle tasks for the agent and retriever activities
worker = await rojak.create_worker(
    agent_activities=[openai_activities], 
    retriever_activities=[qdrant_activities]
)

async with worker:
    agent = OpenAIAgent(
        retriever=QdrantRetriever() # Attach the retriever to the agent
    )
    response = await rojak.run(
        agent=agent,
        messages=[{"role": "user", "content": "Hello, can you tell me more about myself?"}],
    )
    print(response.messages[-1]["content"])
```


## Sessions

Session creates a long-running Orchestrator workflow, allowing for persistence of messages, `context_variables`, and other configurations across multiple calls. This is particularly useful for maintaining a record of messages and updated `context_variables` during long-running conversations. Sessions also allow you to handle multiple concurrent conversations seamlessly.

### `rojak.create_session()`

You can create a session using the rojak.create_session() method by providing the following parameters:
- Session Id: A unique identifier for the session.
- Initial Agent: The agent that will handle the conversation.
- Initial context_variables: The initial state of context_variables for the session.
- Other Settings: Optional settings such as the maximum turns per call or the history size.

If a session with the same session id was already created, `rojak.create_session()` will return the previously created session.

```python
session = await rojak.create_session(
    session_id="unique-session-id",
    agent=initial_agent,
    max_turns=20,
    history_size=15,
    debug=True
)
```

#### Arguments

| Argument              | Type    | Description                                                                                  | Default        |
| --------------------- | ------- | -------------------------------------------------------------------------------------------- | -------------- |
| **session_id**        | `str`   | Unique identifier of the session.                                                            | (required)     |
| **agent**             | `Agent` | The initial agent to be called.                                                              | (required)     |
| **context_variables** | `Agent` | A dictionary of additional context variables, available to functions and Agent instructions. | `{}`           |
| **max_turns**         | `int`   | The maximum number of conversational turns allowed.                                          | `float("inf")` |
| **history_size**      | `int`   | The maximum number of messages.                                                              | `10`           |
| **debug**             | `bool`  | If `True`, enables debug logging.                                                            | `False`        |

The `history_size` argument limits the maximum number of messages. If this limit is reached, Rojak will summarise the conversation history using the previously utilised LLM model. The summary will replace the message history, and Rojak will continue as a new Orchestrator workflow, retaining all current configurations.

### `session.get_session()`

If you have a already running session, you can get it using `session.get_session()` method. This method returns the running session.

#### Arguments
| Argument       | Type  | Description                       | Default    |
| -------------- | ----- | --------------------------------- | ---------- |
| **session_id** | `str` | Unique identifier of the session. | (required) |

### `session.send_message()`

The `send_message()` method passes your message to the agent specified. The response is an `OrchestratorResponse`, similar to the response from `rojak.run()`.

#### Arguments

| Argument    | Type                  | Description                    | Default    |
| ----------- | --------------------- | ------------------------------ | ---------- |
| **message** | `ConversationMessage` | New query as a message object. | (required) |
| **agent**   | `Agent`               | Agent to send the message to.  | (required) |

```python

agent = OpenAIAgent(
    name="Support Agent",
    instructions="Assist users with any issues they face."
)

# Create the session
session = await rojak.create_session(
    session_id="customer-support-123",
    agent=agent,
    context_variables={"order_id": "12345"}
    max_turns=30,
    history_size=20
)

# Send a message
response = await session.send_message(
    message={"role": "user", "content": "Can you help me with my order?"},
    agent=agent
)

print(response.messages[-1]["content"])

# Send another message
await session.send_message(
    message={"role": "user", "content": "Thank you!"},
    agent=agent
)
```

### Other Session methods

- `session.get_result()`: Get the latest `OrchestratorResponse`.
- `session.get_config()`: Get the current configuration.
- `session.update_config()`: Update the current configuration.
- `session.cancel()`: Cancel the long-running workflow.
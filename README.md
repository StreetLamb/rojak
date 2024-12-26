# Rojak 

Highly durable and scalable multi-agent orchestration framework.

> [!WARNING]
>
> This project is a work in progress. Please be aware that significant changes may occur.

## Install

Requires Python 3.10+

```shell
pip install git+ssh://git@github.com/streetlamb/rojak.git
```

or

```shell
pip install git+https://git@github.com/streetlamb/rojak.git
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
  - [What is a Workflow?](#what-is-a-workflow)
  - [What are Activities?](#what-are-activities)
  - [What is Temporal?](#what-is-temporal)
  - [What are Workers?](#what-are-workers)
- [Running Rojak](#running-rojak)
    - [Workers](#workers)
    - [`client.run()`](#clientrun)
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
    - [`client.create_session()`](#clientcreate_session)
      - [Arguments](#arguments-1)
    - [`session.send_message()`](#sessionsend_message)
      - [Arguments](#arguments-2)

# Overview

Rojak simplifies the creation and management of large-scale multi-agent systems by building on Temporal’s durable execution platform. It’s purpose-built to tackle the real-world challenges of agentic systems, such as network outages, unreliable endpoints, failures, and long-running processes, ensuring reliable and scalable multi-agent orchestration without the usual headaches.

Rojak brings together the ease of designing workflows with rock-solid fault tolerance, so you can focus on what your system needs to do without worrying about it breaking down.

At its core, Rojak uses two key ideas:
- **Agents**: These are like individual team members—each one is responsible for specific tasks and has the tools it needs to get the job done.
- **Handoffs**: Think of these as smooth transitions, letting one Agent pass responsibility or context to another seamlessly.


# Examples

Link to examples...


# Understanding Rojak’s Architecture

Rojak is built on Temporal, a powerful framework that simplifies creating and managing workflows. To understand how Rojak operates, it’s essential to grasp some key concepts of Temporal, which provides the foundation for its reliability and scalability.

> [!NOTE]
> If you’re in a hurry: Rojak requires a **running Temporal service** for workflow orchestration. You also need **Workers** that continuously poll the Temporal service for tasks and execute them whenever you send a message to your multi-agent system.

## What is a Workflow?

A workflow is simply a sequence of steps that defines how a task or process unfolds. A multi-agent orchestration workflow might involve:
- Sending a query to an LLM model.
- Executing a tool or function to perform calculations or fetch data.
- Seamlessly handing off tasks between agents.
- Updating shared context variables across steps.

## What are Activities?

Activities are the individual units of work in a workflow, often involving interactions with external systems like APIs or databases. Since these external interactions are prone to failures, activities are designed to handle such scenarios gracefully. In Rojak, activities represent agent actions and are defined as methods within an activity class. Each activity class encapsulates all the methods necessary to fully interact with a specific API or external service, ensuring compatibility even when different APIs are used by various LLM providers.

## What is Temporal?

The Temporal Service is the backbone of Rojak. It ensures workflows are durable, reliable, and recoverable. Here’s how it works:
- Durability: The service tracks every step of a workflow and stores its progress in a database.
- Failure Recovery: If something goes wrong (like a server crash), the Temporal Service can resume the workflow from the last successful step — no work is lost.
- Scalability: Whether you have one task or thousands, the Temporal Service can handle it efficiently.

This design allows Rojak to orchestrate complex multi-agent workflows without you worrying about failures or retries.

## What are Workers?

While the Temporal Service oversees workflow execution, Workers are the ones that actually run your code. A Worker is a component of your application provided by the Temporal SDK. Here’s how it works:
- Workers continuously poll the Temporal Service for tasks.
- When the Temporal Service assigns a task to a Worker, the Worker runs the workflow or activity code for that task.
- Workers directly interact with your data and APIs.

In Rojak, Workers handle tasks like querying LLM models, executing agent functions, or transitioning control between agents. You can deploy multiple Workers — even thousands — to scale up your system’s performance.


# Running Rojak

To run it locally, ensure you have a local Temporal service running. Check out [instructions on how to set it up here](https://learn.temporal.io/getting_started/python/dev_environment/#set-up-a-local-temporal-service-for-development-with-temporal-cli).

```shell
$ temporal server start-dev
```

Once the Temporal service is running, start by instantiating a Rojak client.

```python
from temporalio.client import Client
from rojak import Rojak

temporal_client = await Client.connect("localhost:7233")
client = Rojak(temporal_client, task_queue="tasks")
```

### Workers

Workers are essential components for enabling your agents to execute tasks. At least one active worker is required to support workflow operations. When creating a worker, you must supply all the necessary activity classes for agents and, optionally, any retrievers required by your workflow.

To create and start a worker:
```python
from rojak.agents import OpenAIAgentActivities, OpenAIAgentOptions

agent = OpenAIAgent(name="Agent")

# Since we are using OpenAI agent, we need to provide activities for it.
openai_activities = OpenAIAgentActivities(
    OpenAIAgentOptions(api_key="...")
)

worker = client.create_worker([openai_activities])
await worker.run()

# Alternatively, you can use a context manager.
# async with worker:
#   response = clent.run(...)
```

### `client.run()`

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


Once `client.run()` is finished (after potentially multiple calls to agents and tools) it will return a Response containing all the relevant updated state. Specifically, the new messages, the last Agent to be called, and the most up-to-date context_variables. You can pass these values (plus new user messages) in to your next execution of `client.run()` to continue the interaction where it left off.


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
- OpenAI agent: `OpenAIAgent`


## `Agent` Abstract Class Fields

| Field                   | Type                               | Description                                                                              | Default                      |
| ----------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------- |
| **name**                | `str`                              | The name of the agent.                                                                   | (required)                   |
| **model**               | `str`                              | The model to be used by the agent.                                                       | (required)                   |
| **instructions**        | `str` or `AgentInstructionOptions` | Instructions for the agent, can be a string or a dict representing the function to call. | `"You are a helpful agent."` |
| **functions**           | `List`                             | A list of function names that the agent can call.                                        | `[]`                         |
| **tool_choice**         | `Any`                              | The tool choice for the agent, if any.                                                   | `None`                       |
| **parallel_tool_calls** | `bool`                             | Whether model should perform multiple tool calls together.                               | `True`                       |
| **retriever**           | `Retriever`                        | Specify which retriever to use.                                                          | `None`                       |
| **retry_options**       | `RetryOptions`                     | Options for timeout and retries.                                                         | `None`                       |


### Instructions

`Agent` `instructions` are directly converted into the `system` prompt of a conversation (as the first message). Only the `instructions` of the active `Agent` will be present at any given time (e.g. if there is an `Agent` handoff, the `system` prompt will change, but the chat history will not.)

```python
agent = OpenAIAgent(
   instructions="You are a helpful agent."
)
```

The `instructions` can either be a regular `str`, or a function that returns a `str`. The function can optionally receive a `context_variables` parameter, which will be populated by the `context_variables` passed into `client.run()`.

```python
def instructions(context_variables):
    user_name = context_variables["user_name"]
    return f"Help the user, {user_name}, do whatever they want."

agent = OpenAIAgent(
    instructions=instructions
)
response = client.run(
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

## Functions

- Rojak `Agent`s can call python functions directly.
- Function should usually return a `str` (values will be attempted to be cast as a `str`).
- If a function returns an `Agent`, execution will be transferred to that `Agent`.
- If a function defines a `context_variables` parameter, it will be populated by the `context_variables` passed into `client.run()`.

```python
def greet(context_variables, language):
   user_name = context_variables["user_name"]
   greeting = "Hola" if language.lower() == "spanish" else "Hello"
   print(f"{greeting}, {user_name}!")
   return "Done"

agent = OpenAIAgent(
   functions=[greet]
)

client.run(
    id=str(uuid.uuid4()),
    agent=agent,
    messages=[{"role": "user", "content": "Usa greet() por favor."}],
    context_variables={"user_name": "John"}
)
```

```
Hola, John!
```

- If an `Agent` function call has an error (missing function, wrong argument, error) an error response will be appended to the chat so the `Agent` can recover gracefully.
- If multiple functions are called by the `Agent`, they will be executed in that order.

### Handoffs and Updating Context Variables

An `Agent` can hand off to another `Agent` by returning it in a `function`.

```python
sales_agent = OpenAIAgent(name="Sales Agent")

def transfer_to_sales():
   return sales_agent

agent = OpenAIAgent(name="Triage Agent", functions=[transfer_to_sales])

response = client.run(str(uuid.uuid4()), agent, [{"role":"user", "content":"Transfer me to sales."}])
print(response.agent.name)
```

```
Sales Agent
```

It can also update the `context_variables` by returning a more complete `Result` object. This can also contain a `value` and an `agent`, in case you want a single function to return a value, update the agent, and update the context variables (or any subset of the three).

```python
sales_agent = OpenAIAgent(name="Sales Agent")

def talk_to_sales():
   print("Hello, World!")
   return AgentExecuteFnResult(
       value="Done",
       agent=sales_agent,
       context_variables={"department": "sales"}
   )

agent = OpenAIAgent(functions=[talk_to_sales])

response = client.run(
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

Rojak can automatically converts functions into a JSON Schema. For example, when using `OpenAIAgent`, Rojak uses the `function_to_json()` utils function to convert functions into JSON schema that is passed into Chat Completions `tools`.

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

## Retrievers

`Retriever`s fetch relevant information from a large corpus of data or database in a response to a query. This can help to enhance the performance and accuracy of your agents, especially in tasks that require accessing and utilizing external knowledge sources.

When a `Retriever` is specified in `Agent`, the `Agent` will search for relevant information from the database and insert these relevant information into the `Agent`'s instructions before execution.

Available built-in retrievers:
- Qdrant retriever: `QdrantRetriever`


```python
agent = OpenAIAgent(name="Sales Agent", retriever=QdrantRetriever())

response = client.run(
   agent=agent,
   messages=[{"role": "user", "content": "Hello, can you tell me more about myself?"}],
)
```

## Sessions

Unlike `client.run()` which is does not save states between calls, using `Session` allows you to maintains configurations and conversation history between calls. This is useful for long running conversations where you want to maintain the context of the conversation. Rojak is able to maintain as many sessions as you want.

### `client.create_session()`

You can create a session using the `client.create_session()` method, which initialises a session with a unique ID, an initial agent, and optional settings like maximum turns per call and history size.

```python
session = client.create_session(
    session_id="unique-session-id",
    agent=initial_agent,
    max_turns=20,
    history_size=15,
    debug=True
)
```

#### Arguments

| Argument         | Type    | Description                                                                                                                                  | Default        |
| ---------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| **session_id**   | `str`   | Unique identifier of the session.                                                                                                            | (required)     |
| **agent**        | `Agent` | The initial agent to be called.                                                                                                              | (required)     |
| **max_turns**    | `int`   | The maximum number of conversational turns allowed.                                                                                          | `float("inf")` |
| **history_size** | `int`   | The maximum number of messages retained in the list before older messages are removed. When this limit is exceeded, messages are summarized. | `10`           |
| **debug**        | `bool`  | If `True`, enables debug logging.                                                                                                            | `False`        |


### `session.send_message()`

Once a session is created, use `session.send_message()` to interact with it. The `send_message()` method processes the user’s input and routes the conversation to the appropriate agent while maintaining context and history. The response is same format as response from`client.run()`

```python
from rojak import Rojak
from rojak.agents import OpenAIAgent

# Define the session
session = client.create_session(
    session_id="customer-support-123",
    agent=OpenAIAgent(
        name="Support Agent",
        instructions="Assist users with any issues they face."
    ),
    max_turns=30,
    history_size=20
)

# Send a message
response = session.send_message(
    message={"role": "user", "content": "Can you help me with my order?"},
    context_variables={"order_id": "12345"}
)

print(response.messages[-1]["content"])
```

Then for subsequent messages:
```python

session = client.get_session("customer-support-123")

next_response = session.send_message(
    message={"role": "user", "content": "Thank you!"},
    context_variables={"order_id": "12345"}
)

print(next_response.messages[-1]["content"])
```

#### Arguments

| Argument              | Type                  | Description                                                                                  | Default    |
| --------------------- | --------------------- | -------------------------------------------------------------------------------------------- | ---------- |
| **message**           | `ConversationMessage` | New query as a message object.                                                               | (required) |
| **agent**             | `Agent`               | Agent to send the message to.                                                                | (required) |
| **context_variables** | `dict`                | A dictionary of additional context variables, available to functions and Agent instructions. | `{}`       |
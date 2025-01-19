<p align="center">
    <a href="https://github.com/StreetLamb/rojak/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/StreetLamb/rojak.svg?color=blue"></a>
    <a href="https://github.com/StreetLamb/rojak/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/StreetLamb/rojak.svg"></a>
    <a href="https://github.com/StreetLamb/rojak/actions/workflows/test.yml"><img alt="Test" src="https://github.com/StreetLamb/rojak/actions/workflows/test.yml/badge.svg"></a>
</p>

<h3 align="center">
  <p> Rojak - A library for building highly durable and scalable multi-agent orchestrations.</p>
</h3>

See it in action:

https://github.com/user-attachments/assets/d61b8893-3c33-4002-bca9-740f403f51f1

## Features
- 🛡️ **Durable and Fault-Tolerant** - Agents always completes, even when the server crashes or managing long-running tasks that span weeks, months, or even years.
- 🗂️ **State Management** - Messages, contexts and other states are automatically managed and preserved, even during failures. No complex database transactions required.
- 🤝 **MCP Support** - Supports calling tools via Model Context Protocol (MCP) servers.
- 🧑‍💻 **Human-In-The-Loop** - Integrates human intervention for approving, rejecting, or modifying tool invocations and decisions, ensuring control over critical workflows.
- 📈 **Scalable** - Manage unlimited agents, and handle multiple chat sessions in parallel.
- ⏰ **Scheduling** - Schedule to run your agents at specific times, days, date or intervals.
- 👁️ **Visiblity** - Track your agents’ past and current actions in real time through a user-friendly browser-based UI.
- 🌐 **Universal Deployment** - Deploy and run locally or on any cloud platform.

# Table of Contents

- [Table of Contents](#table-of-contents)
  - [Install](#install)
  - [Usage](#usage)
- [Overview](#overview)
- [Examples](#examples)
- [Understanding Rojak’s Architecture](#understanding-rojaks-architecture)
- [Running Rojak](#running-rojak)
    - [Workers](#workers)
    - [`rojak.run()`](#rojakrun)
    - [Arguments](#arguments)
    - [`ResumeResponse` Fields](#resumeresponse-fields)
    - [Return Value](#return-value)
      - [`RunResponse` Fields](#runresponse-fields)
      - [`OrchestratorResponse` Fields](#orchestratorresponse-fields)
      - [`ResumeRequest` Fields](#resumerequest-fields)
  - [Agents](#agents)
    - [`Agent` Abstract Class Fields](#agent-abstract-class-fields)
    - [Instructions](#instructions)
    - [Functions](#functions)
    - [Handoffs and Updating Context Variables](#handoffs-and-updating-context-variables)
    - [Function Schemas](#function-schemas)
    - [Retrievers](#retrievers)
    - [Timeouts and Retries](#timeouts-and-retries)
  - [Schedules](#schedules)
    - [`rojak.create_schedule()`](#rojakcreate_schedule)
      - [Arguments](#arguments-1)
    - [`rojak.list_scheduled_runs()`](#rojaklist_scheduled_runs)
  - [Human In The Loop](#human-in-the-loop)
    - [Define Interrupts](#define-interrupts)
    - [Handling Interrupts](#handling-interrupts)
    - [Resuming workflows](#resuming-workflows)
  - [Model Context Protocol (MCP) Servers](#model-context-protocol-mcp-servers)

## Install

Install the core Rojak library:
```shell
pip install rojak
```

Install dependencies for the specific model providers you need:
```shell
# For OpenAI models or models supporting OpenAI format e.g. DeepSeek, Ollama.
pip install rojak[openai]

# For Anthropic and Anthropic Bedrock models.
pip install rojak[anthropic]

# To install both
pip install rojak[openai,anthropic]
```

Rojak also supports retrievers to retrieve context from vector stores. Install the dependencies if required:
```shell
# For Qdrant
pip install rojak[qdrant-client]
```

## Usage

Start the Temporal development server.
```shell
temporal server start-dev
```

```python
# main.py
import asyncio
import uuid
from temporalio.client import Client
from rojak import Rojak
from rojak.agents import OpenAIAgentActivities, OpenAIAgentOptions, OpenAIAgent
from rojak.workflows import TaskParams

def transfer_to_agent_b():
    """Handoff to Agent B"""
    return agent_b

agent_a = OpenAIAgent(
    name="Agent A",
    instructions="You are a helpful agent.",
    functions=["transfer_to_agent_b"]
)

agent_b = OpenAIAgent(
    name="Agent B",
    instructions="Only speak in Haikus."
)


async def main():
    # Connect to the Temporal service
    temporal_client = await Client.connect("localhost:7233")

    # Initialize the Rojak client
    rojak = Rojak(temporal_client, task_queue="tasks")

    # Configure agent activities
    openai_activities = OpenAIAgentActivities(
        OpenAIAgentOptions(
            api_key="YOUR_API_KEY_HERE",  # Replace with your OpenAI API key or specify OPENAI_API_KEY in .env 
            all_functions=[transfer_to_agent_b]
        )
    )

    # Create a worker for handling agent activities
    worker = await rojak.create_worker([openai_activities])

    async with worker:
        # Run the workflow with agent A and a handoff to agent B
        response = await rojak.run(
            id=str(uuid.uuid4()),
            type="stateless",
            task=TaskParams(
                agent=agent_a,
                messages=[{"role": "user", "content": "I want to talk to agent B."}]
            ),
        )
        print(response.result.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())
```

```
Agent B is here,  
Ready to chat and assist,  
What do you wish for?
```

View this completed workflow at `http://localhost:8233`.

# Overview

Rojak simplifies the orchestration of reliable multi-agent systems by leveraging Temporal as its backbone. Designed to address the real-world challenges of agentic systems, such as network outages, unreliable endpoints, failures, and long-running processes, Rojak ensures reliability and scalability.

Much like OpenAI’s Swarm, Rojak employs two key concepts:
- **Agents**: These function like individual team members, each responsible for specific tasks and equipped with the necessary tools to accomplish them.
- **Handoffs**: These facilitate seamless transitions, allowing one Agent to pass responsibility or context to another effortlessly.


# Examples

Basic examples can be found in the `/examples` directory:

- [`weather`](examples/weather/): A straightforward example demonstrating tool calling and the use of `context_variables`.
- [`mcp_weather`](examples/mcp_weather/) An example demonstrating connecting to MCP servers and executing tools through them.
- [`pizza`](examples/pizza/) A comprehensive example showcasing the use of multiple agents with human-in-the-loop interventions to help users seamlessly order food.


# Understanding Rojak’s Architecture

![Rojak Diagram](assets/rojak_diagram.png)

Rojak is built on Temporal workflows and activities, and orchestrates agents via the **Orchestrator Workflow**.

- The **Orchestrator Workflow** is responsible for receiving the user’s query, managing the overall execution process, and orchestrating tasks such as retrieving responses from LLM models, executing tools or functions, and handling any necessary Activities.

**Activities** are method functions grouped by class, with each class representing actions for a specific provider. Base classes like `AgentActivities` and `RetrieverActivities` serve as templates, while concrete classes, such as `OpenAIAgentActivities` for OpenAI and `QdrantRetrieverActivities` for Qdrant Vector DB, implement provider-specific methods. This design ensures flexibility and seamless integration with various providers.

After completing its tasks, the Orchestrator Workflow generates a result containing the agent’s response and the next agent (if any) to hand off to. This result is passed back to the Orchestrator Workflow, which then continues the process by executing the specified agent in the result.

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

Rojak's `run()` function orchestrates interactions between agents, tools, and context variables, handling multi-turn conversations and tool interruptions. It allows chaining conversations by returning updated states (messages, context variables, and agents), which can be passed into subsequent calls for continuity.

At its core, Rojak's `run()` executes the following loop:

1. Generate a completion from the current agent.
2. Execute tool calls if needed and append results.
3. Handle agent hand-offs or context updates as necessary.
4. Trigger interruptions (if specified) for approval or rejection.
5. Return the response when all actions are complete or no further actions are needed.

The process may involve multiple turns before returning a final response, depending on the interaction's complexity.

### Arguments

| Argument              | Type             | Description                                                                                    | Default        |
| --------------------- | ---------------- | ---------------------------------------------------------------------------------------------- | -------------- |
| **id**                | `str`            | Unique identifier for the interaction, enabling tracking and continuity across calls.          | (required)     |
| **type**              | `str`            | The run type, either `"persistent"` (multi-turn stateful) or `"stateless"` (single-turn).      | `None`         |
| **task**              | `TaskParams`     | Defines the input task for the orchestrator, including agent, messages, and other parameters.  | `None`         |
| **resume**            | `ResumeResponse` | Allows resuming a paused workflow or interrupt, specifying the action (`approve` or `reject`). | `None`         |
| **context_variables** | `dict`           | A dictionary of additional context variables, accessible by agents and tools during execution. | `{}`           |
| **max_turns**         | `int`            | The maximum number of conversational turns allowed before returning the final result.          | `float("inf")` |
| **debug**             | `bool`           | Enables debug logging when set to `True`.                                                      | `False`        |

**Notes:**
- **`task`** is required for starting a new interaction or adding new user messages.
- **`resume`** is used to continue an interrupted interaction (e.g., after an approval/rejection).
- **`type`** must be specified as `"persistent"` for multi-turn interactions; otherwise, the default is `"stateless"`.

### `ResumeResponse` Fields

The `ResumeResponse` object is used to handle interrupts in a workflow by specifying the action to take on a interrupted tool invocation. It allows users to either approve or reject the tool call, with optional feedback provided when rejecting.

| Field       | Type                      | Description                                                          |
| ----------- | ------------------------- | -------------------------------------------------------------------- |
| **action**  | `"approve"` or `"reject"` | The action to take on the interrupt.                                 |
| **tool_id** | `str`                     | The ID of the tool call to resume.                                   |
| **content** | `str` or `None`           | Feedback to provide to the agent when rejecting the call (optional). |

---
### Return Value

`rojak.run()` returns a `RunResponse` object that encapsulates:

1. The result of the interaction (`OrchestratorResponse` or `ResumeRequest`).
2. The task ID used during the interaction.
3. The workflow handle for the orchestrator.


#### `RunResponse` Fields

| Field               | Type                                      | Description                                                                              |
| ------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------- |
| **id**              | `str`                                     | Unique identifier of the workflow.                                                       |
| **result**          | `OrchestratorResponse` or `ResumeRequest` | The final response from the orchestrator or a request to resume an interrupted workflow. |
| **task_id**         | `str`                                     | A unique identifier for the task, used for tracking and querying results.                |
| **workflow_handle** | `WorkflowHandle`                          | A handle to the orchestrator workflow, allowing advanced operations.                     |


#### `OrchestratorResponse` Fields

| Field                 | Type    | Description                                                                                                                     |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **messages**          | `List`  | A list of messages generated during the interaction. Each message includes a `sender` field indicating the originating `Agent`. |
| **agent**             | `Agent` | The last agent to handle a message during the interaction.                                                                      |
| **context_variables** | `dict`  | The updated context variables, incorporating changes made by tools, agents, or user inputs.                                     |


#### `ResumeRequest` Fields

| Field              | Type       | Description                                                                                  |
| ------------------ | ---------- | -------------------------------------------------------------------------------------------- |
| **tool_id**        | `str`      | The ID of the tool that was interrupted.                                                     |
| **tool_arguments** | `str`      | Arguments that will be passed to the tool when resumed.                                      |
| **task_id**        | `str`      | Unique identifier of the request that triggered the interrupt.                               |
| **tool_name**      | `str`      | The name of the tool that was interrupted.                                                   |
| **question**       | `str`      | A question to ask the user regarding the interrupt.                                          |
| **when**           | `"before"` | Specifies when the interrupt should be triggered (`"before"` is the default and only value). |


## Agents

An `Agent` simply encapsulates a set of `instructions` with a set of `functions` (plus some additional settings below), and has the capability to hand off execution to another `Agent`.

While it's tempting to personify an `Agent` as "someone who does X", it can also be used to represent a very specific workflow or step defined by a set of `instructions` and `functions` (e.g. a set of steps, a complex retrieval, single step of data transformation, etc). This allows `Agent`s to be composed into a network of "agents", "workflows", and "tasks", all represented by the same primitive.

Available built-in `Agent` classes:
- `OpenAIAgent` - For interacting with OpenAI models.
- `AnthropicAgent` - For interacting with Anthropic models.


### `Agent` Abstract Class Fields

| Field                   | Type                               | Description                                                                   | Default                          |
| ----------------------- | ---------------------------------- | ----------------------------------------------------------------------------- | -------------------------------- |
| **model**               | `str`                              | The LLM model to be used by the agent.                                        | (required)                       |
| **name**                | `str`                              | The name of the agent.                                                        | `"Agent"`                        |
| **instructions**        | `str` or `AgentInstructionOptions` | Instructions for the agent, can be a string or a callable returning a string. | `"You are a helpful assistant."` |
| **functions**           | `list[str]`                        | A list of function names that the agent can call.                             | `[]`                             |
| **tool_choice**         | `Any`                              | The tool choice for the agent, if any.                                        | `None`                           |
| **parallel_tool_calls** | `bool`                             | Whether the model should perform multiple tool calls together.                | `True`                           |
| **interrupts**          | `list[Interrupt]`                  | A list of interrupts for reviewing tool use.                                  | `[]`                             |
| **retriever**           | `Retriever`                        | Specify which retriever to use.                                               | `None`                           |
| **retry_options**       | `RetryOptions`                     | Options for timeout and retries.                                              | `RetryOptions()`                 |


### Instructions

`Agent` `instructions` are directly converted into the `system` prompt of a conversation (as the first message). Only the `instructions` of the active `Agent` will be present at any given time (e.g. if there is an `Agent` handoff, the `system` prompt will change, but the chat history will not.)

```python
agent = OpenAIAgent(
   instructions="You are a helpful agent."
)
```

The `instructions` can either be a regular `str`, or a function that returns a `str`. The function can optionally receive a `context_variables` parameter, which will be populated by the `context_variables` passed into `rojak.run()`.

```python
from rojak import Rojak
from rojak.agents import OpenAIAgent, OpenAIAgentActivities, OpenAIAgentOptions
from rojak.workflows import TaskParams
import uuid

def instructions_fn(context_variables):
    user_name = context_variables.get("user_name", "User")
    return f"Help the user, {user_name}, do whatever they want."

openai_activities = OpenAIAgentActivities(
    OpenAIAgentOptions(
        all_functions=[instructions_fn]  # Register the instructions function
    )
)

rojak = Rojak(client=temporal_client, task_queue="tasks")
worker = await rojak.create_worker([openai_activities])

async with worker:
    agent = OpenAIAgent(
        instructions={
            "type": "function", 
            "name": "instructions_fn"
        }  # Specify to use the `instructions_fn`
    )

    task = TaskParams(
        agent=agent,
        messages=[{"role": "user", "content": "Hi!"}],
        context_variables={"user_name": "John"},
    )

    response = await rojak.run(
        id=str(uuid.uuid4()),  # Unique identifier for the workflow
        type="stateless",  # Specify the workflow type
        task=task,
    )

    if isinstance(response.result, OrchestratorResponse):
        print(response.result.messages[-1].content)
    else:
        print("Unexpected result:", response.result)
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
from rojak import Rojak
from rojak.agents import OpenAIAgent, OpenAIAgentActivities, OpenAIAgentOptions
from rojak.workflows import TaskParams
import uuid

def greet(context_variables, language):
    user_name = context_variables.get("user_name", "User")
    greeting = "Hola" if language.lower() == "spanish" else "Hello"
    print(f"{greeting}, {user_name}!")
    return "Done"

openai_activities = OpenAIAgentActivities(
    OpenAIAgentOptions(
        all_functions=[greet]
    )
)

rojak = Rojak(client=temporal_client, task_queue="tasks")
worker = await rojak.create_worker([openai_activities])

async with worker:
    agent = OpenAIAgent(
        functions=["greet"]
    )

    task = TaskParams(
        agent=agent,
        messages=[{"role": "user", "content": "Usa greet() por favor."}],
        context_variables={"user_name": "John"}
    )

    response = await rojak.run(
        id=str(uuid.uuid4()),
        type="stateless",
        task=task,
    )

    if isinstance(response.result, OrchestratorResponse):
        print(response.result.messages[-1].content)
    else:
        print("Unexpected result:", response.result)
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
    OpenAIAgentOptions(all_functions=[transfer_to_sales])
)

rojak = Rojak(temporal_client, task_queue="tasks")
worker = await rojak.create_worker([openai_activities])

async with worker:
    agent = OpenAIAgent(functions=["transfer_to_sales"])
    response = await rojak.run(
        id=str(uuid.uuid4()),
        type="stateless",
        task=TaskParams(
            agent=agent,
            messages=[{"role": "user", "content": "Transfer me to sales."}],
        ),
    )
    print(response.result.agent.name)
```

```
Sales Agent
```

It can also update the `context_variables` by returning a more complete `Result` object. This can also contain a `value` and an `agent`, in case you want a single function to return a value, update the agent, and update the context variables (or any subset of the three).

```python
from rojak import Rojak
from rojak.agents import OpenAIAgent, OpenAIAgentActivities, OpenAIAgentOptions, AgentExecuteFnResult
from rojak.workflows import TaskParams
import uuid

sales_agent = OpenAIAgent(name="Sales Agent")

def talk_to_sales(context_variables):
    print("Hello, World!")
    return AgentExecuteFnResult(
        output="Done",
        agent=sales_agent,
        context_variables={**context_variables, "department": "sales"}
    )

openai_activities = OpenAIAgentActivities(
    OpenAIAgentOptions(all_functions=[talk_to_sales])
)

rojak = Rojak(client=temporal_client, task_queue="tasks")
worker = await rojak.create_worker([openai_activities])

async with worker:
    agent = OpenAIAgent(functions=["talk_to_sales"])
    response = await rojak.run(
        id=str(uuid.uuid4()),
        type="stateless",
        task=TaskParams(
            agent=agent,
            messages=[{"role": "user", "content": "Transfer me to sales"}],
            context_variables={"user_name": "John"},
        ),
    )
    print(response.result.agent.name)
    print(response.result.context_variables)
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
from rojak.agents import OpenAIAgent, OpenAIAgentActivities, OpenAIAgentOptions
from rojak.retrievers import QdrantRetriever, QdrantRetrieverActivities, QdrantRetrieverOptions
from rojak.workflows import TaskParams
import uuid

# Configure Qdrant Retriever Activities
qdrant_activities = QdrantRetrieverActivities(
    QdrantRetrieverOptions(
        url="http://localhost:6333",
        collection_name="demo_collection"
    )
)

# Configure OpenAI Agent Activities
openai_activities = OpenAIAgentActivities(
    OpenAIAgentOptions()
)

# Initialize Rojak and create a worker for the agent and retriever activities
rojak = Rojak(client=temporal_client, task_queue="tasks")
worker = await rojak.create_worker(
    [openai_activities, qdrant_activities]
)

async with worker:
    retriever = QdrantRetriever()  # Define the retriever instance
    agent = OpenAIAgent(retriever=retriever)  # Attach the retriever to the agent

    response = await rojak.run(
        id=str(uuid.uuid4()),
        type="stateless",
        task=TaskParams(
            agent=agent,
            messages=[{"role": "user", "content": "Hello, can you tell me more about myself?"}],
        ),
    )
    print(response.result.messages[-1].content)
```

### Timeouts and Retries

Rojak leverages Temporal’s built-in durability and fault tolerance to ensure robust and reliable workflows. However, you can further fine-tune this behavior using `RetryOptions`, which provides extensive configuration for handling timeouts and retries.

With `RetryOptions`, you can customise parameters such as the maximum number of retry attempts, timeout durations, backoff coefficients, and specify exceptions that should not trigger retries. This level of control allows you to adapt to the specific needs of your workflow.

For instance, if you have a tool-calling function that might take a long time to complete, you can change the timeout to 2 minutes. Additionally, you can change the retry attempts to 10 times in case of failure before abandoning the operation. Here’s an example:
```python
from rojak.types import RetryOptions, RetryPolicy
from rojak.agents.openai_agent import OpenAIAgent

# Create an agent with a custom timeout and retry policy.
agent = OpenAIAgent(retry_options=RetryOptions(
    timeout_in_seconds=120,
    retry_policy=RetryPolicy(
        maximum_attempts=10
    )
))
```


## Schedules

Schedules allow you to automatically execute workflows at specific times, on specific days or dates, or at regular intervals, making them ideal for automating recurring tasks or time-based operations.

### `rojak.create_schedule()`

You can create a schedule by specifying the timing details (schedule_spec) and the required inputs for each associated workflow.

#### Arguments

| Argument              | Type           | Description                                                                             | Default        |
| --------------------- | -------------- | --------------------------------------------------------------------------------------- | -------------- |
| **schedule_id**       | `str`          | Unique identifier of the schedule.                                                      | (required)     |
| **schedule_spec**     | `ScheduleSpec` | Specification on when the action is taken.                                              | (required)     |
| **task**              | `TaskParams`   | Encapsulates the agent, messages, and additional parameters for the scheduled workflow. | (required)     |
| **context_variables** | `dict`         | Additional context variables available to functions and agent instructions.             | `{}`           |
| **max_turns**         | `int`          | Maximum number of conversational turns allowed.                                         | `float("inf")` |
| **history_size**      | `int`          | Maximum number of messages retained in the conversation history.                        | `10`           |
| **debug**             | `bool`         | If True, enables debug logging.                                                         | `False`        |


```python
from rojak import Rojak, ScheduleSpec, ScheduleIntervalSpec
from datetime import timedelta
from temporalio.client import Client
from rojak.workflows import TaskParams

temporal_client = await Client.connect("localhost:7233")
rojak = Rojak(temporal_client, task_queue="tasks")

# Create schedule to start a run every hour.
await rojak.create_schedule(
    schedule_id="schedule_123",
    schedule_spec=ScheduleSpec(
        intervals=[ScheduleIntervalSpec(every=timedelta(hours=1))]
    ),
    task=TaskParams(
        agent=OpenAIAgent(),
        messages=[{"role": "user", "content": "Hello"}],
    ),
)
```

### `rojak.list_scheduled_runs()`

This method retrieves a list of orchestrator workflow IDs associated with a schedule. This can be combined with `rojak.get_run_result()` to access the `OrchestratorResponse` of each run:

```python
rojak = Rojak(temporal_client, task_queue="tasks")

agent_activities = OpenAIAgentActivities()

worker = await rojak.create_worker(agent_activities=[agent_activities])

async for workflow_id in rojak.list_scheduled_runs(schedule_id, statuses=["Completed"]):
    async with worker:
        response = await rojak.get_result(workflow_id)
        print(response.messages[-1].content)
    break
```

```
Hello! How can I assist you today?
```

## Human In The Loop

This feature ensures that humans can seamlessly intervene in workflows to maintain control over critical decisions. This is especially useful in scenarios requiring validation, approval, or manual oversight before proceeding with automated tasks.

### Define Interrupts

Use interrupts to identify points in the workflow where human intervention is required. For example, you can configure an interrupt to pause the workflow when a specific tool is invoked.

```python
from rojak.agent import Interrupt

 agent = OpenAIAgent(
     name="Agent A",
     functions=["process_payment"],
     interrupts=[Interrupt("process_payment")],
 )
```

### Handling Interrupts

When an interrupt is triggered, Rojak pauses the workflow and returns a ResumeRequest, prompting the human to decide whether to approve or reject the tool invocation.

```python
if isinstance(response.result, ResumeRequest):
    print(f"Interrupt triggered: {response.result.tool_name}")
    resume_response = ResumeResponse(action="approve", tool_id=response.result.tool_id)
    final_response = await rojak.run(
        id="workflow_id",
        resume=resume_response,
    )
```

### Resuming workflows

Based on the human’s decision, the workflow either continues execution (on approval) or skips the interrupted action (on rejection) and retry again. Optional feedback can be provided during rejection to inform the agent.

```python
reject_response = ResumeResponse(
    action="reject",
    tool_id=response.result.tool_id,
    content="The tool call is unnecessary."
)
```


## Model Context Protocol (MCP) Servers

MCP is an open protocol that standardises how applications provide context to LLMs, similar to how USB-C standardises device connections. It enables a consistent way to connect AI models to various data sources and tools.

To connect to MCP servers, specify the servers you want to connect to using `MCPServerConfig`. Rojak supports two types of communication with MCP servers:
- **Server-Sent Events (SSE)**: For receiving real-time updates via HTTP.
- **Standard Input/Output (stdio)**: For local processes.

Below is an example demonstrating how to configure MCP servers and handle requests:
```python
# For full code see examples/mcp_weather/main.py
import uuid
from rojak import Rojak
from rojak.agents import OpenAIAgent, OpenAIAgentActivities, OpenAIAgentOptions
from rojak.types import MCPServerConfig
from rojak.workflows import TaskParams

openai_activities = OpenAIAgentActivities(
    OpenAIAgentOptions()
)

rojak = Rojak(client=temporal_client, task_queue="tasks")

# Create a worker with MCP Server configuration
worker = await rojak.create_worker(
    [openai_activities],
    mcp_servers={
        "weather": MCPServerConfig(
            type="stdio",
            command="python",
            args=["mcp_weather_server.py"],
        ),
        # Example of additional server
        # "example_server": MCPServerConfig(
        #     type="sse",
        #     url="http://localhost:8000/sse",
        # ),
    },
)

agent = OpenAIAgent()

try:
    async with worker:
        response = await rojak.run(
            id=str(uuid.uuid4()),
            type="stateless",
            task=TaskParams(
                agent=agent,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the weather like in San Francisco?",
                    }
                ],
            ),
            debug=True,
        )
        print(response.result.messages[-1].content)
finally:
    # Clean up MCP resources
    await rojak.cleanup_mcp()
```

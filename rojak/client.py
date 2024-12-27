import warnings
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleSpec,
    ScheduleHandle,
)
from temporalio import workflow
from temporalio.worker import Worker
from temporalio.exceptions import WorkflowAlreadyStartedError
from rojak.types import ConversationMessage
from rojak.session import Session

with workflow.unsafe.imports_passed_through():
    from rojak.retrievers import RetrieverActivities
    from rojak.agents import Agent, AgentActivities
    from rojak.workflows import (
        AgentWorkflow,
        OrchestratorWorkflow,
        OrchestratorParams,
        OrchestratorResponse,
        ShortOrchestratorParams,
        ShortOrchestratorWorkflow,
    )


class Rojak:
    def __init__(self, client: Client, task_queue: str):
        self.client: Client = client
        self.task_queue: str = task_queue

    async def create_worker(
        self,
        agent_activities: list[AgentActivities],
        retriever_activities: list[RetrieverActivities] = [],
    ) -> Worker:
        """Create a worker.

        Args:
            agent_activities (list[AgentActivities]): List of activity classes that can be called.
            retriever_activities (list[RetrieverActivities], optional): List of retriever activity classes that can be called. Defaults to [].

        Returns:
            Worker: A worker object that can be used to start the worker.
        """
        activities = []
        for activity in agent_activities:
            activities.append(activity.call)
            activities.append(activity.execute_function)
            activities.append(activity.execute_instructions)

        for retriever in retriever_activities:
            activities.append(retriever.retrieve_and_combine_results)

        worker: Worker = Worker(
            self.client,
            task_queue=self.task_queue,
            workflows=[OrchestratorWorkflow, AgentWorkflow, ShortOrchestratorWorkflow],
            activities=activities,
        )

        return worker

    async def create_schedule(
        self,
        schedule_id: str,
        run_id: str,
        schedule_spec: ScheduleSpec,
        agent: Agent,
        messages: list[ConversationMessage],
        context_variables: dict = {},
        max_turns: int = float("inf"),
        debug: bool = False,
    ) -> ScheduleHandle:
        """Create a schedule and return its handle.

        Args:
            schedule_id (str): Unique identifier of the schedule.
            run_id (str): Unique identifier of the run.
            schedule_spec (ScheduleSpec): Specification on when the action is taken.
            agent (Agent): The initial agent to be called.
            messages (list[ConversationMessage]): A list of message objects.
            context_variables (dict, optional): A dictionary of additional context variables, available to functions and Agent instructions. Defaults to {}.
            max_turns (int, optional): The maximum number of conversational turns allowed. Defaults to float("inf").
            debug (bool, optional): If True, enables debug logging. Defaults to False.

        Returns:
            ScheduleHandle: A handle to the created schedule.
        """
        data = ShortOrchestratorParams(
            agent, max_turns, messages, debug, context_variables
        )

        return await self.client.create_schedule(
            schedule_id,
            Schedule(
                action=ScheduleActionStartWorkflow(
                    ShortOrchestratorWorkflow.run,
                    data,
                    id=run_id,
                    task_queue=self.task_queue,
                ),
                spec=schedule_spec,
            ),
        )

    async def run(
        self,
        id: str,
        agent: Agent,
        messages: list[ConversationMessage],
        context_variables: dict = {},
        max_turns: int = float("inf"),
        debug: bool = False,
    ) -> OrchestratorResponse:
        """Send messages to initial agent and wait for completion.

        Args:
            id (str): Unique identifier of the run.
            agent (Agent): The initial agent to be called.
            messages (list[ConversationMessage]): A list of message objects.
            context_variables (dict, optional): A dictionary of additional context variables, available to functions and Agent instructions. Defaults to {}.
            max_turns (int, optional): The maximum number of conversational turns allowed. Defaults to float("inf").
            debug (bool, optional): If True, enables debug logging. Defaults to False.

        Returns:
            OrchestratorResponse: A response object containing updated messages and context_variables.
        """
        data = ShortOrchestratorParams(
            agent, context_variables, max_turns, messages, debug
        )
        return await self.client.execute_workflow(
            ShortOrchestratorWorkflow.run,
            data,
            id=id,
            task_queue=self.task_queue,
        )

    async def get_session(self, session_id: str) -> Session:
        """Retrieve the session with the given ID.

        Args:
            session_id (str): The unique identifier for the session.

        Raises:
            ValueError: If no session with the specified ID exists.

        Returns:
            Session: The Session object associated with the given ID.
        """
        try:
            workflow_handle = self.client.get_workflow_handle(session_id)
            description = await workflow_handle.describe()
            if description.raw_info.type.name == "OrchestratorWorkflow":
                return Session(workflow_handle)
            else:
                raise
        except Exception:
            raise ValueError(
                f"Session with ID {session_id} does not exist. Please create a session first."
            )

    async def create_session(
        self,
        session_id: str,
        agent: Agent,
        context_variables: dict = {},
        max_turns: int = float("inf"),
        history_size: int = 10,
        debug: bool = False,
    ) -> Session:
        """Create a session if not yet started. The session will maintain conversation history and configurations.

        Args:
            session_id (str): Unique identifier of the session.
            agent (Agent): The initial agent to be called.
            context_variables (dict, optional): A dictionary of additional context variables, available to functions and Agent instructions. Defaults to {}.
            max_turns (int, optional): The maximum number of conversational turns allowed. Defaults to float("inf").
            history_size (int, optional): The maximum number of messages retained in the list before older messages are removed. When this limit is exceeded, the messages are summarized, and the summary becomes the first message in a new list. Defaults to 10.
            debug (bool, optional): If True, enables debug logging. Defaults to False.

        Returns:
            Session: The Session object created.
        """
        data = OrchestratorParams(
            agent=agent,
            context_variables=context_variables,
            max_turns=max_turns,
            history_size=history_size,
            debug=debug,
        )
        try:
            workflow_handle = await self.client.start_workflow(
                OrchestratorWorkflow.run,
                data,
                id=session_id,
                task_queue=self.task_queue,
            )
            return Session(workflow_handle)
        except WorkflowAlreadyStartedError:
            warnings.warn(
                "A session with this ID is already running. Returning the existing session.",
                UserWarning,
            )
            workflow_handle = self.client.get_workflow_handle(session_id)
            return Session(workflow_handle)

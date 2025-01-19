from dataclasses import dataclass
from typing import AsyncIterator, Literal, overload
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleSpec,
    ScheduleHandle,
    WithStartWorkflowOperation,
    WorkflowHandle,
)
from temporalio import workflow, common
from temporalio.worker import Worker
from rojak.types import ConversationMessage, MCPServerConfig, InitMcpResult

with workflow.unsafe.imports_passed_through():
    from mcp import Tool
    from rojak.retrievers import RetrieverActivities
    from rojak.agents import Agent, AgentActivities
    from rojak.mcp.mcp_client import MCPClient
    from rojak.workflows import (
        OrchestratorResponse,
        OrchestratorParams,
        OrchestratorWorkflow,
        GetConfigResponse,
        UpdateConfigParams,
    )
    from rojak.workflows.agent_workflow import ResumeRequest, ResumeResponse
    from rojak.workflows.orchestrator_workflow import TaskParams
    from uuid import uuid4


@dataclass
class RunResponse:
    id: str
    result: OrchestratorResponse | ResumeRequest
    task_id: str
    workflow_handle: WorkflowHandle


class Rojak:
    def __init__(self, client: Client, task_queue: str):
        self.client: Client = client
        self.task_queue: str = task_queue
        self.mcp_result: InitMcpResult | None = None

    async def _init_mcp(self, servers: dict[str, MCPServerConfig]) -> None:
        """Initialise MCP servers.

        Args:
            servers (dict[str, MCPServerConfig]): List of MCP servers.

        Returns:
            tuple[dict[str, MCPClient], dict[str, Tool], dict[str, str]]: Response as tuple.
        """
        mcp_clients: dict[str, MCPClient] = {}
        mcp_tools: dict[str, Tool] = {}
        tool_client_mapping: dict[str, str] = {}
        for server_name, config in servers.items():
            try:
                mcp_client = MCPClient()
                await mcp_client.connect_to_server(config)
                list_tools_result = await mcp_client.session.list_tools()
                mcp_clients[server_name] = mcp_client
                for tool in list_tools_result.tools:
                    mcp_tools[tool.name] = tool
                    tool_client_mapping[tool.name] = server_name
            except Exception as e:
                print(f"Unable to connect to MCP server. Skipping. Error: {e}")
        self.mcp_result = InitMcpResult(mcp_clients, mcp_tools, tool_client_mapping)
        if mcp_tools:
            print(f"MCP tools loaded: {list(mcp_tools.keys())}")

    async def cleanup_mcp(self):
        """Cleanup MCP connections."""
        for client in list(self.mcp_result.clients.values())[::-1]:
            await client.cleanup()

    async def create_worker(
        self,
        agent_activities: list[AgentActivities],
        retriever_activities: list[RetrieverActivities] = [],
        mcp_servers: dict[str, MCPServerConfig] = {},
    ) -> Worker:
        """Create a worker.

        Args:
            agent_activities (list[AgentActivities]): List of activity classes that can be called.
            retriever_activities (list[RetrieverActivities], optional): List of retriever activity classes that can be called. Defaults to [].
            mcp_servers (dict[str, MCPServerConfig], optional): Dictionary of MCP server configurations. Each key represents the server name, and the value is its corresponding MCPServerConfig object. Defaults to {}.

        Returns:
            Worker: A worker object that can be used to start the worker.
        """
        await self._init_mcp(mcp_servers)
        activities = []
        for activity in agent_activities:
            if self.mcp_result:
                activity._add_mcp_configs(self.mcp_result)
            activities.append(activity.call)
            activities.append(activity.execute_function)
            activities.append(activity.execute_instructions)

        for retriever in retriever_activities:
            activities.append(retriever.retrieve_and_combine_results)

        worker: Worker = Worker(
            self.client,
            task_queue=self.task_queue,
            workflows=[OrchestratorWorkflow],
            activities=activities,
        )

        return worker

    async def list_scheduled_runs(
        self,
        schedule_id: str,
        statuses: list[
            Literal[
                "Running", "Completed", "Failed", "Cancelled", "Terminated", "TimedOut"
            ]
        ]
        | None = None,
        limit: int = 10,
        page_size: int = 1000,
        next_page_token: bytes | None = None,
    ) -> AsyncIterator[str]:
        """List the ID of orchestrators associated with the schedule.

        Args:
            schedule_id (str): Unique identifier of the schedule.
            statuses (list[ Literal[ 'Running', 'Completed', 'Failed', 'Cancelled', 'Terminated', 'TimedOut' ] ] | None, optional): List of statuses to filter the runs. Defaults to None.
            limit (int, optional): Maximum number of IDs to return. Defaults to 10.
            page_size (int, optional): Maximum number of results per page. Defaults to 1000.
            next_page_token (bytes | None, optional): A previously obtained next page token if doing pagination. Usually not needed as the iterator automatically starts from the beginning. Defaults to None.

        Returns:
            AsyncIterator[str]:  An async iterator that can be used with `async for`.
        """
        status_filter = (
            " OR ".join(f'ExecutionStatus="{status}"' for status in statuses)
            if statuses
            else ""
        )
        query = f'TemporalScheduledById="{schedule_id}"'
        if status_filter:
            query += f" AND ({status_filter})"

        async for workflow_execution in self.client.list_workflows(
            query=query,
            limit=limit,
            page_size=page_size,
            next_page_token=next_page_token,
        ):
            yield workflow_execution.id

    async def create_schedule(
        self,
        schedule_id: str,
        schedule_spec: ScheduleSpec,
        agent: Agent,
        messages: list[ConversationMessage],
        context_variables: dict = {},
        max_turns: int = float("inf"),
        debug: bool = False,
    ) -> ScheduleHandle:
        """Create a schedule and return its handle.

        The schedule periodically executes the equivalent of the `run()` method with the provided inputs.

        Args:
            schedule_id (str): Unique identifier of the schedule.
            schedule_spec (ScheduleSpec): Specification on when the action is taken.
            agent (Agent): The initial agent to be called.
            messages (list[ConversationMessage]): A list of message objects.
            context_variables (dict, optional): A dictionary of additional context variables, available to functions and Agent instructions. Defaults to {}.
            max_turns (int, optional): The maximum number of conversational turns allowed. Defaults to float("inf").
            debug (bool, optional): If True, enables debug logging. Defaults to False.

        Returns:
            ScheduleHandle: A handle to the created schedule.
        """
        data = OrchestratorParams(
            agent=agent,
            context_variables=context_variables,
            max_turns=max_turns,
            messages=messages,
            debug=debug,
            type="stateless",
        )

        return await self.client.create_schedule(
            schedule_id,
            Schedule(
                action=ScheduleActionStartWorkflow(
                    OrchestratorWorkflow.run,
                    data,
                    id=schedule_id,
                    task_queue=self.task_queue,
                ),
                spec=schedule_spec,
            ),
        )

    @overload
    async def run(
        self,
        id: str,
        type: Literal["stateless", "session"],
        task: TaskParams,
        context_variables: dict = {},
        max_turns: int = float("inf"),
        debug: bool = False,
    ) -> RunResponse: ...

    @overload
    async def run(self, id: str, resume: ResumeResponse) -> RunResponse: ...

    async def run(
        self,
        id: str,
        type: Literal["stateless", "persistent"] | None = None,
        task: TaskParams | None = None,
        resume: ResumeResponse | None = None,
        context_variables: dict = {},
        max_turns: int = float("inf"),
        history_size: int = 10,
        debug: bool = False,
    ) -> RunResponse:
        """
        Initialize and execute an orchestrator with the provided inputs, handling tasks, resuming workflows,
        and waiting for completion.

        Requires a running worker.

        Args:
            id (str): Unique identifier of the orchestrator.
            type (Literal["stateless", "persistent"] | None, optional): Whether to keep track of prior conversations through long-running workflows.
            task (TaskParams | None, optional): A task to be executed in the orchestrator. Defaults to None.
            resume (ResumeResponse | None, optional): A resume object for continuing a paused workflow. Defaults to None.
            context_variables (dict, optional): A dictionary of additional context variables available to functions
                and agent instructions. Defaults to an empty dictionary.
            max_turns (int, optional): The maximum number of conversational turns allowed. Defaults to infinity.
            history_size (int, optional): The maximum number of messages retained in the list before older messages are
                removed. When this limit is exceeded, older messages are removed. Defaults to 10.
            debug (bool, optional): If True, enables debug logging for the orchestrator. Defaults to False.

        Returns:
            RunResponse:
                - OrchestratorResponse: A response object containing updated messages, context variables, and agent
                  information, if the workflow completes successfully.
                - ResumeRequest: A request to resume the workflow if the current state requires further inputs or actions.
                - WorkflowHandle: A handle to the orchestrator workflow.

        Notes:
            - If a `task` is provided, the method starts a new orchestrator workflow.
            - If a `resume` is provided, the method resumes the specified workflow.
            - A new workflow is initialized with `context_variables`, `max_turns`, and `debug` settings.
        """
        start_op = None
        task_id = str(uuid4())
        if task:
            start_op = WithStartWorkflowOperation(
                OrchestratorWorkflow.run,
                OrchestratorParams(
                    context_variables=context_variables,
                    max_turns=max_turns,
                    debug=debug,
                    history_size=history_size,
                    type=type,
                ),
                id=id,
                id_conflict_policy=common.WorkflowIDConflictPolicy.USE_EXISTING,
                task_queue=self.task_queue,
            )
            result = await self.client.execute_update_with_start_workflow(
                OrchestratorWorkflow.add_task,
                (task_id, task),
                start_workflow_operation=start_op,
                result_type=OrchestratorResponse | ResumeRequest,
            )
            workflow_handle = await start_op.workflow_handle()
        else:
            workflow_handle = self.client.get_workflow_handle_for(
                workflow=OrchestratorWorkflow, workflow_id=id
            )

            result: (
                OrchestratorResponse | ResumeRequest
            ) = await workflow_handle.execute_update(
                OrchestratorWorkflow.add_task,
                (task_id, resume),
                result_type=OrchestratorResponse | ResumeRequest,
            )

        return RunResponse(
            id=workflow_handle.id,
            result=result,
            task_id=task_id,
            workflow_handle=workflow_handle,
        )

    async def get_result(self, id: str, task_id: str) -> OrchestratorResponse:
        """Get the latest response.

        Requires a running worker.

        Returns:
            OrchestratorResponse: Response object containing updated messages and context_variables.
        """
        return await self.client.get_workflow_handle(id).query(
            OrchestratorWorkflow.get_result, task_id
        )

    async def get_config(self, id: str) -> GetConfigResponse:
        """Retrieve the current session configuration.

        Requires a running worker.

        Returns:
            GetConfigResponse: Current session's configuration values.
        """
        return await self.client.get_workflow_handle(id).query(
            OrchestratorWorkflow.get_config, result_type=GetConfigResponse
        )

    async def update_config(self, id: str, params: UpdateConfigParams):
        """Update the session's configuration with specified changes.

        Requires a running worker.

        Args:
            params (UpdateConfigParams): A dictionary containing only the configuration values that need to be updated.
        """
        await self.client.get_workflow_handle(id).signal(
            OrchestratorWorkflow.update_config, params
        )

    async def cancel(self, id: str):
        """Cancel the session."""
        return await self.client.get_workflow_handle(id).cancel()

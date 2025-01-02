from temporalio.client import (
    WorkflowHandle,
)
from temporalio import workflow
from rojak.types import ConversationMessage

with workflow.unsafe.imports_passed_through():
    from rojak.agents import Agent
    from rojak.workflows import (
        OrchestratorWorkflow,
        OrchestratorResponse,
        SendMessagesParams,
        UpdateConfigParams,
        GetConfigResponse,
    )


class Session:
    def __init__(self, workflow_handle: WorkflowHandle):
        self.workflow_handle = workflow_handle

    async def send_messages(
        self,
        messages: list[ConversationMessage],
        agent: Agent,
    ) -> OrchestratorResponse:
        """Send messages to the agent specified.

        Args:
            messages (list[ConversationMessage]): New query as a list of message object.
            agent (Agent): Agent to send message to.
            context_variables (dict, optional): A dictionary of additional context variables, available to functions and Agent instructions. Defaults to {}.

        Returns:
            OrchestratorResponse: A response object containing updated messages, context_variables and agent.
        """
        return await self.workflow_handle.execute_update(
            OrchestratorWorkflow.send_messages,
            SendMessagesParams(messages, agent),
            result_type=OrchestratorResponse,
        )

    async def get_result(self) -> OrchestratorResponse:
        """Get the latest response.

        Requires a running worker.

        Returns:
            OrchestratorResponse: Response object containing updated messages and context_variables.
        """
        return await self.workflow_handle.query(OrchestratorWorkflow.get_result)

    async def get_config(self) -> GetConfigResponse:
        """Retrieve the current session configuration.

        Requires a running worker.

        Returns:
            GetConfigResponse: Current session's configuration values.
        """
        return await self.workflow_handle.query(
            OrchestratorWorkflow.get_config, result_type=GetConfigResponse
        )

    async def update_config(self, params: UpdateConfigParams):
        """Update the session's configuration with specified changes.

        Requires a running worker.

        Args:
            params (UpdateConfigParams): A dictionary containing only the configuration values that need to be updated.
        """
        await self.workflow_handle.signal(OrchestratorWorkflow.update_config, params)

    async def cancel(self):
        """Cancel the session."""
        return await self.workflow_handle.cancel()

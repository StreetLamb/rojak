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
        SendMessageParams,
        UpdateConfigParams,
    )


class Session:
    def __init__(self, workflow_handle: WorkflowHandle):
        self.workflow_handle = workflow_handle

    async def send_message(
        self,
        message: ConversationMessage,
        agent: Agent,
    ) -> OrchestratorResponse:
        """Send a message to the first agent

        Args:
            message (ConversationMessage): New query as a message object.
            agent (Agent): Agent to send message to.
            context_variables (dict, optional): A dictionary of additional context variables, available to functions and Agent instructions. Defaults to {}.

        Returns:
            OrchestratorResponse: A response object containing updated messages and context_variables.
        """
        return await self.workflow_handle.execute_update(
            OrchestratorWorkflow.send_message,
            SendMessageParams(message, agent),
            result_type=OrchestratorResponse,
        )

    async def get_result(self) -> OrchestratorResponse:
        """Get the latest response.

        Returns:
            OrchestratorResponse: Response object containing updated messages and context_variables.
        """
        return await self.workflow_handle.query(OrchestratorWorkflow.get_result)

    async def get_config(self) -> dict[str, any]:
        """Retrieve the current session configuration.

        Returns:
            dict[str, any]: A dictionary with the current session's configuration values.
        """
        return await self.workflow_handle.query(OrchestratorWorkflow.get_config)

    async def update_config(self, params: UpdateConfigParams):
        """Update the session's configuration with specified changes.

        Args:
            params (UpdateConfigParams): A dictionary containing only the configuration values that need to be updated.
        """
        await self.workflow_handle.signal(OrchestratorWorkflow.update_config, params)

    async def cancel(self):
        """Cancel the session."""
        return await self.workflow_handle.cancel()

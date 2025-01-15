from mcp import ClientSession, StdioServerParameters
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from rojak.types import MCPServerConfig


class MCPClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, config: MCPServerConfig):
        """Connect to an MCP server"""

        if config.type == "stdio":
            params = StdioServerParameters(command=config.command, args=config.args)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
        else:
            stdio_transport = await self.exit_stack.enter_async_context(
                sse_client(config.url)
            )
            self.sse, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.sse, self.write)
            )
        await self.session.initialize()

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


from smolagents import MCPClient

def get_browser_mcp_client():
    """
    Returns an MCPClient connected to the local Browser MCP.
    """
    return MCPClient(
        server_parameters={
            "url": "http://127.0.0.1:12306/mcp",
            "transport": "streamable-http"
        },
        structured_output=True
    )

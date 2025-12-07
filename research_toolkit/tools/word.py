
from smolagents import MCPClient

def get_word_mcp_client():
    """
    Returns an MCPClient connected to the local Word MCP.
    """
    return MCPClient(
        server_parameters={
            "url": "http://127.0.0.1:12307/mcp",
            "transport": "streamable-http"
        },
        structured_output=True
    )

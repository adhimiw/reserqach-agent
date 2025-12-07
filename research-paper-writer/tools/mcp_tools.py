import os
import sys
from mcp import StdioServerParameters
from smolagents import ToolCollection
from config import Config

# Adjust path to include the Word MCP server directory if needed
# But we are running it as a subprocess, so we just need the path to the script.
WORD_MCP_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Office-Word-MCP-Server-main/word_mcp_server.py"))

def get_perplexity_mcp_tools():
    """
    Returns a ToolCollection for Perplexity MCP.
    """
    server_params = StdioServerParameters(
        command="uvx",
        args=["perplexity-mcp"],
        env={
            "PERPLEXITY_API_KEY": Config.PERPLEXITY_API_KEY,
            "PERPLEXITY_MODEL": Config.PERPLEXITY_MODEL,
            **os.environ
        }
    )
    return ToolCollection.from_mcp(server_params, trust_remote_code=True)

def get_word_mcp_tools():
    """
    Returns a ToolCollection for Word MCP.
    """
    # We use the python executable from the current environment
    python_exe = sys.executable
    
    server_params = StdioServerParameters(
        command=python_exe,
        args=[WORD_MCP_SCRIPT_PATH],
        env=os.environ
    )
    return ToolCollection.from_mcp(server_params, trust_remote_code=True)

def get_tavily_mcp_tools():
    """
    Returns a ToolCollection for Tavily Remote MCP.
    """
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "mcp-remote", f"https://mcp.tavily.com/mcp/?tavilyApiKey={Config.TAVILY_API_KEY}"],
        env=os.environ
    )
    return ToolCollection.from_mcp(server_params, trust_remote_code=True)

def get_filesystem_mcp_tools(allowed_directories: list[str]):
    """
    Returns a ToolCollection for Filesystem MCP.
    """
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem"] + allowed_directories,
        env=os.environ
    )
    return ToolCollection.from_mcp(server_params, trust_remote_code=True)

def get_browser_mcp_tools():
    """
    Returns a ToolCollection for the Streamable Browser MCP (HTTP/SSE).
    """
    server_params = {
        "url": "http://127.0.0.1:12306/mcp",
        "transport": "streamable-http"
    }
    return ToolCollection.from_mcp(server_params, trust_remote_code=True)


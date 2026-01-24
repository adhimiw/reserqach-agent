"""
Unified MCP Tools Integration for Data Science System
Consolidates all MCP server connections including Pandas, Jupyter, Perplexity, and Word
"""

import os
import sys
from mcp import StdioServerParameters
from smolagents import ToolCollection
from config import Config


# ============== Helper Functions ==============

def _get_pandas_mcp_tools():
    """Get Pandas MCP tools for data manipulation"""
    pandas_mcp_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "pandas_mcp.py")
    )
    python_exe = sys.executable
    
    server_params = StdioServerParameters(
        command=python_exe,
        args=[pandas_mcp_path],
        env=os.environ
    )
    return ToolCollection.from_mcp(server_params, trust_remote_code=True)


def _get_jupyter_mcp_tools():
    """Get Jupyter MCP tools for code execution and notebook generation"""
    jupyter_mcp_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "jupyter_mcp.py")
    )
    python_exe = sys.executable
    
    server_params = StdioServerParameters(
        command=python_exe,
        args=[jupyter_mcp_path],
        env=os.environ
    )
    return ToolCollection.from_mcp(server_params, trust_remote_code=True)


def _get_perplexity_tools():
    """Get Perplexity MCP tools for real-time web research and reasoning"""
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


def _get_word_mcp_tools():
    """Get Word MCP tools for document generation"""
    word_mcp_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../Office-Word-MCP-Server-main/word_mcp_server.py")
    )
    python_exe = sys.executable
    
    server_params = StdioServerParameters(
        command=python_exe,
        args=[word_mcp_path],
        env=os.environ
    )
    return ToolCollection.from_mcp(server_params, trust_remote_code=True)


def _get_filesystem_tools(allowed_directories=None):
    """Get Filesystem MCP tools for file operations"""
    if allowed_directories is None:
        allowed_directories = [Config.OUTPUT_DIR]
    
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem"] + allowed_directories,
        env=os.environ
    )
    return ToolCollection.from_mcp(server_params, trust_remote_code=True)


# ============== Public API ==============

def get_data_processing_tools():
    """
    Get all tools needed for data processing (Pandas, Jupyter)
    
    Returns:
        dict: Dictionary with 'pandas' and 'jupyter' tool collections
    """
    return {
        'pandas': _get_pandas_mcp_tools(),
        'jupyter': _get_jupyter_mcp_tools()
    }


def get_research_tools():
    """
    Get all tools needed for research (Perplexity, Filesystem)
    
    Returns:
        dict: Dictionary with 'perplexity' and 'filesystem' tool collections
    """
    return {
        'perplexity': _get_perplexity_tools(),
        'filesystem': _get_filesystem_tools()
    }


def get_writing_tools():
    """
    Get all tools needed for writing reports (Word, Filesystem)
    
    Returns:
        dict: Dictionary with 'word' and 'filesystem' tool collections
    """
    return {
        'word': _get_word_mcp_tools(),
        'filesystem': _get_filesystem_tools()
    }


def get_all_tools():
    """
    Get all available tools for complete data science workflow
    
    Returns:
        dict: Dictionary with all available tool collections
    """
    return {
        **get_data_processing_tools(),
        **get_research_tools(),
        **get_writing_tools()
    }


def combine_tool_lists(*tool_collections):
    """
    Combine multiple tool collections into a single list
    
    Args:
        *tool_collections: Variable number of tool collections
    
    Returns:
        list: Combined list of all tools
    """
    combined_tools = []
    for collection in tool_collections:
        if hasattr(collection, 'tools'):
            combined_tools.extend(list(collection.tools))
        elif isinstance(collection, list):
            combined_tools.extend(collection)
    return combined_tools

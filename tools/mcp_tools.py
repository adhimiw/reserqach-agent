"""
Unified MCP Tools Integration for Data Science System
Consolidates all MCP server connections through a single Docker MCP Gateway
"""

import os
import sys
from mcp import StdioServerParameters
from smolagents import ToolCollection
from config import Config


# ============== Helper Functions ==============

def _get_docker_mcp_tools():
    """Get tools from the Docker MCP Gateway"""
    if not hasattr(Config, 'DOCKER_MCP_ENABLED') or not Config.DOCKER_MCP_ENABLED:
        return None
        
    server_params = StdioServerParameters(
        command=Config.DOCKER_MCP_COMMAND,
        args=Config.DOCKER_MCP_ARGS,
        env=os.environ
    )
    try:
        return ToolCollection.from_mcp(server_params, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Failed to connect to Docker MCP Gateway: {e}")
        # Return an empty collection or handle as needed
        return None


# ============== Public API ==============

def get_data_processing_tools():
    """
    Get all tools needed for data processing (via Docker MCP)
    
    Returns:
        dict: Dictionary with tool collections
    """
    docker_tools = _get_docker_mcp_tools()
    return {
        'docker_mcp': docker_tools
    }


def get_research_tools():
    """
    Get all tools needed for research (via Docker MCP)
    
    Returns:
        dict: Dictionary with tool collections
    """
    docker_tools = _get_docker_mcp_tools()
    return {
        'docker_mcp': docker_tools
    }


def get_writing_tools():
    """
    Get all tools needed for writing reports (via Docker MCP)
    
    Returns:
        dict: Dictionary with tool collections
    """
    docker_tools = _get_docker_mcp_tools()
    return {
        'docker_mcp': docker_tools
    }


def get_all_tools():
    """
    Get all available tools for complete data science workflow
    
    Returns:
        dict: Dictionary with all available tool collections
    """
    docker_tools = _get_docker_mcp_tools()
    if docker_tools:
        return {'docker_mcp': docker_tools}
    return {}


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
        if collection is None:
            continue
        if hasattr(collection, 'tools'):
            combined_tools.extend(list(collection.tools))
        elif isinstance(collection, dict):
            for item in collection.values():
                if item and hasattr(item, 'tools'):
                    combined_tools.extend(list(item.tools))
        elif isinstance(collection, list):
            combined_tools.extend(collection)
    return combined_tools

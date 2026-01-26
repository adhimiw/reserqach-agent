"""
Unified MCP Tools Integration for Data Science System
Consolidates all MCP server connections through a single Docker MCP Gateway
"""

import os
import json
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


def _load_mcp_config() -> dict:
    """Load MCP server config from a local config file if available."""
    config_path = os.getenv("MCP_CONFIG_PATH")
    if not config_path:
        config_path = os.path.expanduser("~/.config/Code/User/mcp.json")
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _resolve_env_value(value: str) -> str:
    """Resolve ${VAR} or ${input:VAR} placeholders to environment values."""
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        key = value[2:-1]
        if key.startswith("input:"):
            key = key.split(":", 1)[1]
        return os.getenv(key, "")
    return value


def _get_local_mcp_tools(server_names):
    """Load MCP tools from local mcp.json config for the given server names."""
    config = _load_mcp_config()
    servers = {}
    if isinstance(config.get("servers"), dict):
        servers.update(config["servers"])
    if isinstance(config.get("mcpServers"), dict):
        servers.update(config["mcpServers"])
    
    tool_collections = {}
    for name in server_names:
        server_cfg = servers.get(name)
        if not server_cfg:
            continue
        command = server_cfg.get("command")
        args = server_cfg.get("args", [])
        if not command:
            continue
        env = os.environ.copy()
        for k, v in (server_cfg.get("env") or {}).items():
            env[k] = _resolve_env_value(v)
        try:
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )
            tool_collections[name] = ToolCollection.from_mcp(server_params, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Failed to connect to MCP server '{name}': {e}")
    return tool_collections


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
    local_tools = _get_local_mcp_tools(["playwright"])
    return {
        'docker_mcp': docker_tools,
        **local_tools
    }


def get_writing_tools():
    """
    Get all tools needed for writing reports (via Docker MCP)
    
    Returns:
        dict: Dictionary with tool collections
    """
    docker_tools = _get_docker_mcp_tools()
    local_tools = _get_local_mcp_tools(["word-document-server"])
    return {
        'docker_mcp': docker_tools,
        **local_tools
    }


def get_all_tools():
    """
    Get all available tools for complete data science workflow
    
    Returns:
        dict: Dictionary with all available tool collections
    """
    docker_tools = _get_docker_mcp_tools()
    local_tools = _get_local_mcp_tools(["word-document-server", "playwright"])
    tools = {}
    if docker_tools:
        tools['docker_mcp'] = docker_tools
    tools.update(local_tools)
    return tools


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

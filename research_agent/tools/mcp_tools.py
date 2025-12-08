"""
Advanced MCP Tools Integration
Multi-server connection manager with async support, caching, and error handling
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from functools import lru_cache
from contextlib import asynccontextmanager

from mcp import StdioServerParameters
from smolagents import ToolCollection, Tool
from config import Config

logger = logging.getLogger(__name__)


class MCPServerType(Enum):
    """Available MCP server types"""
    PERPLEXITY = "perplexity"
    WORD = "word"
    BROWSER = "browser"
    FIRECRAWL = "firecrawl"
    TAVILY = "tavily"
    FILESYSTEM = "filesystem"
    CONTAINER = "container"
    POWERBI = "powerbi"


@dataclass
class MCPConnectionConfig:
    """Configuration for MCP server connection"""
    server_type: MCPServerType
    enabled: bool
    connection_params: Dict[str, Any]
    retry_count: int = 3
    timeout: int = 30


class MCPConnectionManager:
    """
    Advanced MCP server connection manager
    Handles multiple MCP servers with connection pooling, retry logic, and health checks
    """
    
    _instance = None
    _connections: Dict[MCPServerType, ToolCollection] = {}
    _connection_lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize connection manager"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._health_status: Dict[MCPServerType, bool] = {}
    
    def _get_connection_config(self, server_type: MCPServerType) -> Optional[MCPConnectionConfig]:
        """Get connection configuration for specific server type"""
        configs = {
            MCPServerType.PERPLEXITY: MCPConnectionConfig(
                server_type=MCPServerType.PERPLEXITY,
                enabled=Config.PERPLEXITY_MCP_ENABLED,
                connection_params={
                    "command": "uvx",
                    "args": ["perplexity-mcp"],
                    "env": {
                        "PERPLEXITY_API_KEY": Config.PERPLEXITY_API_KEY,
                        **os.environ
                    }
                }
            ),
            MCPServerType.WORD: MCPConnectionConfig(
                server_type=MCPServerType.WORD,
                enabled=Config.WORD_MCP_ENABLED,
                connection_params={
                    "command": sys.executable,
                    "args": [
                        os.path.abspath(
                            os.path.join(os.path.dirname(__file__), 
                                       "../../Office-Word-MCP-Server-main/word_mcp_server.py")
                        )
                    ],
                    "env": os.environ
                }
            ),
            MCPServerType.BROWSER: MCPConnectionConfig(
                server_type=MCPServerType.BROWSER,
                enabled=Config.BROWSER_MCP_ENABLED,
                connection_params={
                    "url": Config.BROWSER_MCP_ENDPOINT,
                    "transport": "streamable-http"
                }
            ),
            MCPServerType.FIRECRAWL: MCPConnectionConfig(
                server_type=MCPServerType.FIRECRAWL,
                enabled=Config.FIRECRAWL_MCP_ENABLED,
                connection_params={
                    "command": "uvx",
                    "args": ["firecrawl-mcp"],
                    "env": {
                        "FIRECRAWL_API_KEY": Config.FIRECRAWL_API_KEY,
                        **os.environ
                    }
                }
            ),
            MCPServerType.TAVILY: MCPConnectionConfig(
                server_type=MCPServerType.TAVILY,
                enabled=Config.ENABLE_TAVILY,
                connection_params={
                    "command": "npx",
                    "args": ["-y", "mcp-remote", 
                            f"https://mcp.tavily.com/mcp/?tavilyApiKey={Config.TAVILY_API_KEY}"],
                    "env": os.environ
                }
            ),
            MCPServerType.FILESYSTEM: MCPConnectionConfig(
                server_type=MCPServerType.FILESYSTEM,
                enabled=True,
                connection_params={
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", Config.OUTPUT_DIR],
                    "env": os.environ
                }
            )
        }
        return configs.get(server_type)
    
    async def connect(self, server_type: MCPServerType, force_reconnect: bool = False) -> Optional[ToolCollection]:
        """
        Connect to MCP server with retry logic
        
        Args:
            server_type: Type of MCP server to connect to
            force_reconnect: Force reconnection even if already connected
            
        Returns:
            ToolCollection if successful, None if failed or disabled
        """
        async with self._connection_lock:
            # Return cached connection if exists
            if not force_reconnect and server_type in self._connections:
                return self._connections[server_type]
            
            config = self._get_connection_config(server_type)
            if not config or not config.enabled:
                logger.info(f"MCP server {server_type.value} is disabled")
                return None
            
            # Try to connect with retries
            for attempt in range(config.retry_count):
                try:
                    logger.info(f"Connecting to {server_type.value} MCP server (attempt {attempt + 1}/{config.retry_count})")
                    
                    # Handle different connection types
                    if "url" in config.connection_params:
                        # HTTP-based connection
                        tools = ToolCollection.from_mcp(config.connection_params, trust_remote_code=True)
                    else:
                        # Stdio-based connection
                        server_params = StdioServerParameters(**config.connection_params)
                        tools = ToolCollection.from_mcp(server_params, trust_remote_code=True)
                    
                    self._connections[server_type] = tools
                    self._health_status[server_type] = True
                    logger.info(f"Successfully connected to {server_type.value} MCP server")
                    return tools
                    
                except Exception as e:
                    logger.warning(f"Failed to connect to {server_type.value} (attempt {attempt + 1}): {e}")
                    if attempt == config.retry_count - 1:
                        logger.error(f"Failed to connect to {server_type.value} after {config.retry_count} attempts")
                        self._health_status[server_type] = False
                        return None
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            return None
    
    async def disconnect(self, server_type: MCPServerType):
        """Disconnect from specific MCP server"""
        async with self._connection_lock:
            if server_type in self._connections:
                del self._connections[server_type]
                self._health_status[server_type] = False
                logger.info(f"Disconnected from {server_type.value} MCP server")
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers"""
        async with self._connection_lock:
            self._connections.clear()
            self._health_status.clear()
            logger.info("Disconnected from all MCP servers")
    
    def get_tools(self, server_type: MCPServerType) -> Optional[ToolCollection]:
        """Get tools from connected server (synchronous)"""
        return self._connections.get(server_type)
    
    def is_connected(self, server_type: MCPServerType) -> bool:
        """Check if server is connected and healthy"""
        return server_type in self._connections and self._health_status.get(server_type, False)
    
    def get_health_status(self) -> Dict[str, bool]:
        """Get health status of all servers"""
        return {s.value: status for s, status in self._health_status.items()}
    
    @property
    def health_status(self) -> Dict[str, bool]:
        """Property accessor for health status"""
        return self.get_health_status()


# ============== Helper Functions ==============

async def _get_word_mcp_tools():
    """Get Word MCP tools for document generation"""
    manager = MCPConnectionManager()
    return await manager.connect(MCPServerType.WORD)


async def _get_perplexity_tools():
    """Get Perplexity MCP tools for web search and reasoning"""
    manager = MCPConnectionManager()
    return await manager.connect(MCPServerType.PERPLEXITY)


async def _get_tavily_tools():
    """Get Tavily MCP tools for web search verification"""
    manager = MCPConnectionManager()
    return await manager.connect(MCPServerType.TAVILY)


async def _get_browser_tools():
    """Get Browser MCP tools for web browsing and page scraping"""
    manager = MCPConnectionManager()
    return await manager.connect(MCPServerType.BROWSER)


async def _get_firecrawl_tools():
    """Get Firecrawl MCP tools for advanced web scraping"""
    manager = MCPConnectionManager()
    return await manager.connect(MCPServerType.FIRECRAWL)


async def _get_filesystem_tools(allowed_directories=None):
    """Get Filesystem MCP tools for file operations"""
    manager = MCPConnectionManager()
    return await manager.connect(MCPServerType.FILESYSTEM)


# ============== Public API ==============

async def get_research_tools() -> Dict[str, Optional[ToolCollection]]:
    """
    Get all tools needed for research (web search, browsing, reasoning)
    
    Returns:
        dict: Dictionary with 'perplexity', 'tavily', 'browser', and 'firecrawl' tool collections
    """
    if Config.ENABLE_ASYNC:
        results = await asyncio.gather(
            _get_perplexity_tools(),
            _get_tavily_tools(),
            _get_browser_tools(),
            _get_firecrawl_tools(),
            return_exceptions=True
        )
        perplexity, tavily, browser, firecrawl = results
    else:
        perplexity = await _get_perplexity_tools()
        tavily = await _get_tavily_tools()
        browser = await _get_browser_tools()
        firecrawl = await _get_firecrawl_tools()
    
    return {
        'perplexity': perplexity if not isinstance(perplexity, Exception) else None,
        'tavily': tavily if not isinstance(tavily, Exception) else None,
        'browser': browser if not isinstance(browser, Exception) else None,
        'firecrawl': firecrawl if not isinstance(firecrawl, Exception) else None
    }


async def get_writing_tools() -> Dict[str, Optional[ToolCollection]]:
    """
    Get all tools needed for writing (Word generation, file operations)
    
    Returns:
        dict: Dictionary with 'word' and 'filesystem' tool collections
    """
    if Config.ENABLE_ASYNC:
        results = await asyncio.gather(
            _get_word_mcp_tools(),
            _get_filesystem_tools(),
            return_exceptions=True
        )
        word, filesystem = results
    else:
        word = await _get_word_mcp_tools()
        filesystem = await _get_filesystem_tools()
    
    return {
        'word': word if not isinstance(word, Exception) else None,
        'filesystem': filesystem if not isinstance(filesystem, Exception) else None
    }


async def get_all_tools() -> Dict[str, Optional[ToolCollection]]:
    """
    Get all available tools for complete research workflow
    
    Returns:
        dict: Dictionary with all available tool collections
    """
    if Config.ENABLE_ASYNC:
        research, writing = await asyncio.gather(
            get_research_tools(),
            get_writing_tools()
        )
    else:
        research = await get_research_tools()
        writing = await get_writing_tools()
    
    return {**research, **writing}


def combine_tool_lists(*tool_collections) -> List[Tool]:
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
        elif isinstance(collection, list):
            combined_tools.extend(collection)
    return combined_tools


@asynccontextmanager
async def mcp_context(*server_types: MCPServerType):
    """
    Context manager for MCP connections
    
    Usage:
        async with mcp_context(MCPServerType.PERPLEXITY, MCPServerType.WORD) as tools:
            # Use tools
            pass
    """
    manager = MCPConnectionManager()
    connections = {}
    
    try:
        # Connect to all requested servers
        if Config.ENABLE_ASYNC:
            results = await asyncio.gather(
                *[manager.connect(st) for st in server_types],
                return_exceptions=True
            )
            for st, result in zip(server_types, results):
                if not isinstance(result, Exception) and result is not None:
                    connections[st.value] = result
        else:
            for st in server_types:
                result = await manager.connect(st)
                if result is not None:
                    connections[st.value] = result
        
        yield connections
    finally:
        # Cleanup handled by manager singleton
        pass


def get_connection_manager() -> MCPConnectionManager:
    """Get the global MCP connection manager instance"""
    return MCPConnectionManager()


"""
Specialized Tool Wrappers
Custom tool implementations with enhanced functionality and error handling
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
import logging
from datetime import datetime, timedelta
import hashlib
import json

from smolagents import Tool
from config import Config

logger = logging.getLogger(__name__)


class ToolCache:
    """Simple in-memory cache for tool results"""
    
    def __init__(self, ttl: int = 3600):
        """
        Initialize cache
        
        Args:
            ttl: Time-to-live in seconds (default 1 hour)
        """
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._ttl = ttl
    
    def _make_key(self, tool_name: str, args: tuple, kwargs: dict) -> str:
        """Create cache key from function arguments"""
        key_data = {
            'tool': tool_name,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, tool_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Get cached result if available and not expired"""
        if not Config.ENABLE_CACHING:
            return None
        
        key = self._make_key(tool_name, args, kwargs)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self._ttl):
                logger.debug(f"Cache hit for {tool_name}")
                return result
            else:
                del self._cache[key]
        return None
    
    def set(self, tool_name: str, args: tuple, kwargs: dict, result: Any):
        """Store result in cache"""
        if not Config.ENABLE_CACHING:
            return
        
        key = self._make_key(tool_name, args, kwargs)
        self._cache[key] = (result, datetime.now())
        logger.debug(f"Cached result for {tool_name}")
    
    def clear(self):
        """Clear all cached results"""
        self._cache.clear()


# Global cache instance
_tool_cache = ToolCache()


def cached_tool(func: Callable) -> Callable:
    """Decorator to add caching to tool execution"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Check cache
        cached_result = _tool_cache.get(func.__name__, args, kwargs)
        if cached_result is not None:
            return cached_result
        
        # Execute and cache
        result = await func(*args, **kwargs)
        _tool_cache.set(func.__name__, args, kwargs, result)
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Check cache
        cached_result = _tool_cache.get(func.__name__, args, kwargs)
        if cached_result is not None:
            return cached_result
        
        # Execute and cache
        result = func(*args, **kwargs)
        _tool_cache.set(func.__name__, args, kwargs, result)
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class PerplexitySearchTool(Tool):
    """Enhanced Perplexity search tool with result ranking"""
    
    name = "perplexity_search"
    description = """Search the web using Perplexity AI with real-time information and reasoning.
    Best for: Recent events, fact-checking, comprehensive research topics.
    Input: search query (string)
    Output: Structured search results with sources and relevance scores"""
    
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query"
        },
        "recency": {
            "type": "string", 
            "description": "Recency filter: 'day', 'week', 'month', 'year'",
            "default": "month"
        }
    }
    output_type = "string"
    
    def __init__(self, base_tool: Tool):
        """Initialize with base MCP tool"""
        super().__init__()
        self._base_tool = base_tool
    
    @cached_tool
    async def forward(self, query: str, recency: str = "month") -> str:
        """Execute search with caching"""
        try:
            result = await self._base_tool.forward(query=query, recency=recency)
            return self._format_result(result)
        except Exception as e:
            logger.error(f"Perplexity search failed: {e}")
            return f"Search failed: {str(e)}"
    
    def _format_result(self, raw_result: str) -> str:
        """Format search result for better readability"""
        # Add formatting logic here
        return raw_result


class WebScrapeTool(Tool):
    """Enhanced web scraping tool using Browser MCP or Firecrawl"""
    
    name = "web_scrape"
    description = """Scrape content from a specific URL with structured extraction.
    Best for: Extracting specific information from web pages, downloading documents.
    Input: url (string), selectors (optional list)
    Output: Structured page content"""
    
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL to scrape"
        },
        "extract_type": {
            "type": "string",
            "description": "Type of extraction: 'text', 'markdown', 'html', 'structured'",
            "default": "text"
        }
    }
    output_type = "string"
    
    def __init__(self, browser_tool: Optional[Tool] = None, firecrawl_tool: Optional[Tool] = None):
        """Initialize with browser or firecrawl tool"""
        super().__init__()
        self._browser_tool = browser_tool
        self._firecrawl_tool = firecrawl_tool
    
    @cached_tool
    async def forward(self, url: str, extract_type: str = "text") -> str:
        """Execute scraping with fallback options"""
        # Try Firecrawl first (better quality)
        if self._firecrawl_tool:
            try:
                return await self._firecrawl_tool.forward(url=url)
            except Exception as e:
                logger.warning(f"Firecrawl failed, falling back to browser: {e}")
        
        # Fallback to browser tool
        if self._browser_tool:
            try:
                return await self._browser_tool.forward(url=url)
            except Exception as e:
                logger.error(f"Browser scraping failed: {e}")
                return f"Scraping failed: {str(e)}"
        
        return "No scraping tools available"


class DocumentWriterTool(Tool):
    """Enhanced Word document writer with templating"""
    
    name = "write_document"
    description = """Create professional Word documents with formatting.
    Best for: Research papers, reports, formatted documents.
    Input: content (string or dict), filename (string), style (string)
    Output: File path of created document"""
    
    inputs = {
        "content": {
            "type": "any",
            "description": "Content to write (text or structured data)"
        },
        "filename": {
            "type": "string",
            "description": "Output filename"
        },
        "style": {
            "type": "string",
            "description": "Document style: 'academic', 'technical', 'general'",
            "default": "academic"
        }
    }
    output_type = "string"
    
    def __init__(self, word_tool: Tool):
        """Initialize with Word MCP tool"""
        super().__init__()
        self._word_tool = word_tool
    
    async def forward(self, content: Any, filename: str, style: str = "academic") -> str:
        """Create formatted document"""
        try:
            # Apply style configuration
            style_config = Config.WRITING_STYLES.get(style, Config.WRITING_STYLES["academic"])
            
            # Format content based on style
            formatted_content = self._apply_style(content, style_config)
            
            # Create document
            result = await self._word_tool.forward(
                content=formatted_content,
                filename=filename
            )
            return result
        except Exception as e:
            logger.error(f"Document creation failed: {e}")
            return f"Document creation failed: {str(e)}"
    
    def _apply_style(self, content: Any, style_config: dict) -> Any:
        """Apply styling to content"""
        # Add styling logic here
        return content


class AsyncToolExecutor:
    """Execute multiple tools concurrently with rate limiting"""
    
    def __init__(self, max_concurrent: int = None):
        """
        Initialize executor
        
        Args:
            max_concurrent: Maximum concurrent executions (defaults to Config.MAX_CONCURRENT_TASKS)
        """
        self.max_concurrent = max_concurrent or Config.MAX_CONCURRENT_TASKS
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def execute_one(self, tool: Tool, *args, **kwargs) -> Any:
        """Execute single tool with rate limiting"""
        async with self._semaphore:
            try:
                logger.debug(f"Executing tool: {tool.name}")
                result = await tool.forward(*args, **kwargs)
                logger.debug(f"Tool {tool.name} completed successfully")
                return {"success": True, "tool": tool.name, "result": result}
            except Exception as e:
                logger.error(f"Tool {tool.name} failed: {e}")
                return {"success": False, "tool": tool.name, "error": str(e)}
    
    async def execute_batch(self, tool_calls: List[tuple[Tool, tuple, dict]]) -> List[Dict]:
        """
        Execute multiple tools concurrently
        
        Args:
            tool_calls: List of (tool, args, kwargs) tuples
            
        Returns:
            List of execution results
        """
        tasks = [
            self.execute_one(tool, *args, **kwargs)
            for tool, args, kwargs in tool_calls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r if not isinstance(r, Exception) else {"success": False, "error": str(r)} 
                for r in results]


def create_tool_wrapper(base_tool: Tool, tool_type: str) -> Tool:
    """
    Factory function to create appropriate wrapper for base tool
    
    Args:
        base_tool: Base MCP tool
        tool_type: Type of wrapper to create
        
    Returns:
        Wrapped tool with enhanced functionality
    """
    wrappers = {
        "perplexity_search": lambda t: PerplexitySearchTool(t),
        "web_scrape": lambda t: WebScrapeTool(browser_tool=t),
        "write_document": lambda t: DocumentWriterTool(t)
    }
    
    wrapper_fn = wrappers.get(tool_type)
    return wrapper_fn(base_tool) if wrapper_fn else base_tool

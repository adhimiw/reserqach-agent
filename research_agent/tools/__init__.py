"""
Tools Package - Advanced MCP Integration
Provides unified access to all research and writing tools
"""

from .mcp_tools import (
    MCPConnectionManager,
    MCPServerType,
    get_research_tools,
    get_writing_tools,
    get_all_tools,
    combine_tool_lists,
    mcp_context,
    get_connection_manager
)

from .tool_wrappers import (
    PerplexitySearchTool,
    WebScrapeTool,
    DocumentWriterTool,
    AsyncToolExecutor,
    create_tool_wrapper,
    ToolCache
)

from .dspy_signatures import (
    WebSearchSignature,
    ContentExtractionSignature,
    ResearchSynthesisSignature,
    FactCheckSignature,
    DocumentStructureSignature,
    QueryRefinementSignature,
    CitationGenerationSignature,
    OptimizedSearchModule,
    ResearchSynthesisModule,
    FactCheckModule,
    AdaptiveResearchModule,
    configure_dspy,
    create_optimized_modules
)

__all__ = [
    # MCP Tools
    "MCPConnectionManager",
    "MCPServerType",
    "get_research_tools",
    "get_writing_tools",
    "get_all_tools",
    "combine_tool_lists",
    "mcp_context",
    "get_connection_manager",
    
    # Tool Wrappers
    "PerplexitySearchTool",
    "WebScrapeTool",
    "DocumentWriterTool",
    "AsyncToolExecutor",
    "create_tool_wrapper",
    "ToolCache",
    
    # DSPy Signatures
    "WebSearchSignature",
    "ContentExtractionSignature",
    "ResearchSynthesisSignature",
    "FactCheckSignature",
    "DocumentStructureSignature",
    "QueryRefinementSignature",
    "CitationGenerationSignature",
    "OptimizedSearchModule",
    "ResearchSynthesisModule",
    "FactCheckModule",
    "AdaptiveResearchModule",
    "configure_dspy",
    "create_optimized_modules"
]


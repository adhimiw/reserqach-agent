"""
Researcher Agent - Handles information gathering and web research
"""

from smolagents import CodeAgent
from config import Config

def create_researcher_agent(tools=None, model=None):
    """
    Creates the Researcher Agent responsible for gathering information.
    
    Args:
        tools: List of research tools (web search, browser, etc.)
        model: LLM model instance (defaults to Perplexity)
    
    Returns:
        CodeAgent configured for research tasks
    """
    if model is None:
        model = Config.get_model()
    
    if tools is None:
        tools = []
    
    return CodeAgent(
        tools=tools,
        model=model,
        name="researcher",
        description="Gathers information through web search (Tavily), Perplexity queries, and browser navigation",
        additional_authorized_imports=["json", "datetime", "time", "re"]
    )


def create_recursive_researcher(tools=None, model=None, max_depth=3):
    """
    Creates a recursive researcher agent for multi-level research.
    
    Args:
        tools: List of research tools
        model: LLM model instance
        max_depth: Maximum recursion depth for research
    
    Returns:
        CodeAgent with recursive research capabilities
    """
    if model is None:
        model = Config.get_model()
    
    if tools is None:
        tools = []
    
    return CodeAgent(
        tools=tools,
        model=model,
        name="recursive_researcher",
        description="Recursively researches topics by breaking them down into sub-topics and gathering information at multiple levels",
        additional_authorized_imports=["json", "datetime", "time", "re", "functools"]
    )

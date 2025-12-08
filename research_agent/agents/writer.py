"""
Writer Agent - Handles research paper generation and formatting
"""

from smolagents import CodeAgent
from config import Config
import os

def create_writer_agent(tools=None, model=None):
    """
    Creates the Writer Agent responsible for generating research papers.
    
    Args:
        tools: List of writing tools (Word MCP, etc.)
        model: LLM model instance (defaults to Perplexity)
    
    Returns:
        CodeAgent configured for writing tasks
    """
    if model is None:
        model = Config.get_model()
    
    if tools is None:
        tools = []
    
    return CodeAgent(
        tools=tools,
        model=model,
        name="writer",
        description="Transforms research findings into well-structured academic papers using Word MCP server",
        additional_authorized_imports=["json", "datetime", "docx"]
    )


def create_verifier_agent(tools=None, model=None):
    """
    Creates the Verifier Agent responsible for fact-checking and validation.
    
    Args:
        tools: List of verification tools (web search, browser, etc.)
        model: LLM model instance (defaults to Perplexity)
    
    Returns:
        CodeAgent configured for verification tasks
    """
    if model is None:
        model = Config.get_model()
    
    if tools is None:
        tools = []
    
    return CodeAgent(
        tools=tools,
        model=model,
        name="verifier",
        description="Verifies research claims and citations through web search and fact-checking",
        additional_authorized_imports=["json", "datetime", "re"]
    )


def create_planner_agent(model=None):
    """
    Creates the Planner Agent responsible for research planning.
    
    Args:
        model: LLM model instance (defaults to Perplexity)
    
    Returns:
        CodeAgent configured for planning tasks
    """
    if model is None:
        model = Config.get_model()
    
    return CodeAgent(
        model=model,
        name="planner",
        description="Plans research structure, outlines, and research questions",
        additional_authorized_imports=["json", "datetime"]
    )


def create_orchestrator_agent(planner, researcher, verifier, writer, model=None):
    """
    Creates the Orchestrator Agent that coordinates all other agents.
    
    Args:
        planner: Planner agent instance
        researcher: Researcher agent instance
        verifier: Verifier agent instance
        writer: Writer agent instance
        model: LLM model instance (defaults to Perplexity)
    
    Returns:
        CodeAgent that orchestrates the research workflow
    """
    if model is None:
        model = Config.get_model()
    
    return CodeAgent(
        model=model,
        name="orchestrator",
        description="Orchestrates the entire research pipeline: planning -> research -> verification -> writing",
        additional_authorized_imports=["json", "datetime"]
    )

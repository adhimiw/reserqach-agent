"""
Analyzer Agent - Main analysis orchestrator
Coordinates data loading, hypothesis generation, statistical testing, and modeling
"""

from smolagents import CodeAgent, ToolCallingAgent
from config import Config
from tools import get_all_tools, combine_tool_lists


def create_analyzer_agent(tools=None, model=None):
    """
    Creates the main Analyzer Agent that orchestrates data analysis
    
    Args:
        tools: List of analysis tools (Pandas, Jupyter, etc.)
        model: LLM model instance
    
    Returns:
        CodeAgent configured for data analysis tasks
    """
    if model is None:
        model = Config.get_model()
    
    if tools is None:
        tools = combine_tool_lists(get_all_tools())
    
    return CodeAgent(
        tools=tools,
        model=model,
        name="analyzer",
        description="Main analysis orchestrator that coordinates data loading, hypothesis generation, statistical testing, and predictive modeling",
        additional_authorized_imports=[
            "pandas", "numpy", "json", "os", "sys", 
            "datetime", "typing", "pathlib"
        ]
    )


def create_self_healing_agent(model=None):
    """
    Creates an agent that detects and fixes errors automatically
    
    Args:
        model: LLM model instance
    
    Returns:
        CodeAgent configured for error detection and recovery
    """
    if model is None:
        model = Config.get_model()
    
    return CodeAgent(
        tools=[],
        model=model,
        name="self_healer",
        description="Detects errors in analysis, analyzes root causes, and generates fixes automatically. Retries with alternative approaches when failures occur.",
        additional_authorized_imports=[
            "traceback", "sys", "re", "json", "logging"
        ],
        max_loops=3  # Allow multiple retry attempts
    )


def create_hypothesis_generator_agent(tools=None, model=None):
    """
    Creates agent specialized in generating testable hypotheses
    
    Args:
        tools: List of tools
        model: LLM model instance
    
    Returns:
        CodeAgent configured for hypothesis generation
    """
    if model is None:
        model = Config.get_model()
    
    if tools is None:
        from tools import get_data_processing_tools
        tools = combine_tool_lists(get_data_processing_tools())
    
    return CodeAgent(
        tools=tools,
        model=model,
        name="hypothesis_generator",
        description="Analyzes dataset characteristics and generates testable hypotheses about correlations, distributions, trends, and patterns",
        additional_authorized_imports=[
            "pandas", "numpy", "scipy", "json", "typing"
        ]
    )


def create_statistical_tester_agent(tools=None, model=None):
    """
    Creates agent for performing statistical tests
    
    Args:
        tools: List of tools
        model: LLM model instance
    
    Returns:
        CodeAgent configured for statistical testing
    """
    if model is None:
        model = Config.get_model()
    
    if tools is None:
        from tools import get_data_processing_tools
        tools = combine_tool_lists(get_data_processing_tools())
    
    return CodeAgent(
        tools=tools,
        model=model,
        name="statistical_tester",
        description="Performs comprehensive statistical tests including correlation analysis, normality tests, outlier detection, and group comparisons",
        additional_authorized_imports=[
            "pandas", "numpy", "scipy.stats", "json", "typing"
        ]
    )


def create_model_builder_agent(tools=None, model=None):
    """
    Creates agent for building predictive models
    
    Args:
        tools: List of tools
        model: LLM model instance
    
    Returns:
        CodeAgent configured for modeling
    """
    if model is None:
        model = Config.get_model()
    
    if tools is None:
        from tools import get_data_processing_tools
        tools = combine_tool_lists(get_data_processing_tools())
    
    return CodeAgent(
        tools=tools,
        model=model,
        name="model_builder",
        description="Builds and evaluates predictive models including linear regression, random forests, and gradient boosting for classification and regression tasks",
        additional_authorized_imports=[
            "pandas", "numpy", "sklearn", "json", "typing", "joblib"
        ]
    )


def create_visualizer_agent(tools=None, model=None):
    """
    Creates agent for generating visualizations
    
    Args:
        tools: List of tools
        model: LLM model instance
    
    Returns:
        CodeAgent configured for visualization
    """
    if model is None:
        model = Config.get_model()
    
    if tools is None:
        from tools import get_data_processing_tools
        tools = combine_tool_lists(get_data_processing_tools())
    
    return CodeAgent(
        tools=tools,
        model=model,
        name="visualizer",
        description="Creates publication-quality visualizations including scatter plots, line charts, histograms, heatmaps, and interactive plots using matplotlib, seaborn, and plotly",
        additional_authorized_imports=[
            "matplotlib", "seaborn", "plotly", "pandas", "numpy", "json"
        ]
    )


def create_research_context_agent(model=None, tools=None):
    """
    Creates agent for real-time research using Perplexity
    
    Args:
        model: LLM model instance (should be Perplexity)
        tools: List of research tools
    
    Returns:
        ToolCallingAgent configured for web research
    """
    if model is None:
        model = Config.get_perplexity_model()
    
    if tools is None:
        from tools import get_research_tools
        tools = combine_tool_lists(get_research_tools())
    
    return ToolCallingAgent(
        tools=tools,
        model=model,
        name="research_context",
        description="Researches real-world context and explanations for findings using web search and current information"
    )

from smolagents import CodeAgent, LiteLLMModel
from config import Config

def create_researcher_agent(tools, model=None):
    """
    Creates the Researcher Agent responsible for gathering information.
    """
    if model is None:
        # Use Perplexity via LiteLLM
        model = LiteLLMModel(
            model_id=Config.LLM_MODEL_ID, 
            api_key=Config.PERPLEXITY_API_KEY,
            api_base=Config.LLM_API_BASE,
            drop_params=True
        )

    return CodeAgent(
        tools=tools,
        model=model,
        name="researcher",
        description="An agent that executes research plans by searching the web (Tavily) and browsing pages (Browser MCP).",
        additional_authorized_imports=["json", "datetime"]
    )

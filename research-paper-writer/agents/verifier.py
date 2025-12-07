from smolagents import CodeAgent, LiteLLMModel
from config import Config

def create_verifier_agent(tools, model=None):
    """
    Creates the Verifier Agent responsible for checking claims against ground truth.
    """
    if model is None:
        model = LiteLLMModel(
            model_id=Config.LLM_MODEL_ID, 
            api_key=Config.PERPLEXITY_API_KEY,
            api_base=Config.LLM_API_BASE,
            drop_params=True
        )

    return CodeAgent(
        tools=tools,
        model=model,
        name="verifier",
        description="An agent that verifies claims by visiting URLs and reading page content directly.",
        additional_authorized_imports=["json", "bs4", "requests"] # bs4 might be useful for parsing HTML if returned
    )

from smolagents import CodeAgent, LiteLLMModel
from config import Config

def create_planner_agent(model=None):
    """
    Creates the Planner Agent responsible for generating research plans and outlines.
    Based on GPT Researcher architecture: "The planner generates research questions".
    """
    if model is None:
        model = LiteLLMModel(
            model_id=Config.LLM_MODEL_ID, 
            api_key=Config.PERPLEXITY_API_KEY,
            api_base=Config.LLM_API_BASE,
            drop_params=True
        )

    return CodeAgent(
        tools=[], # Planner relies on LLM knowledge to structure the plan
        model=model,
        name="planner",
        description="An agent that generates comprehensive research plans, outlines, and specific research questions.",
        additional_authorized_imports=["json"]
    )

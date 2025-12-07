from smolagents import CodeAgent, LiteLLMModel
from config import Config

def create_orchestrator_agent(planner, researcher, verifier, writer, model=None):
    """
    Creates the Lead Orchestrator Agent.
    """
    if model is None:
        model = LiteLLMModel(
            model_id=Config.LLM_MODEL_ID, 
            api_key=Config.PERPLEXITY_API_KEY,
            api_base=Config.LLM_API_BASE,
            drop_params=True
        )

    # In smolagents 1.23.0, we pass the agents directly to managed_agents
    # They must have 'name' and 'description' attributes set.

    return CodeAgent(
        tools=[], # The orchestrator mainly delegates
        managed_agents=[planner, researcher, verifier, writer],
        model=model,
        name="orchestrator",
        description="The Lead Orchestrator that manages the research paper writing process."
    )

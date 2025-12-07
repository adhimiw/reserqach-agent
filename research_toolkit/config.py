
import os
from smolagents import LiteLLMModel
import litellm

# Configure LiteLLM to drop unsupported parameters
litellm.drop_params = True

def get_model(model_id="perplexity/sonar-pro"):
    """
    Returns a LiteLLMModel instance, strictly using Perplexity.
    """
    # Hardcoded from user context for convenience
    pplx_key = os.environ.get("PERPLEXITY_API_KEY") or "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"
    
    if pplx_key:
        os.environ["PERPLEXITY_API_KEY"] = pplx_key
        print(f"Using Perplexity Model ({model_id}).")
        return LiteLLMModel(model_id=model_id, api_key=pplx_key)

    print("Warning: No Perplexity API key found. Agent may fail.")
    return LiteLLMModel(model_id=model_id)

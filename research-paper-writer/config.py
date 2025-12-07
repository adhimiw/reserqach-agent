import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Perplexity Configuration
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "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")
    PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
    
    # Main LLM Configuration (Using Perplexity as the Brain)
    # LiteLLM format for Perplexity: "perplexity/sonar-pro"
    LLM_MODEL_ID = "perplexity/sonar-pro"
    LLM_API_BASE = "https://api.perplexity.ai"
    
    # Tavily Configuration
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")

    # Word MCP Configuration
    # Assuming Word MCP is running locally or via stdio
    
    # Chrome MCP Configuration
    # Assuming Chrome MCP is running locally or via stdio

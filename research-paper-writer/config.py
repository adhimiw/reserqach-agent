import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Perplexity Configuration
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "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")
    PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
    
    # Mistral Configuration
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "HKK5Q0lja9HBOwIEXt82sncuQb3RksPW")
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    
    # Cohere Configuration
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "39sIDzEasPItQPgAReUIuiwLoq9tKiOcgQYseK6E")
    COHERE_MODEL = os.getenv("COHERE_MODEL", "command")
    
    # Main LLM Configuration (Using Perplexity as the Brain)
    # LiteLLM format for Perplexity: "perplexity/sonar-pro"
    LLM_MODEL_ID = "perplexity/sonar-pro"
    LLM_API_BASE = "https://api.perplexity.ai"
    
    # Tavily Configuration
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")
    
    # Token Optimization Configuration
    ENABLE_PROMPT_COMPRESSION = os.getenv("ENABLE_PROMPT_COMPRESSION", "true").lower() == "true"
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "2000"))
    COMPRESSION_RATIO = float(os.getenv("COMPRESSION_RATIO", "0.4"))

    # Word MCP Configuration
    # Assuming Word MCP is running locally or via stdio
    
    # Chrome MCP Configuration
    # Assuming Chrome MCP is running locally or via stdio

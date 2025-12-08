"""
Unified Research Agent Configuration
Consolidates settings from research-paper-writer and research_toolkit
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Master configuration for research agent system"""
    
    # ============== LLM Models ==============
    # Primary LLM - Perplexity (with reasoning capabilities)
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "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")
    PERPLEXITY_MODEL = "sonar-pro"
    LLM_MODEL_ID = "perplexity/sonar-pro"
    LLM_API_BASE = "https://api.perplexity.ai"
    
    # Fallback models
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "HKK5Q0lja9HBOwIEXt82sncuQb3RksPW")
    MISTRAL_MODEL = "mistral-large-latest"
    
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "39sIDzEasPItQPgAReUIuiwLoq9tKiOcgQYseK6E")
    COHERE_MODEL = "command"
    
    # ============== Search & Research Tools ==============
    # Tavily API for web search
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")
    
    # ============== MCP Server Configurations ==============
    # Word MCP Server
    WORD_MCP_STDIO = True  # Use stdio for Word MCP
    WORD_MCP_PORT = 12307  # If not using stdio
    
    # Browser/Chrome MCP Server
    BROWSER_MCP_STDIO = False  # Can be local Selenium instance
    BROWSER_MCP_ENDPOINT = "http://localhost:8765"  # Local browser MCP endpoint
    
    # ============== Performance & Optimization ==============
    # Prompt compression to reduce token usage
    ENABLE_PROMPT_COMPRESSION = os.getenv("ENABLE_PROMPT_COMPRESSION", "true").lower() == "true"
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "2000"))
    COMPRESSION_RATIO = float(os.getenv("COMPRESSION_RATIO", "0.4"))
    
    # ============== Output Configuration ==============
    OUTPUT_DIR = "output"
    PAPERS_DIR = "output/papers"
    CACHE_DIR = "output/cache"
    
    # ============== Workflow Settings ==============
    # Recursive search depth for complex topics
    MAX_RESEARCH_DEPTH = 3
    MAX_ITERATIONS = 5
    
    # Research stages
    ENABLE_PLANNING = True
    ENABLE_RESEARCH = True
    ENABLE_VERIFICATION = True
    ENABLE_WRITING = True
    
    # ============== Logging ==============
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = "logs"
    
    @classmethod
    def get_model(cls, model_id=None):
        """Factory method to get configured LLM model"""
        from smolagents import LiteLLMModel
        import litellm
        
        litellm.drop_params = True
        
        model_id = model_id or cls.LLM_MODEL_ID
        api_key = cls.PERPLEXITY_API_KEY if "perplexity" in model_id else None
        
        return LiteLLMModel(model_id=model_id, api_key=api_key)
    
    @classmethod
    def ensure_output_dirs(cls):
        """Create necessary output directories"""
        for directory in [cls.OUTPUT_DIR, cls.PAPERS_DIR, cls.CACHE_DIR, cls.LOG_DIR]:
            os.makedirs(directory, exist_ok=True)

"""
Advanced Research Agent Configuration
Multi-model support with DSPy integration, MCP servers, and smolagents framework
"""

import os
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class ModelProvider(Enum):
    """Supported LLM providers"""
    PERPLEXITY = "perplexity"
    MISTRAL = "mistral"
    COHERE = "cohere"
    SMITHERY = "smithery"


class AgentRole(Enum):
    """Specialized agent roles"""
    RESEARCHER = "researcher"
    WRITER = "writer"
    FACT_CHECKER = "fact_checker"
    EDITOR = "editor"
    ORCHESTRATOR = "orchestrator"


@dataclass
class ModelConfig:
    """Configuration for individual model"""
    provider: ModelProvider
    model_id: str
    api_key: str
    api_base: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    supports_streaming: bool = True


class ModelFactory:
    """Factory for creating and managing LLM instances"""
    
    _models_cache: Dict[str, Any] = {}
    
    @staticmethod
    def get_model_config(provider: ModelProvider) -> ModelConfig:
        """Get configuration for specific provider"""
        configs = {
            ModelProvider.PERPLEXITY: ModelConfig(
                provider=ModelProvider.PERPLEXITY,
                model_id="perplexity/sonar-pro",
                api_key=os.getenv("PERPLEXITY_API_KEY", "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"),
                api_base="https://api.perplexity.ai",
                max_tokens=8192,
                temperature=0.7
            ),
            ModelProvider.MISTRAL: ModelConfig(
                provider=ModelProvider.MISTRAL,
                model_id="mistral/mistral-large-latest",
                api_key=os.getenv("MISTRAL_API_KEY", "HKK5Q0lja9HBOwIEXt82sncuQb3RksPW"),
                max_tokens=8192,
                temperature=0.7
            ),
            ModelProvider.COHERE: ModelConfig(
                provider=ModelProvider.COHERE,
                model_id="cohere/command-r-plus",
                api_key=os.getenv("COHERE_API_KEY", "39sIDzEasPItQPgAReUIuiwLoq9tKiOcgQYseK6E"),
                max_tokens=4096,
                temperature=0.7
            ),
            ModelProvider.SMITHERY: ModelConfig(
                provider=ModelProvider.SMITHERY,
                model_id="smithery/anthropic-claude-3-7-sonnet-20250219",
                api_key=os.getenv("SMITHERY_API_KEY", "sk-Ft1ZPf2OhtzXwJiJJPe1vEyPrvGj71BDpV65eTPvLyGS2lZt"),
                api_base="https://api.smith.langchain.com",
                max_tokens=8192,
                temperature=0.7
            )
        }
        return configs.get(provider)
    
    @classmethod
    def create_model(cls, provider: ModelProvider, cache: bool = True):
        """Create LiteLLM model instance with smolagents"""
        cache_key = f"{provider.value}"
        
        if cache and cache_key in cls._models_cache:
            return cls._models_cache[cache_key]
        
        from smolagents import LiteLLMModel
        import litellm
        
        # Configure litellm
        litellm.drop_params = True
        litellm.set_verbose = False
        
        config = cls.get_model_config(provider)
        if not config:
            raise ValueError(f"Unknown provider: {provider}")
        
        model = LiteLLMModel(
            model_id=config.model_id,
            api_key=config.api_key,
            api_base=config.api_base,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        
        if cache:
            cls._models_cache[cache_key] = model
        
        return model
    
    @classmethod
    def get_model_for_role(cls, role: AgentRole) -> Any:
        """Get optimal model for specific agent role"""
        role_to_provider = {
            AgentRole.RESEARCHER: ModelProvider.PERPLEXITY,  # Best for search & reasoning
            AgentRole.WRITER: ModelProvider.SMITHERY,  # Claude for writing quality
            AgentRole.FACT_CHECKER: ModelProvider.PERPLEXITY,  # Real-time verification
            AgentRole.EDITOR: ModelProvider.MISTRAL,  # Good at refinement
            AgentRole.ORCHESTRATOR: ModelProvider.SMITHERY  # Complex coordination
        }
        
        provider = role_to_provider.get(role, ModelProvider.PERPLEXITY)
        return cls.create_model(provider)


class Config:
    """Master configuration for research agent system"""
    
    # ============== LLM Models ==============
    DEFAULT_PROVIDER = ModelProvider.PERPLEXITY
    
    # API Keys
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "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")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "HKK5Q0lja9HBOwIEXt82sncuQb3RksPW")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "39sIDzEasPItQPgAReUIuiwLoq9tKiOcgQYseK6E")
    SMITHERY_API_KEY = os.getenv("SMITHERY_API_KEY", "sk-Ft1ZPf2OhtzXwJiJJPe1vEyPrvGj71BDpV65eTPvLyGS2lZt")
    
    # ============== MCP Server Configurations ==============
    # Perplexity MCP
    PERPLEXITY_MCP_ENABLED = os.getenv("PERPLEXITY_MCP_ENABLED", "true").lower() == "true"
    
    # Word MCP Server
    WORD_MCP_ENABLED = os.getenv("WORD_MCP_ENABLED", "true").lower() == "true"
    WORD_MCP_STDIO = True
    WORD_MCP_PORT = 12307
    
    # Browser/Chrome MCP Server
    BROWSER_MCP_ENABLED = os.getenv("BROWSER_MCP_ENABLED", "true").lower() == "true"
    BROWSER_MCP_STDIO = False
    BROWSER_MCP_ENDPOINT = "http://localhost:8765"
    
    # Firecrawl MCP
    FIRECRAWL_MCP_ENABLED = os.getenv("FIRECRAWL_MCP_ENABLED", "false").lower() == "true"
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "fc-5af9e28ea4d74654adac7c60f89e37cd")
    
    # Container MCP (Docker)
    CONTAINER_MCP_ENABLED = os.getenv("CONTAINER_MCP_ENABLED", "false").lower() == "true"
    
    # PowerBI MCP
    POWERBI_MCP_ENABLED = os.getenv("POWERBI_MCP_ENABLED", "false").lower() == "true"
    
    # ============== DSPy Configuration ==============
    DSPY_ENABLED = os.getenv("DSPY_ENABLED", "true").lower() == "true"
    DSPY_CACHE_DIR = "output/dspy_cache"
    DSPY_TRAINING_EXAMPLES = 20
    DSPY_OPTIMIZER = "BootstrapFewShot"  # or "MIPRO", "MIPROv2"
    
    # ============== Search & Research Tools ==============
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")
    ENABLE_TAVILY = os.getenv("ENABLE_TAVILY", "true").lower() == "true"
    
    # ============== Agent Orchestration ==============
    ENABLE_MULTI_AGENT = os.getenv("ENABLE_MULTI_AGENT", "true").lower() == "true"
    MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
    AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "300"))  # seconds
    
    # ============== Sandbox Configuration ==============
    ENABLE_E2B_SANDBOX = os.getenv("ENABLE_E2B_SANDBOX", "false").lower() == "true"
    E2B_API_KEY = os.getenv("E2B_API_KEY", "")
    SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "600"))
    
    # ============== Performance & Optimization ==============
    ENABLE_PROMPT_COMPRESSION = os.getenv("ENABLE_PROMPT_COMPRESSION", "true").lower() == "true"
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "4000"))
    COMPRESSION_RATIO = float(os.getenv("COMPRESSION_RATIO", "0.4"))
    
    # Async configuration
    ENABLE_ASYNC = os.getenv("ENABLE_ASYNC", "true").lower() == "true"
    MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))
    
    # ============== Output Configuration ==============
    OUTPUT_DIR = "output"
    PAPERS_DIR = "output/papers"
    CACHE_DIR = "output/cache"
    ARTIFACTS_DIR = "output/artifacts"
    
    # ============== Workflow Settings ==============
    MAX_RESEARCH_DEPTH = int(os.getenv("MAX_RESEARCH_DEPTH", "3"))
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
    
    # Research stages
    ENABLE_PLANNING = os.getenv("ENABLE_PLANNING", "true").lower() == "true"
    ENABLE_RESEARCH = os.getenv("ENABLE_RESEARCH", "true").lower() == "true"
    ENABLE_VERIFICATION = os.getenv("ENABLE_VERIFICATION", "true").lower() == "true"
    ENABLE_WRITING = os.getenv("ENABLE_WRITING", "true").lower() == "true"
    
    # Writing style configuration
    WRITING_STYLES = {
        "academic": {
            "formality": "high",
            "citation_style": "APA",
            "voice": "third_person",
            "tone": "objective"
        },
        "technical": {
            "formality": "medium-high",
            "detail_level": "comprehensive",
            "voice": "third_person",
            "tone": "instructive"
        },
        "general": {
            "formality": "medium",
            "voice": "third_person",
            "tone": "informative"
        }
    }
    DEFAULT_WRITING_STYLE = "academic"
    
    # ============== Logging ==============
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = "logs"
    ENABLE_STRUCTURED_LOGGING = os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"
    
    # ============== Factory Methods ==============
    @classmethod
    def get_model(cls, provider: Optional[Union[ModelProvider, AgentRole]] = None, role: Optional[AgentRole] = None):
        """Get LLM model instance - either by provider or by role"""
        # Handle case where AgentRole is passed as provider
        if isinstance(provider, AgentRole):
            return ModelFactory.get_model_for_role(provider)
        
        if role:
            return ModelFactory.get_model_for_role(role)
        
        provider = provider or cls.DEFAULT_PROVIDER
        return ModelFactory.create_model(provider)
    
    @classmethod
    def get_model_config(cls, provider: ModelProvider) -> ModelConfig:
        """Get configuration for specific provider"""
        return ModelFactory.get_model_config(provider)
    
    @classmethod
    def ensure_output_dirs(cls):
        """Create necessary output directories"""
        dirs = [
            cls.OUTPUT_DIR, 
            cls.PAPERS_DIR, 
            cls.CACHE_DIR, 
            cls.LOG_DIR,
            cls.ARTIFACTS_DIR,
            cls.DSPY_CACHE_DIR
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_enabled_mcp_servers(cls) -> list[str]:
        """Get list of enabled MCP servers"""
        servers = []
        if cls.PERPLEXITY_MCP_ENABLED:
            servers.append("perplexity")
        if cls.WORD_MCP_ENABLED:
            servers.append("word")
        if cls.BROWSER_MCP_ENABLED:
            servers.append("browser")
        if cls.FIRECRAWL_MCP_ENABLED:
            servers.append("firecrawl")
        if cls.CONTAINER_MCP_ENABLED:
            servers.append("container")
        if cls.POWERBI_MCP_ENABLED:
            servers.append("powerbi")
        return servers

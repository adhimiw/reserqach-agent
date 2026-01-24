"""
Enhanced Configuration with LLM Council Support and Logging
"""

import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import json
from pathlib import Path

load_dotenv()


class Config:
    """Master configuration for autonomous data science system with LLM Council support"""
    
    # ============== LLM Models ==============
    # Primary LLM - Mistral Agent API for code generation
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_MODEL = "mistral-large-latest"
    MISTRAL_API_BASE = "https://api.mistral.ai/v1"
    MISTRAL_API_ENDPOINTS = {
        "chat": f"{MISTRAL_API_BASE}/chat/completions",
        "models": f"{MISTRAL_API_BASE}/models",
        "usage": f"{MISTRAL_API_BASE}/usage"
    }
    
    # Perplexity API for real-time research and reasoning
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
    PERPLEXITY_MODEL = "sonar-pro"
    PERPLEXITY_API_BASE = "https://api.perplexity.ai"
    PERPLEXITY_API_ENDPOINTS = {
        "chat": f"{PERPLEXITY_API_BASE}/chat/completions",
        "models": f"{PERPLEXITY_API_BASE}/models",
        "usage": f"{PERPLEXITY_API_BASE}/usage"
    }
    
    # Google Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.5-flash-exp"
    GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    GEMINI_API_ENDPOINTS = {
        "generate": f"{GEMINI_API_BASE}/models/gemini-2.5-flash-exp:generateContent",
        "chat": f"{GEMINI_API_BASE}/models/gemini-2.5-flash-exp:generateContent"
    }
    
    # OpenRouter API (for LLM Council)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
    OPENROUTER_API_ENDPOINTS = {
        "chat": f"{OPENROUTER_API_BASE}/chat/completions",
        "models": f"{OPENROUTER_API_BASE}/models"
    }
    
    # Fallback models
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    COHERE_MODEL = "command"
    COHERE_API_BASE = "https://api.cohere.ai/v1"
    
    # ============== LLM Council Configuration ==============
    # Council members available
    COUNCIL_MODELS = [
        {
            "name": "GPT-4",
            "id": "openai/gpt-4",
            "provider": "openrouter",
            "model": "openai/gpt-4",
            "capabilities": ["reasoning", "code", "analysis"],
            "max_tokens": 128000,
            "cost_per_1k_tokens": 0.03
        },
        {
            "name": "Claude Sonnet 4",
            "id": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "capabilities": ["reasoning", "analysis", "insight"],
            "max_tokens": 200000,
            "cost_per_1k_tokens": 0.015
        },
        {
            "name": "Gemini Pro",
            "id": "google/gemini-3-pro-preview",
            "provider": "openrouter",
            "model": "google/gemini-3-pro-preview",
            "capabilities": ["reasoning", "analysis", "coding"],
            "max_tokens": 120000,
            "cost_per_1k_tokens": 0.002
        },
        {
            "name": "Grok 4",
            "id": "x-ai/grok-4",
            "provider": "openrouter",
            "model": "x-ai/grok-4",
            "capabilities": ["reasoning", "analysis"],
            "max_tokens": 131072,
            "cost_per_1k_tokens": 0.005
        }
    ]
    
    # Chairman model (synthesizes final response)
    CHAIRMAN_MODEL = "google/gemini-3-pro-preview"
    CHAIRMAN_MODEL_ID = "google/gemini-3-pro-preview"
    
    # Council operations
    ENABLE_COUNCIL = os.getenv("ENABLE_COUNCIL", "true").lower() == "true"
    COUNCIL_TIMEOUT = 120  # seconds
    COUNCIL_MAX_RETRIES = 3
    
    # ============== Search & Research Tools ==============
    # Tavily API for web search verification
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    
    # ============== Performance & Optimization ==============
    # Prompt compression to reduce token usage
    ENABLE_PROMPT_COMPRESSION = os.getenv("ENABLE_PROMPT_COMPRESSION", "true").lower() == "true"
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "2000"))
    COMPRESSION_RATIO = float(os.getenv("COMPRESSION_RATIO", "0.4"))
    
    # ============== Output Configuration ==============
    OUTPUT_DIR = "output"
    ANALYSES_DIR = "output/analyses"
    PAPERS_DIR = "output/papers"
    CACHE_DIR = "output/cache"
    
    # Report settings
    REPORT_FORMATS = ['markdown', 'word']
    INCLUDE_CODE = True
    INCLUDE_VISUALIZATIONS = True
    
    # ============== Workflow Settings ==============
    # Recursive search depth for complex topics
    MAX_RESEARCH_DEPTH = 3
    MAX_ITERATIONS = 5
    
    # Research stages
    ENABLE_PLANNING = True
    ENABLE_RESEARCH = True
    ENABLE_VERIFICATION = True
    ENABLE_WRITING = True
    
    # ============== Data Processing ==============
    # Supported file formats
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
    
    # Data quality thresholds
    MISSING_VALUE_THRESHOLD = 0.5  # Drop columns with >50% missing
    OUTLIER_DETECTION_METHOD = 'iqr'  # 'iqr' or 'zscore'
    OUTLIER_THRESHOLD = 1.5  # For IQR method
    
    # ============== Analysis Settings ==============
    # Hypothesis generation
    MIN_CORRELATION_THRESHOLD = 0.3  # Minimum absolute correlation to report
    MAX_HYPOTHESES = 100  # Maximum number of hypotheses to generate
    COUNCIL_HYPOTHESES = 150  # With council, generate more hypotheses
    
    # Statistical testing
    SIGNIFICANCE_LEVEL = 0.05  # Alpha for statistical tests
    MIN_SAMPLE_SIZE = 30  # Minimum sample size for tests
    
    # Modeling
    TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing
    TIME_SERIES_FORECAST_HORIZON = 12  # Default forecast periods
    
    # ============== Insight Generation ==============
    # Insight generation
    MIN_INSIGHTS = 50  # Minimum number of insights to generate
    COUNCIL_INSIGHTS = 75  # With council, generate more insights
    INSIGHT_TYPES = ['correlation', 'trend', 'anomaly', 'pattern', 'outlier', 'distribution']
    
    # ============== Logging ==============
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = "output/logs"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Council-specific logs
    COUNCIL_LOG_DIR = "output/logs/council"
    COUNCIL_LOG_FILE = os.path.join(COUNCIL_LOG_DIR, "council_operations.jsonl")
    
    # Token tracking
    ENABLE_TOKEN_TRACKING = True
    TOKEN_LOG_FILE = os.path.join(LOG_DIR, "token_usage.jsonl")
    
    # ============== Error Handling & Self-Healing ==============
    MAX_RETRIES = 3  # Maximum retry attempts for failed operations
    RETRY_DELAY = 5  # Seconds between retries
    ENABLE_SELF_HEALING = os.getenv("ENABLE_SELF_HEALING", "true").lower() == "true"
    
    # Fallback API priority
    FALLBACK_APIS = ['perplexity', 'mistral', 'gemini', 'cohere']
    ENABLE_API_FALLBACK = True
    
    # ============== Chatbot & RAG ==============
    CHATBOT_ENABLED = True
    RAG_TOP_K = 5  # Top K documents to retrieve
    RAG_TEMPERATURE = 0.7  # Temperature for generation
    CHROMADB_PERSIST_DIR = "output/chromadb"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
    
    # ============== Dashboard ==============
    DASHBOARD_PORT = 8501  # Streamlit port
    ENABLE_DASHBOARD = True
    
    # ============== MCP Server Configurations ==============
    # Docker MCP Gateway (Unified access to all tools)
    DOCKER_MCP_ENABLED = True
    DOCKER_MCP_COMMAND = "docker"
    DOCKER_MCP_ARGS = ["mcp", "gateway", "run"]

    # ============== LLM Council Backend ==============
    COUNCIL_BACKEND_PATH = os.path.abspath("/home/engine/project/llm-council/backend")
    COUNCIL_CONFIG_FILE = os.path.join(COUNCIL_BACKEND_PATH, "config.py")
    
    @classmethod
    def ensure_output_dirs(cls):
        """Create necessary output directories"""
        directories = [
            cls.OUTPUT_DIR,
            cls.ANALYSES_DIR,
            cls.PAPERS_DIR,
            cls.CACHE_DIR,
            cls.LOG_DIR,
            cls.COUNCIL_LOG_DIR,
            cls.CHROMADB_PERSIST_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_model(cls, model_id=None):
        """Factory method to get configured LLM model"""
        from smolagents import LiteLLMModel
        import litellm
        
        litellm.drop_params = True
        
        model_id = model_id or f"perplexity/{cls.PERPLEXITY_MODEL}"
        api_key = cls.PERPLEXITY_API_KEY if "perplexity" in model_id else None
        
        return LiteLLMModel(model_id=model_id, api_key=api_key)
    
    @classmethod
    def get_perplexity_model(cls):
        """Get Perplexity model for research"""
        from smolagents import LiteLLMModel
        import litellm
        
        litellm.drop_params = True
        
        model_id = f"perplexity/{cls.PERPLEXITY_MODEL}"
        api_key = cls.PERPLEXITY_API_KEY
        
        return LiteLLMModel(model_id=model_id, api_key=api_key)
    
    @classmethod
    def get_mistral_model(cls):
        """Get Mistral model for code generation"""
        from smolagents import LiteLLMModel
        import litellm
        
        litellm.drop_params = True
        
        model_id = f"mistral/{cls.MISTRAL_MODEL}"
        api_key = cls.MISTRAL_API_KEY
        
        return LiteLLMModel(model_id=model_id, api_key=api_key)
    
    @classmethod
    def get_gemini_model(cls):
        """Get Gemini model"""
        from smolagents import LiteLLMModel
        import litellm
        
        litellm.drop_params = True
        
        model_id = f"gemini/{cls.GEMINI_MODEL}"
        api_key = cls.GEMINI_API_KEY
        
        return LiteLLMModel(model_id=model_id, api_key=api_key)
    
    @classmethod
    def get_council_models(cls):
        """Get list of council models"""
        return [model["id"] for model in cls.COUNCIL_MODELS]
    
    @classmethod
    def get_analysis_output_dir(cls, dataset_name: str) -> str:
        """Get output directory for a specific dataset analysis"""
        safe_name = dataset_name.replace('/', '_').replace('\\', '_')
        analysis_dir = os.path.join(cls.ANALYSES_DIR, safe_name)
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['data', 'code', 'visualizations', 'insights', 'logs']:
            os.makedirs(os.path.join(analysis_dir, subdir), exist_ok=True)
        
        return analysis_dir
    
    @classmethod
    def is_council_enabled(cls) -> bool:
        """Check if LLM Council is enabled"""
        return cls.ENABLE_COUNCIL


class LLMAPIEndpointManager:
    """Manages API endpoints and token usage for all LLM providers"""
    
    def __init__(self):
        self.token_usage_log = {}
        self.endpoint_status = {}
        
    def get_endpoint(self, provider: str, endpoint_type: str = "chat") -> str:
        """Get API endpoint for a specific provider"""
        endpoints = {
            "mistral": Config.MISTRAL_API_ENDPOINTS,
            "perplexity": Config.PERPLEXITY_API_ENDPOINTS,
            "gemini": Config.GEMINI_API_ENDPOINTS,
            "openrouter": Config.OPENROUTER_API_ENDPOINTS,
            "cohere": {
                "chat": f"{Config.COHERE_API_BASE}/chat"
            }
        }
        
        return endpoints.get(provider, {}).get(endpoint_type, "")
    
    def get_model_info(self, model_id: str) -> dict:
        """Get complete model information"""
        # Determine provider from model_id
        provider = model_id.split('/')[0]
        
        # Find model in council or fallback to known models
        model_config = None
        
        for model in Config.COUNCIL_MODELS:
            if model["id"] == model_id:
                model_config = model
                break
        
        if not model_config:
            # Fallback configurations
            fallback_models = {
                "mistral": Config.MISTRAL_MODEL,
                "perplexity": Config.PERPLEXITY_MODEL,
                "gemini": Config.GEMINI_MODEL,
                "cohere": Config.COHERE_MODEL
            }
            model_config = {
                "id": model_id,
                "name": model_id.split('/')[-1],
                "provider": provider,
                "model": fallback_models.get(provider, "")
            }
        
        return {
            "model_id": model_id,
            "model_name": model_config.get("name", "Unknown"),
            "provider": model_config.get("provider", "Unknown"),
            "provider_endpoint": self.get_endpoint(model_config.get("provider", "Unknown")),
            "max_tokens": model_config.get("max_tokens", 4096),
            "cost_per_1k_tokens": model_config.get("cost_per_1k_tokens", 0.001)
        }
    
    def log_token_usage(self, model_id: str, prompt_tokens: int, completion_tokens: int, operation: str):
        """Log token usage for an operation"""
        if not Config.ENABLE_TOKEN_TRACKING:
            return
        
        # Calculate cost
        model_info = self.get_model_info(model_id)
        cost_per_1k = model_info.get("cost_per_1k_tokens", 0.001)
        
        total_tokens = prompt_tokens + completion_tokens
        cost = (total_tokens / 1000) * cost_per_1k
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "model_name": model_info.get("model_name", "Unknown"),
            "provider": model_info.get("provider", "Unknown"),
            "operation": operation,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(cost, 6)
        }
        
        self.token_usage_log[datetime.now().isoformat()] = log_entry
        
        # Write to log file
        try:
            with open(Config.TOKEN_LOG_FILE, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not write to token log: {e}")
        
        return log_entry
    
    def get_token_summary(self) -> dict:
        """Get summary of token usage"""
        total_cost = 0
        total_tokens = 0
        model_usage = {}
        
        for entry in self.token_usage_log.values():
            total_tokens += entry.get("total_tokens", 0)
            total_cost += entry.get("estimated_cost_usd", 0)
            
            model_id = entry.get("model_id")
            if model_id not in model_usage:
                model_usage[model_id] = {
                    "model_name": entry.get("model_name"),
                    "total_tokens": 0,
                    "total_cost": 0,
                    "operations": 0
                }
            
            model_usage[model_id]["total_tokens"] += entry.get("total_tokens", 0)
            model_usage[model_id]["total_cost"] += entry.get("estimated_cost_usd", 0)
            model_usage[model_id]["operations"] += 1
        
        return {
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "operations_logged": len(self.token_usage_log),
            "models_used": model_usage
        }
    
    def log_endpoint_status(self, provider: str, endpoint: str, status: str, response_time: float = None, error: str = None):
        """Log endpoint status for monitoring"""
        if not Config.ENABLE_TOKEN_TRACKING:
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "endpoint": endpoint,
            "status": status,  # success, error, timeout
            "response_time_ms": response_time * 1000 if response_time else None,
            "error": error
        }
        
        self.endpoint_status[datetime.now().isoformat()] = log_entry
        
        return log_entry


class CouncilOperationLogger:
    """Logger for LLM Council operations"""
    
    def __init__(self, log_file: str = None):
        """Initialize council operation logger"""
        self.log_file = log_file or Config.COUNCIL_LOG_FILE
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
    
    def log_stage1(self, user_query: str, responses: list):
        """Log Stage 1: Individual responses from all council models"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": "stage1_collect_responses",
            "user_query": user_query,
            "models_queried": [r.get("model") for r in responses],
            "successful_responses": len([r for r in responses if r.get("response")]),
            "failed_responses": len([r for r in responses if not r.get("response")])
        }
        
        self._write_log(log_entry)
    
    def log_stage2(self, user_query: str, rankings: list):
        """Log Stage 2: Peer rankings of anonymized responses"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": "stage2_collect_rankings",
            "user_query": user_query,
            "number_of_rankings": len(rankings),
            "ranking_models": [r.get("model") for r in rankings]
        }
        
        self._write_log(log_entry)
    
    def log_stage3(self, user_query: str, synthesis: dict):
        """Log Stage 3: Final synthesis from Chairman"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": "stage3_synthesize_final",
            "user_query": user_query,
            "chairman_model": synthesis.get("model", "unknown"),
            "final_response_length": len(synthesis.get("response", "")),
            "has_recommendation": bool(synthesis.get("recommendation", {}))
        }
        
        self._write_log(log_entry)
    
    def log_council_summary(self, user_query: str, metadata: dict):
        """Log complete council operation summary"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": "council_summary",
            "user_query": user_query,
            "total_models_queried": len(metadata.get("label_to_model", {})),
            "council_used": True,
            "council_duration_ms": metadata.get("duration_ms", 0)
        }
        
        self._write_log(log_entry)
    
    def log_error(self, stage: str, error: str, context: dict = None):
        """Log council operation error"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "error": error,
            "context": context or {},
            "is_error": True
        }
        
        self._write_log(log_entry)
    
    def _write_log(self, log_entry: dict):
        """Write log entry to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not write to council log: {e}")

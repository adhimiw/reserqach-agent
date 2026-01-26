"""
Autonomous Data Science System Configuration
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Master configuration for autonomous data science system"""
    
    # ============== LLM Models ==============
    # Primary LLM - Mistral Agent API for code generation
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "HKK5Q0lja9HBOwIEXt82sncuQb3RksPW")
    MISTRAL_MODEL = "mistral-large-latest"
    MISTRAL_API_BASE = "https://api.mistral.ai/v1"
    
    # Perplexity API for real-time research and reasoning
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "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")
    PERPLEXITY_MODEL = "sonar-pro"
    PERPLEXITY_API_BASE = "https://api.perplexity.ai"
    
    # Fallback models
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "39sIDzEasPItQPgAReUIuiwLoq9tKiOcgQYseK6E")
    COHERE_MODEL = "command"
    
    # ============== Data Processing ==============
    # Supported file formats
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
    
    # Data quality thresholds
    MISSING_VALUE_THRESHOLD = 0.5  # Drop columns with >50% missing
    OUTLIER_DETECTION_METHOD = 'iqr'  # 'iqr' or 'zscore'
    
    # ============== Analysis Settings ==============
    # Hypothesis generation
    MIN_CORRELATION_THRESHOLD = 0.3  # Minimum absolute correlation to report
    MAX_HYPOTHESES = 100  # Maximum number of hypotheses to generate
    
    # Statistical testing
    SIGNIFICANCE_LEVEL = 0.05  # Alpha for statistical tests
    MIN_SAMPLE_SIZE = 30  # Minimum sample size for tests
    
    # Modeling
    TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing
    TIME_SERIES_FORECAST_HORIZON = 12  # Default forecast periods
    
    # ============== Insight Generation ==============
    MIN_INSIGHTS = 50  # Minimum number of insights to generate
    INSIGHT_TYPES = ['correlation', 'trend', 'anomaly', 'pattern', 'outlier', 'distribution']
    
    # ============== Output Configuration ==============
    OUTPUT_DIR = "output"
    ANALYSES_DIR = "output/analyses"
    NOTEBOOKS_DIR = "output/notebooks"
    REPORTS_DIR = "output/reports"
    VISUALIZATIONS_DIR = "output/visualizations"
    LOGS_DIR = "output/logs"
    
    # Report settings
    REPORT_FORMATS = ['markdown', 'word']  # Output formats
    INCLUDE_CODE = True  # Include Python code in reports
    INCLUDE_VISUALIZATIONS = True  # Embed plots in reports
    
    # ============== MCP Server Configurations ==============
    # Docker MCP Gateway (Unified access to all tools)
    DOCKER_MCP_ENABLED = True
    DOCKER_MCP_COMMAND = "docker"
    DOCKER_MCP_ARGS = ["mcp", "gateway", "run"]
    
    # ChromaDB for RAG
    CHROMADB_PERSIST_DIR = "output/chromadb"
    
    # ============== Logging Dashboard ==============
    DASHBOARD_PORT = 8501  # Streamlit port
    ENABLE_DASHBOARD = True
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # ============== Error Handling & Self-Healing ==============
    MAX_RETRIES = 3  # Maximum retry attempts for failed operations
    RETRY_DELAY = 5  # Seconds between retries
    ENABLE_SELF_HEALING = True  # Enable automatic error recovery
    
    # Fallback API priority
    FALLBACK_APIS = ['perplexity', 'mistral', 'cohere']
    
    # ============== Chatbot & RAG ==============
    CHATBOT_ENABLED = True
    RAG_TOP_K = 5  # Top K documents to retrieve
    RAG_TEMPERATURE = 0.7  # Temperature for generation
    
    @classmethod
    def get_model(cls, model_id=None):
        """Factory method to get configured LLM model"""
        from smolagents import LiteLLMModel
        import litellm
        
        litellm.drop_params = True
        
        model_id = model_id or f"mistral/{cls.MISTRAL_MODEL}"
        api_key = cls.MISTRAL_API_KEY
        
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
    def ensure_output_dirs(cls):
        """Create necessary output directories"""
        for directory in [
            cls.OUTPUT_DIR,
            cls.ANALYSES_DIR,
            cls.NOTEBOOKS_DIR,
            cls.REPORTS_DIR,
            cls.VISUALIZATIONS_DIR,
            cls.LOGS_DIR,
            cls.CHROMADB_PERSIST_DIR
        ]:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_analysis_output_dir(cls, dataset_name: str) -> str:
        """Get output directory for a specific dataset analysis (per-run)"""
        safe_name = dataset_name.replace('/', '_').replace('\\', '_')
        analysis_root = os.path.join(cls.ANALYSES_DIR, safe_name)
        os.makedirs(analysis_root, exist_ok=True)

        # Create a unique run directory per execution
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(analysis_root, run_id)
        if os.path.exists(run_dir):
            counter = 1
            while os.path.exists(f"{run_dir}_{counter}"):
                counter += 1
            run_dir = f"{run_dir}_{counter}"
            run_id = os.path.basename(run_dir)
        
        # Create subdirectories
        for subdir in ['data', 'code', 'visualizations', 'insights', 'logs']:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

        # Track latest run for this dataset
        latest_path = os.path.join(analysis_root, "latest.json")
        try:
            with open(latest_path, 'w') as f:
                json.dump({
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
        except Exception:
            pass
        
        return run_dir

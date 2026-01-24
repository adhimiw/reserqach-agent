"""
LLM API Integration Module
Provides comprehensive token tracking, endpoint management, and logging for all LLM providers
"""

import httpx
import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Setup logging
logger = logging.getLogger("LLMAPIManager")
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class LLMAPIClient:
    """Client for making actual API requests to LLM providers"""
    
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120):
        """
        Initialize API client
        
        Args:
            base_url: Base URL for API
            api_key: API key for authentication
            model: Model identifier
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # Token tracking for this client
        self.tokens_used = defaultdict(int)
        self.requests_made = 0
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                              max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Send chat completion request
        
        Args:
            messages: List of message dicts
            max_tokens: Maximum tokens for response (optional)
        
        Returns:
            Dictionary with response, tokens used, and cost
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "no-cache"
        }
        
        payload = {
            "model": self.model,
            "messages": messages
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        start_time = datetime.now()
        
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract response and token usage
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens
            
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Track usage
            self.tokens_used["total"] += total_tokens
            self.tokens_used["prompt"] += prompt_tokens
            self.tokens_used["completion"] += completion_tokens
            self.requests_made += 1
            
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            logger.info(f"API Request Success: {self.model} | {total_tokens} tokens | {duration_ms:.0f}ms")
            
            return {
                "success": True,
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "duration_ms": duration_ms,
                "model": self.model,
                "provider": self._get_provider()
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"API Request Failed: {self.model} | Status: {e.response.status_code} | {e.response.text}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}",
                "status_code": e.response.status_code,
                "message": e.response.text,
                "model": self.model,
                "provider": self._get_provider()
            }
        except Exception as e:
            logger.error(f"API Request Error: {self.model} | {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
                "provider": self._get_provider()
            }
    
    def _get_provider(self) -> str:
        """Extract provider name from model"""
        if "openai" in self.model:
            return "openai"
        elif "anthropic" in self.model:
            return "anthropic"
        elif "perplexity" in self.model:
            return "perplexity"
        elif "google" in self.model:
            return "google"
        elif "mistral" in self.model:
            return "mistral"
        elif "x-ai" in self.model:
            return "x-ai"
        elif "cohere" in self.model:
            return "cohere"
        else:
            return "unknown"
    
    def get_total_tokens_used(self) -> Dict[str, int]:
        """Get total tokens used by type"""
        return dict(self.tokens_used)
    
    def get_request_count(self) -> int:
        """Get total number of requests made"""
        return self.requests_made


class LLMAPIManager:
    """Manages multiple LLM API clients with unified logging and token tracking"""
    
    # Provider configurations
    PROVIDERS = {
        "perplexity": {
            "name": "Perplexity",
            "base_url": "https://api.perplexity.ai",
            "models": {
                "sonar-pro": {"id": "perplexity/sonar-pro", "name": "Sonar Pro", "tokens_per_1k": 1.0, "cost_per_1k": 0.01},
                "sonar-medium": {"id": "perplexity/sonar-medium", "name": "Sonar Medium", "tokens_per_1k": 3.0, "cost_per_1k": 0.03},
            "sonar-mini": {"id": "perplexity/sonar-mini", "name": "Sonar Mini", "tokens_per_1k": 0.6, "cost_per_1k": 0.006}
            },
            "endpoints": {
                "chat": "/chat/completions",
                "models": "/models",
                "usage": "/usage"
            },
            "headers": {
                "Authorization": f"Bearer {{api_key}}",
                "Content-Type": "application/json"
            }
        },
        
        "mistral": {
            "name": "Mistral",
            "base_url": "https://api.mistral.ai/v1",
            "models": {
                "mistral-large-latest": {"id": "mistral/mistral-large-latest", "name": "Mistral Large Latest", "tokens_per_1k": 0.5, "cost_per_1k": 0.005},
                "mistral-medium": {"id": "mistral/mistral-medium", "name": "Mistral Medium", "tokens_per_1k": 0.25, "cost_per_1k": 0.002},
                "codestral-latest": {"id": "codestral/codestral-latest", "name": "Codestral Latest", "tokens_per_1k": 0.15, "cost_per_1k": 0.0015}
            },
            "endpoints": {
                "chat": "/chat/completions",
                "models": "/models"
            },
            "headers": {
                "Authorization": f"Bearer {{api_key}}",
                "Content-Type": "application/json"
            }
        },
        
        "google": {
            "name": "Google Gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "models": {
                "gemini-2.5-flash": {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "tokens_per_1k": 0.00, "cost_per_1k": 0.0005},
                "gemini-2.0-flash": {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "tokens_per_1k": 0.00, "cost_per_1k": 0.0002},
                "gemini-1.5-flash": {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "tokens_per_1k": 0.001, "cost_per_1k": 0.00004},
                "gemini-pro-preview": {"id": "gemini-pro", "name": "Gemini Pro", "tokens_per_1k": 1.0, "cost_per_1k": 0.0035}
            },
            "endpoints": {
                "generate": "/models/gemini-2.5-flash-exp:generateContent",
                "chat": "/models/gemini-pro:generateContent"
            },
            "headers": {
                "Authorization": "Bearer {{api_key}}",
                "Content-Type": "application/json"
            }
        },
        
        "openrouter": {
            "name": "OpenRouter (Multi-Provider)",
            "base_url": "https://openrouter.ai/api/v1",
            "available_providers": ["openai", "anthropic", "google", "mistral", "x-ai", "cohere"],
            "endpoints": {
                "chat": "/chat/completions"
            },
            "headers": {
                "Authorization": "Bearer {{api_key}}",
                "Content-Type": "application/json",
                "HTTP-Referer": "openrouter:1.2.8"
            }
        }
    }
    
    def __init__(self, config: dict = None):
        """
        Initialize API manager with provider configurations
        
        Args:
            config: Dictionary with API keys for each provider
        """
        self.config = config or {}
        self.clients = {}
        self.enabled_providers = set()
        
        # Initialize clients for enabled providers
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients for all configured providers"""
        # Perplexity
        if self.config.get("PERPLEXITY_API_KEY"):
            self.clients["perplexity"] = LLMAPIClient(
                base_url=self.PROVIDERS["perplexity"]["base_url"],
                api_key=self.config.get("PERPLEXITY_API_KEY"),
                model="sonar-pro"  # Default model
            )
            self.enabled_providers.add("perplexity")
            logger.info("Perplexity API client initialized")
        
        # Mistral
        if self.config.get("MISTRAL_API_KEY"):
            self.clients["mistral"] = LLMAPIClient(
                base_url=self.PROVIDERS["mistral"]["base_url"],
                api_key=self.config.get("MISTRAL_API_KEY"),
                model="mistral-large-latest"
            )
            self.enabled_providers.add("mistral")
            logger.info("Mistral API client initialized")
        
        # Google Gemini
        if self.config.get("GEMINI_API_KEY"):
            self.clients["google"] = LLMAPIClient(
                base_url=self.PROVIDERS["google"]["base_url"],
                api_key=self.config.get("GEMINI_API_KEY"),
                model="gemini-2.0-flash"
            )
            self.enabled_providers.add("google")
            logger.info("Google Gemini API client initialized")
        
        # OpenRouter
        if self.config.get("OPENROUTER_API_KEY"):
            # OpenRouter handles all providers through single endpoint
            self.clients["openrouter"] = LLMAPIClient(
                base_url=self.PROVIDERS["openrouter"]["base_url"],
                api_key=self.config.get("OPENROUTER_API_KEY"),
                model="openai/gpt-4"
            )
            self.enabled_providers.add("openrouter")
            logger.info("OpenRouter API client initialized")
    
    async def query_model(self, provider: str, model_id: str, 
                     messages: List[Dict[str, str]], max_tokens: int = None) -> Dict[str, Any]:
        """
        Query a specific model
        
        Args:
            provider: Provider name (perplexity, mistral, google, openrouter)
            model_id: Model identifier
            messages: List of message dicts
            max_tokens: Maximum tokens for response
        
        Returns:
            Response dictionary
        """
        if provider not in self.clients:
            return {
                "success": False,
                "error": f"Provider {provider} not configured or initialized",
                "model": model_id,
                "tokens_used": {}
            }
        
        client = self.clients[provider]
        
        # Map model_id to actual model for client
        # For OpenRouter, model_id should be the full path
        actual_model = model_id
        
        response = await client.chat_completion(messages, max_tokens=max_tokens)
        response["model"] = actual_model
        response["provider"] = provider
        
        return response
    
    async def query_model_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Query multiple models in parallel
        
        Args:
            requests: List of dicts with 'provider', 'model_id', 'messages' keys
        
        Returns:
            List of response dictionaries
        """
        tasks = []
        for req in requests:
            task = self.query_model(
                provider=req["provider"],
                model_id=req["model_id"],
                messages=req["messages"]
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return responses
    
    def get_client(self, provider: str) -> Optional[LLMAPIClient]:
        """Get API client for a specific provider"""
        return self.clients.get(provider)
    
    def get_provider_info(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get provider information"""
        return self.PROVIDERS.get(provider)
    
    def get_all_tokens_used(self) -> Dict[str, Dict[str, int]]:
        """Get token usage from all clients"""
        all_tokens = {}
        
        for provider, client in self.clients.items():
            all_tokens[provider] = client.get_total_tokens_used()
        
        return all_tokens
    
    def get_total_requests(self) -> int:
        """Get total number of API requests made"""
        total = 0
        for client in self.clients.values():
            total += client.get_request_count()
        return total
    
    def get_total_cost(self) -> float:
        """Calculate total cost based on token usage"""
        total_cost = 0.0
        
        for provider, client in self.clients.items():
            provider_info = self.get_provider_info(provider)
            if not provider_info:
                continue
            
            tokens = client.get_total_tokens_used().get("total", 0)
            
            # Get pricing for default model
            models = provider_info.get("models", {})
            if not models:
                cost_per_1k = 0.0
            else:
                default_model = list(models.values())[0]
                cost_per_1k = default_model.get("cost_per_1k", 0.0)
            
            # Calculate cost
            provider_cost = (tokens / 1000) * cost_per_1k
            total_cost += provider_cost
        
        return round(total_cost, 6)
    
    def get_api_status(self) -> Dict[str, Dict[str, str]]:
        """Get API status for all providers"""
        status = {}
        
        for provider in self.enabled_providers:
            # Check if provider has a client
            if provider in self.clients:
                # Could add health check here
                status[provider] = {
                    "configured": True,
                    "endpoint": self.PROVIDERS[provider]["base_url"],
                    "requests": self.clients[provider].get_request_count(),
                    "tokens_used": self.clients[provider].get_total_tokens_used().get("total", 0)
                }
            else:
                status[provider] = {
                    "configured": False,
                    "endpoint": "N/A",
                    "requests": 0,
                    "tokens_used": 0
                }
        
        return status


class TokenLogger:
    """Comprehensive token usage logger"""
    
    def __init__(self, log_file: str = "output/logs/token_usage.jsonl"):
        """
        Initialize token logger
        
        Args:
            log_file: Path to token log file
        """
        self.log_file = log_file
        self.usage_log = []
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_request(self, provider: str, model: str, prompt_tokens: int,
                   completion_tokens: int, cost: float, duration_ms: float,
                   success: bool, error: str = None):
        """
        Log an API request
        
        Args:
            provider: LLM provider name
            model: Model identifier
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cost: Estimated cost in USD
            duration_ms: Request duration in milliseconds
            success: Whether request was successful
            error: Error message if failed
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(cost, 6),
            "duration_ms": duration_ms,
            "success": success,
            "error": error
        }
        
        self.usage_log.append(log_entry)
        
        # Write to file immediately
        self._write_log()
        
        # Also log to console
        if success:
            logger.info(f"✓ {provider}/{model} | {prompt_tokens + completion_tokens} tokens | ${log_entry['cost_usd']:.6f} | {duration_ms:.0f}ms")
        else:
            logger.error(f"✗ {provider}/{model} | FAILED | {error}")
    
    def log_batch_requests(self, responses: List[Dict[str, Any]]):
        """
        Log multiple batch requests
        
        Args:
            responses: List of API response dictionaries
        """
        for response in responses:
            if response.get("success"):
                self.log_request(
                    provider=response.get("provider", "unknown"),
                    model=response.get("model", "unknown"),
                    prompt_tokens=response.get("prompt_tokens", 0),
                    completion_tokens=response.get("completion_tokens", 0),
                    cost=response.get("estimated_cost", 0),
                    duration_ms=response.get("duration_ms", 0),
                    success=True
                )
            else:
                self.log_request(
                    provider=response.get("provider", "unknown"),
                    model=response.get("model", "unknown"),
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost=0,
                    duration_ms=0,
                    success=False,
                    error=response.get("error", "Unknown error")
                )
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of usage"""
        if not self.usage_log:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "successful_requests": 0,
                "failed_requests": 0,
                "by_provider": {},
                "by_model": {}
            }
        
        total_requests = len(self.usage_log)
        total_tokens = sum(log.get("total_tokens", 0) for log in self.usage_log)
        total_cost = sum(log.get("cost_usd", 0) for log in self.usage_log)
        successful = sum(1 for log in self.usage_log if log.get("success", False))
        failed = total_requests - successful
        
        # By provider
        by_provider = defaultdict(lambda: {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "successful_requests": 0,
            "failed_requests": 0,
            "models": {}
        })
        
        by_model = defaultdict(lambda: {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "models": {}
        })
        
        for log in self.usage_log:
            provider = log.get("provider", "unknown")
            model = log.get("model", "unknown")
            
            by_provider[provider]["total_requests"] += 1
            by_provider[provider]["total_tokens"] += log.get("total_tokens", 0)
            by_provider[provider]["total_cost"] += log.get("cost_usd", 0)
            by_provider[provider]["successful_requests"] += 1 if log.get("success", False) else 0
            by_provider[provider]["failed_requests"] += 0 if log.get("success", False) else 1
            
            by_model[model]["total_requests"] += 1
            by_model[model]["total_tokens"] += log.get("total_tokens", 0)
            by_model[model]["total_cost"] += log.get("cost_usd", 0)
        
        # Convert to regular dicts
        by_provider = {k: dict(v) for k, v in by_provider.items()}
        by_model = {k: dict(v) for k, v in by_model.items()}
        
        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "successful_requests": successful,
            "failed_requests": failed,
            "by_provider": by_provider,
            "by_model": by_model
        }
    
    def _write_log(self):
        """Write log to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.usage_log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write token log: {e}")
    
    def read_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read token usage log"""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            return data[-limit:] if len(data) > limit else data
        except Exception as e:
            logger.error(f"Failed to read token log: {e}")
            return []
    
    def clear_log(self):
        """Clear token log"""
        self.usage_log = []
        self._write_log()


class EndpointHealthChecker:
    """Checks health and availability of API endpoints"""
    
    def __init__(self, api_manager: LLMAPIManager):
        """
        Initialize health checker
        
        Args:
            api_manager: API manager instance
        """
        self.api_manager = api_manager
        self.health_status = {}
        self.last_check_time = None
    
    async def check_provider(self, provider: str) -> Dict[str, Any]:
        """
        Check health of a specific provider
        
        Args:
            provider: Provider name
        
        Returns:
            Health status dictionary
        """
        provider_info = self.api_manager.get_provider_info(provider)
        
        if not provider_info:
            return {
                "provider": provider,
                "status": "not_configured",
                "endpoint": "N/A",
                "last_checked": None,
                "response_time_ms": None
            }
        
        base_url = provider_info["base_url"]
        model_id = list(provider_info["models"].keys())[0]
        
        # Simple health check - send a minimal request
        test_messages = [{"role": "user", "content": "Health check"}]
        
        start_time = datetime.now()
        
        try:
            # Try to use the API manager's query method
            # For this, we'll just test connectivity
            import httpx
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(base_url + "/models", headers={
                    "Authorization": f"Bearer {self.api_manager.config.get(provider.upper() + '_API_KEY', '')}"
                })
                
                if response.status_code == 200:
                    end_time = datetime.now()
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                    
                    return {
                        "provider": provider,
                        "status": "healthy",
                        "endpoint": base_url,
                        "last_checked": end_time.isoformat(),
                        "response_time_ms": duration_ms,
                        "message": "API is accessible"
                    }
                else:
                    return {
                        "provider": provider,
                        "status": "unhealthy",
                        "endpoint": base_url,
                        "last_checked": end_time.isoformat(),
                        "response_time_ms": None,
                        "message": f"API returned status {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "provider": provider,
                "status": "error",
                "endpoint": base_url,
                "last_checked": datetime.now().isoformat(),
                "response_time_ms": None,
                "message": f"Health check failed: {str(e)}"
            }
    
    async def check_all_providers(self) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all configured providers
        
        Returns:
            Dictionary with all provider health statuses
        """
        health_results = {}
        
        providers_to_check = ["perplexity", "mistral", "google"]
        
        tasks = []
        for provider in providers_to_check:
            task = self.check_provider(provider)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                health_results[result.args[0]] = {
                    "status": "error",
                    "message": f"Health check failed: {str(result)}"
                }
            else:
                health_results[result.args[0]] = result
        
        self.last_check_time = datetime.now()
        
        return health_results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of all provider health"""
        healthy = sum(1 for s in self.health_status.values() if s.get("status") == "healthy")
        total = len(self.health_status)
        
        return {
            "total_providers": total,
            "healthy_providers": healthy,
            "unhealthy_providers": total - healthy,
            "last_check_time": self.last_check_time,
            "providers": list(self.health_status.keys())
        }

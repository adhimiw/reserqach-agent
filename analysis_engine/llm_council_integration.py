"""
LLM Council Integration for Data Science System
Integrates LLM Council (multi-agent consensus) into analysis workflow
Uses multiple LLM providers: Gemini, Mistral, Cohere, OpenRouter
Based on Karpathy's LLM Council: https://github.com/karpathy/llm-council
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging
import sys
import os
import json
import re
import httpx
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pandas (lazy, only if needed for actual data processing)
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Note: pandas not available, running structural tests only")

# Import LLM API management modules (with graceful fallback)
try:
    from llm_api_integration import LLMAPIManager, TokenLogger, EndpointHealthChecker
    HAS_API_MANAGER = True
except ImportError:
    HAS_API_MANAGER = False
    LLMAPIManager = None
    TokenLogger = None
    EndpointHealthChecker = None

logger = logging.getLogger("LLMCouncilIntegration")

# ============================================================================
# MULTI-PROVIDER LLM COUNCIL (Inspired by Karpathy's llm-council)
# ============================================================================

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# API URLs
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
COHERE_API_URL = "https://api.cohere.com/v2/chat"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Council members - diverse models from different providers
COUNCIL_MODELS = [
    {"provider": "mistral", "model": "mistral-small-latest", "name": "Mistral Small"},
    {"provider": "cohere", "model": "command-r7b-12-2024", "name": "Cohere Command-R"},
    {"provider": "openrouter", "model": "mistralai/devstral-2512:free", "name": "Devstral (OpenRouter)"},
    {"provider": "gemini", "model": "gemini-flash-latest", "name": "Gemini Flash"},
]

# Chairman model - synthesizes final response  
CHAIRMAN_MODEL = {"provider": "mistral", "model": "mistral-small-latest", "name": "Mistral Small"}

# Fallback models per provider
FALLBACK_MODELS = {
    "gemini": ["gemini-2.5-flash", "gemini-2.0-flash"],
    "mistral": ["mistral-tiny-latest", "open-mistral-nemo"],
    "cohere": ["c4ai-aya-expanse-8b"],
    "openrouter": ["liquid/lfm-2.5-1.2b-instruct:free", "nvidia/nemotron-3-nano-30b-a3b:free"],
}


async def query_mistral_model(
    model: str,
    prompt: str,
    timeout: float = 60.0,
    temperature: float = 0.7,
    max_retries: int = 2
) -> Optional[Dict[str, Any]]:
    """Query Mistral API."""
    if not MISTRAL_API_KEY:
        logger.error("MISTRAL_API_KEY not set")
        return None
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 4096
    }
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(MISTRAL_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return {"content": content, "model_used": model, "provider": "mistral"}
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                await asyncio.sleep(5 * (attempt + 1))
                continue
            logger.error(f"Mistral error ({model}): {e.response.status_code}")
            break
        except Exception as e:
            logger.error(f"Mistral error ({model}): {e}")
            break
    
    # Try fallbacks
    for fallback in FALLBACK_MODELS.get("mistral", []):
        try:
            payload["model"] = fallback
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(MISTRAL_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                logger.info(f"Used Mistral fallback: {fallback}")
                return {"content": content, "model_used": fallback, "provider": "mistral"}
        except:
            continue
    return None


async def query_cohere_model(
    model: str,
    prompt: str,
    timeout: float = 60.0,
    temperature: float = 0.7,
    max_retries: int = 2
) -> Optional[Dict[str, Any]]:
    """Query Cohere API."""
    if not COHERE_API_KEY:
        logger.error("COHERE_API_KEY not set")
        return None
    
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096
    }
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(COHERE_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", [{}])[0].get("text", "")
                return {"content": content, "model_used": model, "provider": "cohere"}
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                await asyncio.sleep(5 * (attempt + 1))
                continue
            logger.error(f"Cohere error ({model}): {e.response.status_code}")
            break
        except Exception as e:
            logger.error(f"Cohere error ({model}): {e}")
            break
    
    # Try fallbacks
    for fallback in FALLBACK_MODELS.get("cohere", []):
        try:
            payload["model"] = fallback
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(COHERE_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", [{}])[0].get("text", "")
                logger.info(f"Used Cohere fallback: {fallback}")
                return {"content": content, "model_used": fallback, "provider": "cohere"}
        except:
            continue
    return None


async def query_openrouter_model(
    model: str,
    prompt: str,
    timeout: float = 60.0,
    temperature: float = 0.7,
    max_retries: int = 2
) -> Optional[Dict[str, Any]]:
    """Query OpenRouter API."""
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set")
        return None
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 4096
    }
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return {"content": content, "model_used": model, "provider": "openrouter"}
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                await asyncio.sleep(5 * (attempt + 1))
                continue
            logger.error(f"OpenRouter error ({model}): {e.response.status_code}")
            break
        except Exception as e:
            logger.error(f"OpenRouter error ({model}): {e}")
            break
    
    # Try fallbacks
    for fallback in FALLBACK_MODELS.get("openrouter", []):
        try:
            payload["model"] = fallback
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                logger.info(f"Used OpenRouter fallback: {fallback}")
                return {"content": content, "model_used": fallback, "provider": "openrouter"}
        except:
            continue
    return None


async def query_gemini_model(
    model: str,
    prompt: str,
    timeout: float = 120.0,
    temperature: float = 0.7,
    max_retries: int = 2,
    use_fallback: bool = True
) -> Optional[Dict[str, Any]]:
    """Query Gemini API."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set")
        return None
    
    models_to_try = [model]
    if use_fallback:
        models_to_try.extend(FALLBACK_MODELS.get("gemini", []))
    
    for current_model in models_to_try:
        url = f"{GEMINI_API_URL}/{current_model}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 8192}
        }
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    if "candidates" in data and len(data["candidates"]) > 0:
                        candidate = data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            parts = candidate["content"]["parts"]
                            if len(parts) > 0 and "text" in parts[0]:
                                return {"content": parts[0]["text"], "model_used": current_model, "provider": "gemini"}
                    break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    break
                break
            except:
                break
    return None


async def query_model(model_config: Dict[str, str], prompt: str) -> Optional[Dict[str, Any]]:
    """
    Query a model based on its provider configuration.
    
    Args:
        model_config: Dict with 'provider', 'model', 'name' keys
        prompt: The prompt to send
    
    Returns:
        Response dict with 'content', 'model_used', 'provider' keys
    """
    provider = model_config["provider"]
    model = model_config["model"]
    
    if provider == "mistral":
        return await query_mistral_model(model, prompt)
    elif provider == "cohere":
        return await query_cohere_model(model, prompt)
    elif provider == "openrouter":
        return await query_openrouter_model(model, prompt)
    elif provider == "gemini":
        return await query_gemini_model(model, prompt)
    else:
        logger.error(f"Unknown provider: {provider}")
        return None


async def query_council_models_parallel(
    models: List[Dict[str, str]],
    prompt: str,
    stagger_delay: float = 1.0
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query all council models with staggered requests.
    
    Args:
        models: List of model configurations
        prompt: Prompt to send to each model
        stagger_delay: Delay between requests
    
    Returns:
        Dict mapping model name to response
    """
    results = {}
    
    for i, model_config in enumerate(models):
        if i > 0:
            await asyncio.sleep(stagger_delay)
        
        model_name = model_config["name"]
        logger.info(f"  Querying {model_name} ({model_config['provider']})...")
        response = await query_model(model_config, prompt)
        results[model_name] = response
    
    return results


async def stage1_collect_responses(user_query: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.
    """
    logger.info(f"Stage 1: Collecting responses from {len(COUNCIL_MODELS)} council models")
    
    responses = await query_council_models_parallel(COUNCIL_MODELS, user_query)
    
    stage1_results = []
    for model_name, response in responses.items():
        if response is not None:
            stage1_results.append({
                "model": model_name,
                "provider": response.get("provider", "unknown"),
                "model_used": response.get("model_used", "unknown"),
                "response": response.get("content", "")
            })
            logger.info(f"  ✓ {model_name} responded")
        else:
            logger.warning(f"  ✗ {model_name} failed to respond")
    
    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each council model ranks the anonymized responses.
    """
    logger.info("Stage 2: Collecting peer rankings from council")
    
    labels = [chr(65 + i) for i in range(len(stage1_results))]
    label_to_model = {
        f"Response {label}": result["model"]
        for label, result in zip(labels, stage1_results)
    }
    
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response'][:2000]}"
        for label, result in zip(labels, stage1_results)
    ])
    
    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. Briefly evaluate each response (what it does well and poorly).
2. At the end, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as:
FINAL RANKING:
1. Response X
2. Response Y
(etc.)

Now provide your evaluation and ranking:"""
    
    responses = await query_council_models_parallel(COUNCIL_MODELS, ranking_prompt)
    
    stage2_results = []
    for model_name, response in responses.items():
        if response is not None:
            full_text = response.get("content", "")
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model_name,
                "ranking": full_text,
                "parsed_ranking": parsed
            })
            logger.info(f"  ✓ {model_name} provided ranking")
    
    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.
    """
    logger.info(f"Stage 3: Chairman ({CHAIRMAN_MODEL['name']}) synthesizing final response")
    
    stage1_text = "\n\n".join([
        f"Model: {result['model']} ({result['provider']})\nResponse: {result['response'][:1500]}"
        for result in stage1_results
    ])
    
    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking'][:1000]}"
        for result in stage2_results
    ])
    
    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses and ranked each other.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Synthesize all information into a single, comprehensive, accurate answer. Consider:
- Individual responses and their insights
- Peer rankings and response quality
- Patterns of agreement or disagreement

Provide a clear, well-reasoned final answer:"""
    
    response = await query_model(CHAIRMAN_MODEL, chairman_prompt)
    
    if response is None:
        return {
            "model": CHAIRMAN_MODEL["name"],
            "response": "Error: Unable to generate final synthesis."
        }
    
    return {
        "model": CHAIRMAN_MODEL["name"],
        "provider": response.get("provider", "unknown"),
        "response": response.get("content", "")
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.
    
    Args:
        ranking_text: The full text response from the model
    
    Returns:
        List of response labels in ranked order
    """
    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]
            
            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches
    
    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.
    """
    model_positions = defaultdict(list)
    
    for ranking in stage2_results:
        parsed_ranking = ranking.get("parsed_ranking", [])
        
        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)
    
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })
    
    aggregate.sort(key=lambda x: x["average_rank"])
    return aggregate


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage multi-provider council process.
    
    Args:
        user_query: The user's question
    
    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    logger.info("=" * 60)
    logger.info("Starting Multi-Provider LLM Council (3-stage process)")
    logger.info("=" * 60)
    
    # Stage 1: Collect individual responses
    stage1_results = await stage1_collect_responses(user_query)
    
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please check your API keys."
        }, {}
    
    # Stage 2: Collect rankings
    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results)
    
    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
    
    # Stage 3: Synthesize final answer
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results
    )
    
    # Prepare metadata
    council_model_names = [m["name"] for m in COUNCIL_MODELS]
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "council_models": council_model_names,
        "chairman_model": CHAIRMAN_MODEL["name"]
    }
    
    logger.info("=" * 60)
    logger.info("Multi-Provider LLM Council completed")
    logger.info(f"  - Stage 1 responses: {len(stage1_results)}")
    logger.info(f"  - Stage 2 rankings: {len(stage2_results)}")
    logger.info(f"  - Chairman: {CHAIRMAN_MODEL['name']}")
    logger.info("=" * 60)
    
    return stage1_results, stage2_results, stage3_result, metadata


# Alias for backwards compatibility
run_full_gemini_council = run_full_council


class LLMCouncilAdapter:
    """Adapter for Multi-Provider LLM Council to work with Data Science System"""
    
    def __init__(self, council_backend_path: str = None):
        """
        Initialize LLM Council adapter (now using multiple LLM providers)
        
        Args:
            council_backend_path: Deprecated - no longer used (kept for backwards compatibility)
        """
        self.enabled = True
        
        # Initialize API manager for logging (optional)
        self.api_manager = None
        self.token_logger = None
        self.health_checker = None
        
        if HAS_API_MANAGER:
            mock_config = {
                "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY", ""),
                "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY", ""),
                "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
                "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
            }
            
            try:
                self.api_manager = LLMAPIManager(config=mock_config)
                self.token_logger = TokenLogger()
                self.health_checker = EndpointHealthChecker(api_manager=self.api_manager)
                logger.info("LLM API manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize API manager: {e}")
        
        # Verify Gemini API key is available
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set - LLM Council will use mock responses")
            self.enabled = False
        else:
            logger.info(f"Gemini LLM Council initialized with models: {COUNCIL_MODELS}")
            logger.info(f"Chairman model: {CHAIRMAN_MODEL}")
    
    async def generate_hypotheses_with_council(self, dataset_info: Dict[str, Any],
                                           max_hypotheses: int = 50) -> List[Dict[str, Any]]:
        """
        Generate hypotheses using Gemini LLM Council consensus
        
        Args:
            dataset_info: Dataset information (shape, columns, types, etc.)
            max_hypotheses: Maximum number of hypotheses to generate
        
        Returns:
            List of hypotheses generated by council consensus
        """
        if not self.enabled:
            logger.info("LLM Council disabled, using mock hypotheses")
            return self._generate_mock_hypotheses(dataset_info, max_hypotheses)
        
        logger.info(f"Generating hypotheses with Gemini LLM Council (max: {max_hypotheses})")
        
        # Build prompt for hypothesis generation
        prompt = f"""You are a team of expert data scientists analyzing a dataset.

Dataset Information:
- Shape: {dataset_info.get('shape', 'Unknown')}
- Columns: {dataset_info.get('columns', [])}
- Data Types: {dataset_info.get('dtypes', {})}

Your task is to generate {max_hypotheses} testable hypotheses about this dataset.

For each hypothesis, provide:
1. **Type**: correlation, distribution, categorical, outlier, or trend
2. **Columns involved**: Which columns this hypothesis relates to
3. **Hypothesis statement**: Clear testable statement
4. **Test method**: Statistical test to validate this hypothesis
5. **Reasoning**: Why this hypothesis is worth testing

Generate diverse hypotheses covering different aspects of data. Focus on insights that would be actionable for a business or research context.

Format each hypothesis as a JSON object with these fields: type, columns, hypothesis, test_method, reasoning.

Return all hypotheses as a JSON array. Start your response with [ and end with ].
"""
        
        try:
            # Run Gemini LLM Council
            stage1_results, stage2_results, stage3_result, metadata = await run_full_gemini_council(prompt)
            
            # Parse final synthesis for hypotheses
            hypotheses = self._parse_hypotheses_from_response(
                stage3_result, metadata, max_hypotheses
            )
            
            # If we got too few hypotheses from the final synthesis, also try stage1 responses
            if len(hypotheses) < max_hypotheses // 2:
                for stage1 in stage1_results:
                    additional = self._extract_hypotheses_from_text(
                        stage1.get("response", ""), 
                        max_hypotheses - len(hypotheses)
                    )
                    hypotheses.extend(additional)
                    if len(hypotheses) >= max_hypotheses:
                        break
            
            logger.info(f"Generated {len(hypotheses)} hypotheses using Gemini LLM Council")
            return hypotheses[:max_hypotheses]
            
        except Exception as e:
            logger.error(f"Error in Gemini LLM Council hypothesis generation: {e}")
            return self._generate_mock_hypotheses(dataset_info, max_hypotheses)
    
    def _parse_hypotheses_from_response(self, stage3_result: Dict, metadata: Dict, 
                                       max_hypotheses: int) -> List[Dict[str, Any]]:
        """Parse hypotheses from council's final response"""
        hypotheses = []
        
        if not stage3_result or "response" not in stage3_result:
            return hypotheses
        
        final_response = stage3_result["response"]
        hypotheses = self._extract_hypotheses_from_text(final_response, max_hypotheses)
        
        # Add council metadata to each hypothesis
        for h in hypotheses:
            h["generated_by"] = "gemini_council"
            h["council_models"] = metadata.get("council_models", COUNCIL_MODELS)
        
        return hypotheses
    
    def _extract_hypotheses_from_text(self, text: str, max_count: int) -> List[Dict[str, Any]]:
        """Extract hypotheses from text response"""
        hypotheses = []
        
        # Try to find JSON array
        try:
            # Look for JSON array pattern
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    for i, item in enumerate(data[:max_count]):
                        if isinstance(item, dict):
                            hypotheses.append({
                                "id": f"council_hypothesis_{len(hypotheses) + 1}",
                                "type": item.get("type", self._guess_hypothesis_type(str(item.get("hypothesis", "")))),
                                "columns": item.get("columns", []),
                                "hypothesis": str(item.get("hypothesis", ""))[:500],
                                "test_method": item.get("test_method", "To be determined"),
                                "reasoning": item.get("reasoning", "Generated by Gemini LLM Council")
                            })
        except json.JSONDecodeError:
            pass
        
        # If JSON parsing failed, try to extract individual JSON objects
        if not hypotheses:
            json_objects = re.findall(r'\{[^{}]+\}', text)
            for obj_str in json_objects[:max_count]:
                try:
                    item = json.loads(obj_str)
                    if isinstance(item, dict) and ("hypothesis" in item or "type" in item):
                        hypotheses.append({
                            "id": f"council_hypothesis_{len(hypotheses) + 1}",
                            "type": item.get("type", "general"),
                            "columns": item.get("columns", []),
                            "hypothesis": str(item.get("hypothesis", ""))[:500],
                            "test_method": item.get("test_method", "To be determined"),
                            "reasoning": item.get("reasoning", "Generated by Gemini LLM Council")
                        })
                except json.JSONDecodeError:
                    continue
        
        return hypotheses
    
    def _generate_mock_hypotheses(self, dataset_info: Dict[str, Any], 
                               max_hypotheses: int) -> List[Dict[str, Any]]:
        """Generate mock hypotheses when LLM Council is unavailable"""
        logger.info("Generating mock hypotheses (LLM Council not available)")
        
        # Create realistic mock hypotheses based on dataset info
        columns = dataset_info.get('columns', [])
        dtypes = dataset_info.get('dtypes', {})
        numeric_cols = [col for col, dtype in dtypes.items() if 'int' in dtype or 'float' in dtype]
        categorical_cols = [col for col, dtype in dtypes.items() if dtype == 'object']
        
        mock_hypotheses = []
        
        # Generate correlation hypotheses
        for i, col1 in enumerate(numeric_cols[:min(10, len(numeric_cols)-1)]):
            if i < len(numeric_cols) - 1:
                col2 = numeric_cols[i + 1]
                mock_hypotheses.append({
                    "id": f"mock_hypothesis_{len(mock_hypotheses) + 1}",
                    "type": "correlation",
                    "columns": [col1, col2],
                    "hypothesis": f"Potential correlation exists between {col1} and {col2}",
                    "test_method": "Pearson correlation test",
                    "reasoning": "Based on data types and column analysis, a correlation relationship may exist between these numerical columns"
                })
        
        # Generate distribution hypotheses
        for col in numeric_cols[:5]:
            mock_hypotheses.append({
                "id": f"mock_hypothesis_{len(mock_hypotheses) + 1}",
                "type": "distribution",
                "columns": [col],
                "hypothesis": f"{col} may not follow a normal distribution",
                "test_method": "Shapiro-Wilk normality test",
                "reasoning": f"Categorical analysis of {col} would reveal distribution characteristics"
            })
        
        # Generate categorical hypotheses
        for col in categorical_cols[:5]:
            unique_count = dataset_info.get('shape', [0, 0])[1] // len(categorical_cols) if len(categorical_cols) > 0 else 10
            mock_hypotheses.append({
                "id": f"mock_hypothesis_{len(mock_hypotheses) + 1}",
                "type": "categorical",
                "columns": [col],
                "hypothesis": f"Different values in {col} may have distinct patterns",
                "test_method": "Chi-square test of independence",
                "reasoning": f"With {unique_count} unique values, chi-square tests could reveal associations"
            })
        
        return mock_hypotheses[:max_hypotheses]
    
    def _guess_hypothesis_type(self, text: str) -> str:
        """Guess hypothesis type from text content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['correlation', 'relationship', 'associated', 'linked']):
            return "correlation"
        elif any(word in text_lower for word in ['distribution', 'normal', 'skewed', 'spread']):
            return "distribution"
        elif any(word in text_lower for word in ['category', 'group', 'segment', 'type']):
            return "categorical"
        elif any(word in text_lower for word in ['outlier', 'anomaly', 'extreme', 'unusual']):
            return "outlier"
        elif any(word in text_lower for word in ['trend', 'increase', 'decrease', 'over time', 'temporal', 'time series']):
            return "trend"
        elif any(word in text_lower for word in ['predict', 'forecast', 'model', 'regress']):
            return "model"
        else:
            return "general"
    
    async def generate_insights_with_council(self, analysis_results: Dict[str, Any],
                                         min_insights: int = 50) -> List[Dict[str, Any]]:
        """
        Generate insights using Gemini LLM Council consensus
        
        Args:
            analysis_results: Analysis results (statistical tests, models, etc.)
            min_insights: Minimum number of insights to generate
        
        Returns:
            List of insights generated by council consensus
        """
        if not self.enabled:
            logger.info("LLM Council disabled, using mock insights")
            return self._generate_mock_insights(analysis_results, min_insights)
        
        logger.info(f"Generating insights with Gemini LLM Council (min: {min_insights})")
        
        # Build prompt for insight generation
        statistical_summary = self._summarize_statistical_results(analysis_results)
        
        prompt = f"""You are a team of expert data analysts reviewing analysis results.

Analysis Summary:
{statistical_summary}

Your task is to generate at least {min_insights} actionable insights from these results.

For each insight, provide:
1. **Title**: Short descriptive title
2. **Type**: correlation, distribution, outlier, statistical_test, model_performance, or data_quality
3. **What**: Clear statement of the finding
4. **Why**: Explanation of the underlying reason (data-driven)
5. **How**: Practical application and how to use this insight
6. **Recommendation**: Actionable next steps

Focus on insights that are:
- Actionable for business or research decisions
- Backed by statistical evidence
- Explained in clear, non-technical language
- Have clear "why" and "how" components

Format each insight as a JSON object with these fields: title, type, what, why, how, recommendation.

Return all insights as a JSON array. Start your response with [ and end with ].
"""
        
        try:
            # Run Gemini LLM Council
            stage1_results, stage2_results, stage3_result, metadata = await run_full_gemini_council(prompt)
            
            # Parse final synthesis for insights
            insights = self._parse_insights_from_response(
                stage3_result, metadata, min_insights
            )
            
            # If we got too few insights from the final synthesis, also try stage1 responses
            if len(insights) < min_insights // 2:
                for stage1 in stage1_results:
                    additional = self._extract_insights_from_text(
                        stage1.get("response", ""),
                        min_insights - len(insights)
                    )
                    insights.extend(additional)
                    if len(insights) >= min_insights:
                        break
            
            logger.info(f"Generated {len(insights)} insights using Gemini LLM Council")
            return insights
            
        except Exception as e:
            logger.error(f"Error in Gemini LLM Council insight generation: {e}")
            return self._generate_mock_insights(analysis_results, min_insights)
    
    def _parse_insights_from_response(self, stage3_result: Dict, metadata: Dict,
                                     min_insights: int) -> List[Dict[str, Any]]:
        """Parse insights from council's final response"""
        insights = []
        
        if not stage3_result or "response" not in stage3_result:
            return insights
        
        final_response = stage3_result["response"]
        insights = self._extract_insights_from_text(final_response, min_insights)
        
        # Add council metadata
        for insight in insights:
            insight["generated_by"] = "gemini_council"
            insight["council_models"] = metadata.get("council_models", COUNCIL_MODELS)
        
        return insights
    
    def _extract_insights_from_text(self, text: str, max_count: int) -> List[Dict[str, Any]]:
        """Extract insights from text response"""
        insights = []
        
        # Try to find JSON array
        try:
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    for item in data[:max_count]:
                        if isinstance(item, dict):
                            insights.append({
                                "id": f"council_insight_{len(insights) + 1}",
                                "title": str(item.get("title", "N/A"))[:200],
                                "type": item.get("type", self._guess_insight_type(str(item.get("what", "")))),
                                "what": str(item.get("what", ""))[:500],
                                "why": str(item.get("why", "Generated by Gemini LLM Council"))[:500],
                                "how": str(item.get("how", "Apply findings according to context"))[:500],
                                "recommendation": str(item.get("recommendation", "Use this insight to inform decisions"))[:500]
                            })
        except json.JSONDecodeError:
            pass
        
        # If JSON parsing failed, try individual objects
        if not insights:
            json_objects = re.findall(r'\{[^{}]+\}', text)
            for obj_str in json_objects[:max_count]:
                try:
                    item = json.loads(obj_str)
                    if isinstance(item, dict) and ("title" in item or "what" in item):
                        insights.append({
                            "id": f"council_insight_{len(insights) + 1}",
                            "title": str(item.get("title", "Insight"))[:200],
                            "type": item.get("type", "general"),
                            "what": str(item.get("what", ""))[:500],
                            "why": str(item.get("why", "Generated by Gemini LLM Council"))[:500],
                            "how": str(item.get("how", "Apply according to context"))[:500],
                            "recommendation": str(item.get("recommendation", "Use to inform decisions"))[:500]
                        })
                except json.JSONDecodeError:
                    continue
        
        return insights
    
    def _generate_mock_insights(self, analysis_results: Dict[str, Any], 
                              min_insights: int) -> List[Dict[str, Any]]:
        """Generate mock insights when LLM Council is unavailable"""
        logger.info("Generating mock insights (LLM Council not available)")
        
        insights = []
        
        # Generate insights from statistical tests
        statistical_tests = analysis_results.get("statistical_tests", [])
        significant_tests = [t for t in statistical_tests if t.get("significant", False)]
        
        for i, test in enumerate(significant_tests[:min(30, len(significant_tests))]):
            test_type = test.get("test", "unknown")
            interpretation = test.get("interpretation", "")
            
            insights.append({
                "id": f"mock_insight_{i + 1}",
                "title": f"Significant {test_type} Finding",
                "type": "statistical_test",
                "what": f"{interpretation[:200]}",
                "why": f"Statistical test {test_type} was significant (p < 0.05), indicating a genuine pattern rather than random chance",
                "how": "This finding can be used to validate assumptions and guide further analysis",
                "recommendation": "Consider this result in your interpretation and subsequent analysis steps"
            })
        
        # Generate insights from models
        models = analysis_results.get("models", {}).get("models", {})
        for model_name, model_info in models.items():
            if isinstance(model_info, dict) and "metrics" in model_info:
                metrics = model_info["metrics"]
                test_r2 = metrics.get("test_r2", 0)
                
                if test_r2 > 0.7:
                    insights.append({
                        "id": f"mock_insight_{len(insights) + 1}",
                        "title": f"Strong {model_info.get('model_type', model_name)} Performance",
                        "type": "model_performance",
                        "what": f"{model_name} achieved R² = {test_r2:.3f} on test data",
                        "why": f"High R² score indicates {model_name} explains {test_r2*100:.0f}% of the variance in the target variable",
                        "how": "Use this model for predictions. The high R² suggests good fit to the data.",
                        "recommendation": "Consider using this model for production predictions."
                    })
        
        return insights[:min_insights]
    
    def _summarize_statistical_results(self, results: Dict[str, Any]) -> str:
        """Summarize analysis results for prompt"""
        summary_parts = []
        
        # Statistical tests
        if "statistical_tests" in results:
            tests = results["statistical_tests"]
            significant_tests = [t for t in tests if t.get("significant", False)]
            summary_parts.append(f"- Statistical Tests: {len(tests)} tests performed, {len(significant_tests)} significant")
            
            # Add key findings
            for test in significant_tests[:5]:
                summary_parts.append(f"  - {test.get('test', 'Unknown')}: {test.get('interpretation', '')[:100]}")
        
        # Models
        if "models" in results:
            models = results["models"]
            if "models" in models:
                model_count = len(models["models"])
                summary_parts.append(f"- Predictive Models: {model_count} models built")
                
                # Add best model info
                if models.get("best_model"):
                    best = models["best_model"]
                    if "metrics" in best:
                        metrics = best["metrics"]
                        summary_parts.append(f"  - Best Model: {metrics.get('model_type', 'Unknown')}")
                        summary_parts.append(f"  - Performance: {metrics.get('interpretation', '')[:100]}")
        
        # Correlations
        if "statistical_tests" in results:
            correlations = [t for t in results["statistical_tests"] if t.get('test') == 'correlation']
            if correlations:
                strong_corrs = [c for c in correlations if abs(c.get('correlation_coefficient', 0)) >= 0.5]
                summary_parts.append(f"- Strong Correlations: {len(strong_corrs)} found")
        
        return "\n".join(summary_parts)
    
    def _guess_insight_type(self, text: str) -> str:
        """Guess insight type from text content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['correlation', 'relationship', 'associated', 'linked']):
            return "correlation"
        elif any(word in text_lower for word in ['distribution', 'normal', 'skew']):
            return "distribution"
        elif any(word in text_lower for word in ['outlier', 'anomaly', 'unusual']):
            return "outlier"
        elif any(word in text_lower for word in ['model', 'predict', 'forecast', 'accuracy', 'performance']):
            return "model_performance"
        elif any(word in text_lower for word in ['missing', 'data quality', 'clean']):
            return "data_quality"
        elif any(word in text_lower for word in ['significant', 'test', 'p-value', 'statistic']):
            return "statistical_test"
        else:
            return "general"
    
    async def rank_models_with_council(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Gemini LLM Council to rank and select best models
        
        Args:
            model_results: Dictionary with model names and their metrics
        
        Returns:
            Dictionary with ranking and recommendation
        """
        if not self.enabled:
            logger.info("LLM Council disabled, returning empty ranking")
            return {"council_used": False}
        
        logger.info("Ranking models with Gemini LLM Council consensus")
        
        # Build model summary for prompt
        model_summary = "\n".join([
            f"- {name}: {metrics.get('model_type', name)}\n  Metrics: {metrics}"
            for name, metrics in model_results.items()
        ])
        
        prompt = f"""You are a team of machine learning experts evaluating predictive models.

Models Evaluated:
{model_summary}

Your task is to:
1. Evaluate each model's performance based on metrics
2. Rank models from best to worst
3. Provide a recommendation for which model to use

Consider:
- Accuracy/performance metrics (R², accuracy, F1-score, etc.)
- Model complexity and interpretability
- Training time and computational cost
- Suitability for use case (real-time, batch, interpretability vs accuracy tradeoff)

Provide your ranking as:
1. A numbered list from best (1) to worst
2. A final recommendation with justification

Format:
EVALUATION:
[Your detailed evaluation of each model]

FINAL RANKING:
1. [Model Name]
2. [Model Name]
...

RECOMMENDATION:
[Model name] with [justification]
"""
        
        try:
            # Run Gemini LLM Council
            stage1_results, stage2_results, stage3_result, metadata = await run_full_gemini_council(prompt)
            
            result = {
                "council_used": True,
                "individual_evaluations": stage1_results,
                "peer_rankings": stage2_results,
                "final_synthesis": stage3_result,
                "ranking": [],
                "recommendation": {},
                "metadata": metadata
            }
            
            if stage3_result and "response" in stage3_result:
                response_text = stage3_result["response"]
                
                # Parse ranking
                ranking = parse_ranking_from_text(response_text)
                result["ranking"] = ranking
                
                # Parse recommendation
                if "RECOMMENDATION:" in response_text:
                    rec_section = response_text.split("RECOMMENDATION:")[-1]
                    
                    # Try to extract model name
                    for model_name in model_results.keys():
                        if model_name.lower() in rec_section.lower():
                            result["recommendation"] = {
                                "model": model_name,
                                "justification": rec_section[:500].strip()
                            }
                            break
            
            logger.info(f"Model ranking completed: {len(result['ranking'])} items ranked")
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini LLM Council model ranking: {e}")
            return {"council_used": False, "error": str(e)}
            
        except Exception as e:
            logger.error(f"Error in LLM Council model ranking: {e}")
            return {"council_used": False, "error": str(e)}
    
    def enable(self):
        """Enable LLM Council"""
        self.enabled = True
        logger.info("LLM Council enabled")
    
    def disable(self):
        """Disable LLM Council"""
        self.enabled = False
        logger.info("LLM Council disabled")
    
    def is_enabled(self) -> bool:
        """Check if LLM Council is enabled"""
        return self.enabled


class EnhancedAnalysisPipeline:
    """Enhanced analysis pipeline with LLM Council integration"""
    
    def __init__(self, dataset_path: str, use_council: bool = True,
                 council_backend_path: str = None):
        """
        Initialize enhanced pipeline with LLM Council integration
        
        Args:
            dataset_path: Path to dataset
            use_council: Whether to use LLM Council for consensus
            council_backend_path: Path to llm-council backend
        """
        # Import pipeline components
        from workflow import AnalysisPipeline
        from analysis_engine import LLMCouncilAdapter
        
        # Base pipeline
        self.base_pipeline = AnalysisPipeline(dataset_path)
        
        # LLM Council integration
        self.use_council = use_council
        self.council_adapter = LLMCouncilAdapter(council_backend_path)
        
        logger.info(f"Enhanced pipeline initialized (Council: {use_council})")
    
    @property
    def dataset_name(self):
        return self.base_pipeline.dataset_name
    
    @property
    def output_dir(self):
        return self.base_pipeline.output_dir
    
    async def generate_hypotheses_async(self, max_hypotheses: int = 100) -> List[Dict[str, Any]]:
        """
        Generate hypotheses (with or without LLM Council)
        
        Args:
            max_hypotheses: Maximum hypotheses to generate
        
        Returns:
            List of hypotheses
        """
        if self.use_council:
            # Use LLM Council for consensus-based hypothesis generation
            dataset_info = self.base_pipeline.results.get('dataset_info', {})
            return await self.council_adapter.generate_hypotheses_with_council(
                dataset_info, max_hypotheses
            )
        else:
            # Use Agentic approach with tools
            from agents import create_hypothesis_generator_agent
            import json
            
            agent = create_hypothesis_generator_agent()
            dataset_info = self.base_pipeline.results.get('dataset_info', {})
            
            prompt = f"""Analyze this dataset and generate {max_hypotheses} testable hypotheses.
            Dataset Info: {dataset_info}
            
            Use your tools to explore the data if needed.
            Return the hypotheses as a JSON list of objects with fields: type, columns, hypothesis, test_method, reasoning.
            """
            
            try:
                result = agent.run(prompt)
                # Parse result if it's a string containing JSON
                if isinstance(result, str):
                    import re
                    json_match = re.search(r'\[.*\]', result, re.DOTALL)
                    if json_match:
                        hypotheses = json.loads(json_match.group())
                        return hypotheses
                elif isinstance(result, list):
                    return result
            except Exception as e:
                logger.warning(f"Agentic hypothesis generation failed: {e}. Falling back to procedural method.")
            
            # Fallback to single LLM (original procedural behavior)
            from analysis_engine import HypothesisGenerator
            
            generator = HypothesisGenerator(self.base_pipeline.df)
            return generator.generate_all_hypotheses(max_hypotheses)
    
    async def extract_insights_async(self, min_insights: int = 50) -> List[Dict[str, Any]]:
        """
        Extract insights (with or without LLM Council)
        
        Args:
            min_insights: Minimum insights to generate
        
        Returns:
            List of insights
        """
        if self.use_council:
            # Use LLM Council for consensus-based insight generation
            return await self.council_adapter.generate_insights_with_council(
                self.base_pipeline.results, min_insights
            )
        else:
            # Use Agentic approach with tools
            from agents import create_analyzer_agent
            import json
            
            agent = create_analyzer_agent()
            analysis_results = {
                'statistical_tests': self.base_pipeline.results['statistical_tests'],
                'modeling': self.base_pipeline.results.get('models', {})
            }
            
            prompt = f"""Review these analysis results and extract at least {min_insights} actionable insights.
            Results: {analysis_results}
            
            Use your tools to further explore the data if needed.
            Return the insights as a JSON list of objects with fields: title, type, what, why, how, recommendation.
            """
            
            try:
                result = agent.run(prompt)
                if isinstance(result, str):
                    import re
                    json_match = re.search(r'\[.*\]', result, re.DOTALL)
                    if json_match:
                        insights = json.loads(json_match.group())
                        return insights
                elif isinstance(result, list):
                    return result
            except Exception as e:
                logger.warning(f"Agentic insight extraction failed: {e}. Falling back to procedural method.")
            
            # Fallback to single LLM (original procedural behavior)
            from analysis_engine import InsightExtractor
            
            extractor = InsightExtractor(self.base_pipeline.df)
            
            analysis_results = {
                'correlations': [r for r in self.base_pipeline.results['statistical_tests'] 
                               if r.get('test') == 'correlation'],
                'distributions': [r for r in self.base_pipeline.results['statistical_tests'] 
                                 if r.get('test') == 'normality'],
                'outliers': [r for r in self.base_pipeline.results['statistical_tests'] 
                             if r.get('test') == 'outliers'],
                'statistical_tests': self.base_pipeline.results['statistical_tests'],
                'modeling': self.base_pipeline.results.get('models', {})
            }
            
            return extractor.generate_all_insights(analysis_results)
    
    async def rank_models_async(self) -> Dict[str, Any]:
        """
        Rank models using LLM Council consensus
        
        Returns:
            Dictionary with ranking and recommendation
        """
        if self.use_council and self.base_pipeline.results.get('models'):
            models = self.base_pipeline.results['models'].get('models', {})
            return await self.council_adapter.rank_models_with_council(models)
        else:
            return {"council_used": False}
    
    async def run_full_pipeline_with_council(self, target_column: str = None,
                                          generate_word: bool = True) -> Dict[str, Any]:
        """
        Run full pipeline with LLM Council integration
        
        Args:
            target_column: Target variable for modeling
            generate_word: Whether to generate Word document
        
        Returns:
            Dictionary with all results including council metadata
        """
        logger.info("Running full pipeline with LLM Council integration")
        
        # Run base pipeline steps
        self.base_pipeline.load_data()
        self.base_pipeline.clean_data()
        
        # Generate hypotheses with council
        hypotheses = await self.generate_hypotheses_async(max_hypotheses=100)
        self.base_pipeline.results['hypotheses'] = hypotheses
        self.base_pipeline.results['used_council_for_hypotheses'] = self.use_council
        
        # Initialize and update heuristic generator for report formatting
        from analysis_engine import HypothesisGenerator
        self.base_pipeline.heuristic_generator = HypothesisGenerator(self.base_pipeline.df)
        self.base_pipeline.heuristic_generator.hypotheses = hypotheses
        
        # Run statistical tests
        self.base_pipeline.run_statistical_tests()
        
        # Build models
        self.base_pipeline.build_models(target_column)
        
        # Rank models with council
        model_ranking = await self.rank_models_async()
        self.base_pipeline.results['model_ranking'] = model_ranking
        
        # Extract insights with council
        insights = await self.extract_insights_async(min_insights=50)
        self.base_pipeline.results['insights'] = insights
        self.base_pipeline.results['used_council_for_insights'] = self.use_council
        
        # Initialize and update insight extractor for report formatting
        from analysis_engine import InsightExtractor
        self.base_pipeline.insight_extractor = InsightExtractor(self.base_pipeline.df)
        self.base_pipeline.insight_extractor.insights = insights
        
        # Create visualizations
        self.base_pipeline.create_visualizations()
        
        # Generate reports
        self.base_pipeline.generate_reports(formats=['markdown', 'word'] if generate_word else ['markdown'])
        
        # Save execution log
        self.base_pipeline.save_execution_log()
        
        # Log token usage if available
        if HAS_API_MANAGER and self.council_adapter.api_manager:
            total_cost = self.council_adapter.api_manager.get_total_cost()
            logger.info(f"Total estimated API cost: ${total_cost:.6f}")
        
        return self.base_pipeline.results

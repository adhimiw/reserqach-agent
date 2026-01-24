import asyncio
import logging
import json
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("CouncilBackend")

async def run_full_council(prompt: str) -> Tuple[List[Dict], List[Dict], Dict, Dict]:
    """
    Mock implementation of LLM Council for testing integration
    """
    logger.info("Running Mock LLM Council backend")
    
    # Return fallback JSON based on prompt
    if "hypotheses" in prompt.lower():
        content = json.dumps([
            {"type": "correlation", "columns": ["age", "income"], "hypothesis": "Age and income are correlated", "test_method": "pearson", "reasoning": "Standard assumption"},
            {"type": "distribution", "columns": ["monthly_charge"], "hypothesis": "Monthly charges are non-normal", "test_method": "shapiro", "reasoning": "Likely contains outliers"},
            {"type": "categorical", "columns": ["gender", "churn"], "hypothesis": "Churn differs by gender", "test_method": "chi-square", "reasoning": "Sociodemographic factor"}
        ])
    else:
        # Insights
        content = json.dumps([
            {"title": "Churn Correlation", "type": "correlation", "what": "High monthly charges correlate with churn", "why": "Cost pressure", "how": "Offer discounts", "recommendation": "Retain high-value customers"},
            {"title": "Age Factor", "type": "distribution", "what": "Older customers are more loyal", "why": "Stability", "how": "Target older demographics", "recommendation": "Focus marketing on 40+"},
            {"title": "Mixed Data Quality", "type": "data_quality", "what": "Mixed types found in mixed_column", "why": "Data entry issues", "how": "Clean data before analysis", "recommendation": "Standardize input fields"}
        ])

    stage1_results = [{"model": "mock_model_a", "response": content}]
    stage2_results = [{"model": "mock_model_b", "ranking": "Ranking"}]
    stage3_result = {"model": "mock_chairman", "response": content}
    metadata = {
        "duration_ms": 500,
        "label_to_model": {"Response A": "mock_model_a"}
    }
    
    return stage1_results, stage2_results, stage3_result, metadata

def parse_ranking_from_text(text: str) -> List[str]:
    return ["Random Forest", "Gradient Boosting"]

"""
DSPy Signatures for Research Agent Tools
Optimized prompts and signatures for tool usage
"""

import dspy
from typing import List, Optional


class WebSearchSignature(dspy.Signature):
    """Generate optimal search query for web research"""
    
    topic: str = dspy.InputField(desc="Research topic or question")
    context: Optional[str] = dspy.InputField(desc="Additional context or constraints", default="")
    recency_required: bool = dspy.InputField(desc="Whether recent information is needed", default=True)
    
    search_query: str = dspy.OutputField(desc="Optimized search query")
    recency_filter: str = dspy.OutputField(desc="Recency filter: day, week, month, or year")
    reasoning: str = dspy.OutputField(desc="Why this query will find relevant information")


class ContentExtractionSignature(dspy.Signature):
    """Extract key information from search results"""
    
    raw_content: str = dspy.InputField(desc="Raw search result or web page content")
    extraction_goal: str = dspy.InputField(desc="What information to extract")
    
    extracted_facts: List[str] = dspy.OutputField(desc="List of key facts extracted")
    sources: List[str] = dspy.OutputField(desc="Source URLs or references")
    confidence: float = dspy.OutputField(desc="Confidence score (0-1) in the information")


class ResearchSynthesisSignature(dspy.Signature):
    """Synthesize research findings into coherent content"""
    
    research_data: List[str] = dspy.InputField(desc="Collection of research findings")
    topic: str = dspy.InputField(desc="Main research topic")
    writing_style: str = dspy.InputField(desc="Target writing style: academic, technical, or general")
    
    synthesized_content: str = dspy.OutputField(desc="Coherent synthesis of research")
    key_points: List[str] = dspy.OutputField(desc="Main points covered")
    gaps: List[str] = dspy.OutputField(desc="Information gaps that need more research")


class FactCheckSignature(dspy.Signature):
    """Verify factual claims against sources"""
    
    claim: str = dspy.InputField(desc="Factual claim to verify")
    sources: List[str] = dspy.InputField(desc="Available source materials")
    
    is_verified: bool = dspy.OutputField(desc="Whether the claim is verified")
    confidence: float = dspy.OutputField(desc="Confidence in verification (0-1)")
    supporting_evidence: str = dspy.OutputField(desc="Evidence supporting the verification")
    contradictions: Optional[str] = dspy.OutputField(desc="Any contradicting information found")


class DocumentStructureSignature(dspy.Signature):
    """Generate optimal document structure"""
    
    topic: str = dspy.InputField(desc="Document topic")
    content_summary: str = dspy.InputField(desc="Summary of available content")
    style: str = dspy.InputField(desc="Document style: academic, technical, or general")
    
    outline: List[str] = dspy.OutputField(desc="Document outline with sections")
    key_sections: List[str] = dspy.OutputField(desc="Critical sections that must be included")
    estimated_length: str = dspy.OutputField(desc="Estimated document length")


class QueryRefinementSignature(dspy.Signature):
    """Refine search query based on previous results"""
    
    original_query: str = dspy.InputField(desc="Original search query")
    previous_results: str = dspy.InputField(desc="Summary of previous search results")
    missing_info: str = dspy.InputField(desc="What information is still needed")
    
    refined_query: str = dspy.OutputField(desc="Refined search query")
    search_strategy: str = dspy.OutputField(desc="Strategy for finding missing information")
    alternative_queries: List[str] = dspy.OutputField(desc="Alternative query formulations")


class CitationGenerationSignature(dspy.Signature):
    """Generate proper citations from sources"""
    
    source_url: str = dspy.InputField(desc="Source URL or reference")
    source_content: str = dspy.InputField(desc="Relevant content from source")
    citation_style: str = dspy.InputField(desc="Citation style: APA, MLA, Chicago, etc.")
    
    citation: str = dspy.OutputField(desc="Properly formatted citation")
    in_text_citation: str = dspy.OutputField(desc="In-text citation format")
    reliability_score: float = dspy.OutputField(desc="Source reliability (0-1)")


# ============== DSPy Modules ==============

class OptimizedSearchModule(dspy.Module):
    """Module for optimized web searching"""
    
    def __init__(self):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(WebSearchSignature)
    
    def forward(self, topic: str, context: str = "", recency_required: bool = True):
        """Generate optimized search query"""
        return self.generate_query(
            topic=topic,
            context=context,
            recency_required=recency_required
        )


class ResearchSynthesisModule(dspy.Module):
    """Module for synthesizing research findings"""
    
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(ResearchSynthesisSignature)
    
    def forward(self, research_data: List[str], topic: str, writing_style: str = "academic"):
        """Synthesize research into coherent content"""
        return self.synthesize(
            research_data=research_data,
            topic=topic,
            writing_style=writing_style
        )


class FactCheckModule(dspy.Module):
    """Module for fact-checking claims"""
    
    def __init__(self):
        super().__init__()
        self.verify = dspy.ChainOfThought(FactCheckSignature)
    
    def forward(self, claim: str, sources: List[str]):
        """Verify a factual claim"""
        return self.verify(claim=claim, sources=sources)


class AdaptiveResearchModule(dspy.Module):
    """Advanced module that adapts search strategy based on results"""
    
    def __init__(self):
        super().__init__()
        self.initial_search = dspy.ChainOfThought(WebSearchSignature)
        self.refine_query = dspy.ChainOfThought(QueryRefinementSignature)
        self.extract_content = dspy.ChainOfThought(ContentExtractionSignature)
    
    def forward(self, topic: str, max_iterations: int = 3):
        """
        Perform adaptive research with iterative refinement
        
        Args:
            topic: Research topic
            max_iterations: Maximum search refinement iterations
            
        Returns:
            Comprehensive research results
        """
        # Initial search
        search_result = self.initial_search(topic=topic, context="", recency_required=True)
        
        all_findings = []
        query = search_result.search_query
        
        for i in range(max_iterations):
            # This would integrate with actual search tools
            # For now, it's the DSPy structure
            extraction = self.extract_content(
                raw_content=f"Search results for: {query}",
                extraction_goal=topic
            )
            
            all_findings.extend(extraction.extracted_facts)
            
            # Check if we need to refine
            if len(extraction.extracted_facts) < 3:  # Arbitrary threshold
                refined = self.refine_query(
                    original_query=query,
                    previous_results=str(all_findings),
                    missing_info=topic
                )
                query = refined.refined_query
        
        return {
            "findings": all_findings,
            "final_query": query,
            "iterations": i + 1
        }


# ============== Optimizer Configuration ==============

def configure_dspy(model_provider="perplexity"):
    """
    Configure DSPy with the appropriate LLM
    
    Args:
        model_provider: Which model provider to use for DSPy
    """
    from config import Config, ModelProvider
    
    if not Config.DSPY_ENABLED:
        return None
    
    # Get model based on provider
    if model_provider == "perplexity":
        provider = ModelProvider.PERPLEXITY
    elif model_provider == "mistral":
        provider = ModelProvider.MISTRAL
    elif model_provider == "smithery":
        provider = ModelProvider.SMITHERY
    else:
        provider = ModelProvider.PERPLEXITY
    
    model = Config.get_model(provider=provider)
    
    # Configure DSPy
    dspy.settings.configure(lm=model, cache_dir=Config.DSPY_CACHE_DIR)
    
    return model


def create_optimized_modules():
    """
    Create and return all DSPy modules
    
    Returns:
        dict: Dictionary of initialized DSPy modules
    """
    return {
        "search": OptimizedSearchModule(),
        "synthesis": ResearchSynthesisModule(),
        "fact_check": FactCheckModule(),
        "adaptive_research": AdaptiveResearchModule()
    }

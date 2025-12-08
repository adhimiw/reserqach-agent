import dspy
from typing import List, Dict, Optional
from smolagents import CodeAgent, LiteLLMModel, tool
from config import Config
from signatures.academic import WriteAcademicSection, GenerateOutline
from tools.token_optimizer import TokenOptimizer, optimize_text_for_llm

# Initialize token optimizer
token_optimizer = TokenOptimizer()

def setup_dspy():
    # Configure DSPy to use Perplexity (OpenAI-compatible endpoint)
    # We use dspy.LM with the openai provider but point to Perplexity
    lm = dspy.LM(
        model="openai/" + Config.PERPLEXITY_MODEL,
        api_key=Config.PERPLEXITY_API_KEY,
        api_base=Config.LLM_API_BASE
    )
    dspy.configure(lm=lm)

@tool
def generate_academic_text(section_title: str, context: str, notes: str) -> str:
    """
    Generates high-quality academic text using DSPy optimization with token reduction.
    
    Args:
        section_title: The title of the section.
        context: What this section should cover.
        notes: Raw research notes and citations.
        
    Returns:
        The generated text content.
    """
    # Ensure DSPy is configured
    setup_dspy()
    
    # Optimize notes for token efficiency
    optimized_notes = token_optimizer.optimize_prompt(notes)
    
    # Create the module
    writer = dspy.ChainOfThought(WriteAcademicSection)
    
    # Execute
    prediction = writer(
        section_title=section_title,
        section_context=context,
        research_notes=optimized_notes
    )
    
    return prediction.content

@tool
def generate_multi_section_outline(topic: str, sections: List[str]) -> Dict[str, str]:
    """
    Generate outlines for multiple sections collaboratively.
    Uses multi-agent approach for each section type.
    
    Args:
        topic: Main research topic
        sections: List of section titles
        
    Returns:
        Dictionary with section outlines
    """
    setup_dspy()
    
    outlines = {}
    outline_generator = dspy.ChainOfThought(GenerateOutline)
    
    for section in sections:
        prompt = f"Create detailed outline for '{section}' in context of {topic}"
        optimized = optimize_text_for_llm(prompt)
        
        prediction = outline_generator(section_title=section, topic=topic)
        outlines[section] = prediction.outline
    
    return outlines

@tool
def synthesize_research_findings(findings: Dict[str, str], topic: str, 
                                writing_style: str = "academic") -> str:
    """
    Synthesize multiple research findings into coherent narrative.
    Implements synthesis agent pattern.
    
    Args:
        findings: Dictionary of research findings by topic
        topic: Main research topic
        writing_style: Academic, technical, general, etc.
        
    Returns:
        Synthesized text
    """
    setup_dspy()
    
    # Combine findings with optimization
    combined_findings = "\n".join([f"{k}: {v}" for k, v in findings.items()])
    optimized = token_optimizer.optimize_prompt(combined_findings)
    
    prompt = f"""
    Synthesize these research findings about '{topic}' into a coherent {writing_style} narrative:
    
    {optimized}
    
    Create flowing text that integrates these findings naturally.
    """
    
    writer = dspy.ChainOfThought(WriteAcademicSection)
    prediction = writer(
        section_title="Synthesis",
        section_context=topic,
        research_notes=optimized
    )
    
    return prediction.content

@tool
def generate_literature_review(sources: List[Dict[str, str]], topic: str) -> str:
    """
    Generate literature review from multiple sources.
    Multi-agent approach to synthesize sources.
    
    Args:
        sources: List of source dictionaries with 'title', 'author', 'year', 'summary'
        topic: Research topic
        
    Returns:
        Formatted literature review
    """
    setup_dspy()
    
    # Organize sources by theme
    sources_text = "\n".join([
        f"- {s.get('title', 'Unknown')} by {s.get('author', 'Unknown')} ({s.get('year', 'N/A')}): {s.get('summary', '')}"
        for s in sources
    ])
    
    optimized = token_optimizer.optimize_prompt(sources_text)
    
    prompt = f"""
    Create a comprehensive literature review on '{topic}' synthesizing these sources:
    
    {optimized}
    
    Structure: Overview, Key themes, Critical analysis, Gaps and future directions.
    """
    
    writer = dspy.ChainOfThought(WriteAcademicSection)
    prediction = writer(
        section_title="Literature Review",
        section_context=topic,
        research_notes=optimized
    )
    
    return prediction.content

@tool
def optimize_for_clarity_and_conciseness(text: str, max_words: int = 500) -> str:
    """
    Optimize generated text for clarity and conciseness while maintaining quality.
    
    Args:
        text: Text to optimize
        max_words: Maximum words in output
        
    Returns:
        Optimized text
    """
    setup_dspy()
    
    prompt = f"""
    Optimize this text for maximum clarity and conciseness (target: {max_words} words):
    
    {text}
    
    Keep all essential information but remove redundancy and improve flow.
    """
    
    optimized = token_optimizer.optimize_prompt(text)
    
    writer = dspy.ChainOfThought(WriteAcademicSection)
    prediction = writer(
        section_title="Optimized Text",
        section_context="Clarity and conciseness",
        research_notes=optimized
    )
    
    return prediction.content

def create_writer_agent(tools, model=None):
    """
    Creates the Writer Agent responsible for drafting the paper in Word.
    Supports multi-agent collaboration through shared tools.
    """
    if model is None:
        model = LiteLLMModel(
            model_id=Config.LLM_MODEL_ID, 
            api_key=Config.PERPLEXITY_API_KEY,
            api_base=Config.LLM_API_BASE,
            drop_params=True
        )

    # Add all advanced writing tools
    all_tools = tools + [
        generate_academic_text,
        generate_multi_section_outline,
        synthesize_research_findings,
        generate_literature_review,
        optimize_for_clarity_and_conciseness
    ]

    return CodeAgent(
        tools=all_tools,
        model=model,
        name="writer",
        description="Multi-agent writer that generates academic content with token optimization and collaboration support.",
        additional_authorized_imports=["dspy", "signatures.academic", "tools.token_optimizer"]
    )

def create_collaborative_writer_system(tools_by_agent: Dict[str, List],
                                      model: Optional[LiteLLMModel] = None):
    """
    Create a collaborative multi-agent writing system.
    
    Args:
        tools_by_agent: Dictionary mapping agent roles to their tools
        model: Optional model to use (defaults to Perplexity)
        
    Returns:
        Dictionary of specialized writer agents
    """
    if model is None:
        model = LiteLLMModel(
            model_id=Config.LLM_MODEL_ID,
            api_key=Config.PERPLEXITY_API_KEY,
            api_base=Config.LLM_API_BASE,
            drop_params=True
        )
    
    writers = {}
    
    # Primary writer (all tools)
    writers['primary'] = create_writer_agent(
        tools_by_agent.get('primary', []),
        model
    )
    
    # Content synthesizer (specializes in combining findings)
    synthesizer_tools = [synthesize_research_findings, generate_literature_review]
    writers['synthesizer'] = CodeAgent(
        tools=synthesizer_tools + tools_by_agent.get('synthesizer', []),
        model=model,
        name="synthesizer",
        description="Specializes in synthesizing and integrating research findings."
    )
    
    # Content optimizer (focuses on clarity and conciseness)
    optimizer_tools = [optimize_for_clarity_and_conciseness]
    writers['optimizer'] = CodeAgent(
        tools=optimizer_tools + tools_by_agent.get('optimizer', []),
        model=model,
        name="optimizer",
        description="Specializes in optimizing text for clarity and token efficiency."
    )
    
    return writers


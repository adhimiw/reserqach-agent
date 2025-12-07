import dspy
from smolagents import CodeAgent, LiteLLMModel, tool
from config import Config
from signatures.academic import WriteAcademicSection, GenerateOutline

def setup_dspy():
    # Configure DSPy to use Perplexity (OpenAI-compatible endpoint)
    # We use dspy.LM with the openai provider but point to Perplexity
    lm = dspy.LM(
        model="openai/" + Config.PERPLEXITY_MODEL, # dspy often expects 'provider/model' or just model if configured
        api_key=Config.PERPLEXITY_API_KEY,
        api_base=Config.LLM_API_BASE
    )
    dspy.configure(lm=lm)

@tool
def generate_academic_text(section_title: str, context: str, notes: str) -> str:
    """
    Generates high-quality academic text using DSPy optimization.
    
    Args:
        section_title: The title of the section.
        context: What this section should cover.
        notes: Raw research notes and citations.
        
    Returns:
        The generated text content.
    """
    # Ensure DSPy is configured
    setup_dspy()
    
    # Create the module
    writer = dspy.ChainOfThought(WriteAcademicSection)
    
    # Execute
    prediction = writer(
        section_title=section_title,
        section_context=context,
        research_notes=notes
    )
    
    return prediction.content

def create_writer_agent(tools, model=None):
    """
    Creates the Writer Agent responsible for drafting the paper in Word.
    """
    if model is None:
        model = LiteLLMModel(
            model_id=Config.LLM_MODEL_ID, 
            api_key=Config.PERPLEXITY_API_KEY,
            api_base=Config.LLM_API_BASE,
            drop_params=True
        )

    # Add the DSPy tool to the list of tools
    all_tools = tools + [generate_academic_text]

    return CodeAgent(
        tools=all_tools,
        model=model,
        name="writer",
        description="An agent that writes academic content and saves it to a Word document.",
        additional_authorized_imports=["dspy", "signatures.academic"]
    )

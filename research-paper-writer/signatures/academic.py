import dspy

class GenerateOutline(dspy.Signature):
    """Generate a detailed academic research paper outline based on a topic."""
    
    topic = dspy.InputField(desc="The main topic of the research paper")
    context = dspy.InputField(desc="Any specific context or requirements", optional=True)
    
    outline = dspy.OutputField(desc="Structured outline with sections and subsections")
    research_questions = dspy.OutputField(desc="List of key research questions to address")

class WriteAcademicSection(dspy.Signature):
    """Write a detailed, academic section based on provided research notes.
    Ensure the tone is formal, objective, and cites sources."""
    
    section_title = dspy.InputField(desc="The title of the section")
    section_context = dspy.InputField(desc="What this section should cover")
    research_notes = dspy.InputField(desc="Raw notes, facts, and citations from research")
    citation_style = dspy.InputField(desc="Citation style (e.g., APA, IEEE)", default="APA")
    
    content = dspy.OutputField(desc="The written section text, formatted with markdown")
    citations_used = dspy.OutputField(desc="List of citations actually used in the text")

class VerifyClaim(dspy.Signature):
    """Verify if a specific claim is supported by the source text."""
    
    claim = dspy.InputField(desc="The claim to verify")
    source_text = dspy.InputField(desc="The text content from the source URL")
    
    verdict = dspy.OutputField(desc="TRUE, FALSE, or UNSUPPORTED")
    explanation = dspy.OutputField(desc="Reasoning for the verdict")
    relevant_quote = dspy.OutputField(desc="Exact quote from source supporting the verdict")

class RefineText(dspy.Signature):
    """Refine the given text to reduce hallucinations and improve academic tone."""
    
    draft_text = dspy.InputField(desc="The initial draft text")
    critique = dspy.InputField(desc="Specific feedback or issues to address")
    
    refined_text = dspy.OutputField(desc="The improved text")

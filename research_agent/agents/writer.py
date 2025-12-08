"""
Writer and Editor Agents - Document generation with style-aware formatting
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from smolagents import Tool
from config import Config, AgentRole
from .base_agent import BaseResearchAgent, AgentWithMemory

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Structured document section"""
    title: str
    content: str
    level: int  # Heading level
    subsections: List['DocumentSection'] = None


class WriterAgent(AgentWithMemory):
    """
    Specialized agent for writing research documents
    Uses Claude/Smithery for highest quality writing
    """
    
    def __init__(
        self,
        tools: List[Tool],
        model=None,
        name: str = "writer",
        default_style: str = "academic"
    ):
        """
        Initialize writer agent
        
        Args:
            tools: Writing tools (Word MCP, filesystem)
            model: LLM model (defaults to Smithery/Claude for writing quality)
            name: Agent name
            default_style: Default writing style
        """
        super().__init__(
            name=name,
            role=AgentRole.WRITER,
            tools=tools,
            model=model,
            use_code_agent=True
        )
        self.default_style = default_style
    
    def _get_description(self) -> str:
        """Get agent description"""
        return """Expert writing agent specializing in:
        - Research paper composition with academic rigor
        - Technical documentation with clarity and precision
        - Structured document creation (abstracts, introductions, conclusions)
        - Citation integration and bibliography management
        - Style-aware formatting (academic, technical, general)
        
        Produces well-structured, coherent, and professionally formatted documents."""
    
    def _get_authorized_imports(self) -> List[str]:
        """Get authorized imports"""
        return ["json", "datetime", "docx", "os", "typing"]
    
    async def write_section(
        self,
        title: str,
        content_brief: str,
        style: str = None,
        word_count: int = 500
    ) -> str:
        """
        Write a document section
        
        Args:
            title: Section title
            content_brief: Brief description of what to write
            style: Writing style (defaults to agent's default_style)
            word_count: Target word count
            
        Returns:
            Written section content
        """
        style = style or self.default_style
        style_config = Config.WRITING_STYLES.get(style, Config.WRITING_STYLES["academic"])
        
        task = f"""Write a {style} {title} section.
        
        Content requirements:
        {content_brief}
        
        Style guidelines:
        - Formality: {style_config.get('formality', 'high')}
        - Voice: {style_config.get('voice', 'third_person')}
        - Tone: {style_config.get('tone', 'objective')}
        - Target length: approximately {word_count} words
        
        Write the complete section now. Be comprehensive, well-structured, and maintain consistent style throughout.
        """
        
        result = await self.run(task)
        return str(result)
    
    async def create_document(
        self,
        topic: str,
        research_data: Dict[str, Any],
        filename: str,
        style: str = None
    ) -> str:
        """
        Create complete research document
        
        Args:
            topic: Document topic
            research_data: Research findings to incorporate
            filename: Output filename
            style: Writing style
            
        Returns:
            Path to created document
        """
        style = style or self.default_style
        
        task = f"""Create a comprehensive research document on: {topic}
        
        Using this research data:
        {research_data}
        
        Document requirements:
        - Style: {style}
        - Include: Title, Abstract, Introduction, Main Body (with subsections), Conclusion, References
        - Properly cite all sources from research data
        - Use the Word MCP tool to create the document
        - Save as: {filename}
        
        Follow {style} writing conventions throughout.
        """
        
        result = await self.run(task)
        self.memory.store_artifact("last_document", filename)
        return str(result)


class EditorAgent(AgentWithMemory):
    """
    Specialized agent for editing and refining documents
    Reviews for clarity, coherence, grammar, and style consistency
    """
    
    def __init__(self, tools: List[Tool], model=None, name: str = "editor"):
        """Initialize editor agent"""
        super().__init__(
            name=name,
            role=AgentRole.EDITOR,
            tools=tools,
            model=model,
            use_code_agent=True
        )
    
    def _get_description(self) -> str:
        """Get agent description"""
        return """Expert editor specializing in:
        - Document review and refinement
        - Grammar, clarity, and coherence improvement
        - Style consistency enforcement
        - Citation and reference verification
        - Structural improvements and reorganization
        
        Ensures documents meet professional standards."""
    
    def _get_authorized_imports(self) -> List[str]:
        """Get authorized imports"""
        return ["json", "datetime", "re", "typing"]
    
    async def review_document(
        self,
        content: str,
        style: str = "academic"
    ) -> Dict[str, Any]:
        """
        Review document and provide feedback
        
        Args:
            content: Document content to review
            style: Expected style
            
        Returns:
            Review feedback with suggestions
        """
        task = f"""Review this document for a {style} audience:
        
        {content}
        
        Provide feedback on:
        1. Clarity and coherence
        2. Grammar and syntax
        3. Style consistency with {style} standards
        4. Structural organization
        5. Citation quality
        
        Format your response as:
        STRENGTHS:
        - [strength 1]
        - [strength 2]
        
        ISSUES:
        - [issue 1]: [suggestion]
        - [issue 2]: [suggestion]
        
        OVERALL_SCORE: [0-10]
        """
        
        result = await self.run(task)
        return self._parse_review(str(result))
    
    def _parse_review(self, raw_review: str) -> Dict[str, Any]:
        """Parse review result"""
        strengths = []
        issues = []
        score = 7.0
        
        lines = raw_review.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'STRENGTHS:' in line:
                current_section = 'strengths'
            elif 'ISSUES:' in line:
                current_section = 'issues'
            elif 'OVERALL_SCORE:' in line:
                try:
                    score = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('-') and current_section:
                if current_section == 'strengths':
                    strengths.append(line[1:].strip())
                elif current_section == 'issues':
                    issues.append(line[1:].strip())
        
        return {
            "strengths": strengths,
            "issues": issues,
            "overall_score": score,
            "raw_review": raw_review
        }


class FactCheckerAgent(AgentWithMemory):
    """
    Specialized agent for fact-checking and verification
    """
    
    def __init__(self, tools: List[Tool], model=None, name: str = "fact_checker"):
        """Initialize fact checker agent"""
        super().__init__(
            name=name,
            role=AgentRole.FACT_CHECKER,
            tools=tools,
            model=model,
            use_code_agent=True
        )
    
    def _get_description(self) -> str:
        """Get agent description"""
        return """Expert fact-checker specializing in:
        - Claim verification against reliable sources
        - Source credibility assessment
        - Citation accuracy checking
        - Cross-referencing information
        - Identifying unsupported claims
        
        Ensures factual accuracy and proper attribution."""
    
    def _get_authorized_imports(self) -> List[str]:
        """Get authorized imports"""
        return ["json", "datetime", "re", "typing"]
    
    async def verify_claims(
        self,
        claims: List[str],
        sources: List[str] = None
    ) -> Dict[str, Any]:
        """
        Verify factual claims
        
        Args:
            claims: List of claims to verify
            sources: Optional list of source materials
            
        Returns:
            Verification results
        """
        sources_text = f"Using these sources:\n{sources}" if sources else "Search for sources as needed"
        
        task = f"""Verify these factual claims:
        
        {chr(10).join(f'{i+1}. {claim}' for i, claim in enumerate(claims))}
        
        {sources_text}
        
        For each claim, determine:
        - Is it verified? (yes/no/partially)
        - Confidence level (0-1)
        - Supporting evidence
        - Any contradictions found
        
        Format:
        CLAIM [number]:
        Status: [verified/unverified/partially verified]
        Confidence: [0-1]
        Evidence: [description]
        Contradictions: [if any]
        """
        
        result = await self.run(task)
        return {"verification": str(result), "claims_count": len(claims)}


# Factory functions for backward compatibility
def create_writer_agent(tools=None, model=None):
    """Create writer agent"""
    if tools is None:
        tools = []
    return WriterAgent(tools=tools, model=model)


def create_verifier_agent(tools=None, model=None):
    """Create fact checker agent"""
    if tools is None:
        tools = []
    return FactCheckerAgent(tools=tools, model=model)

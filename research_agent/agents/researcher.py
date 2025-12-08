"""
Research Agent - Advanced information gathering with multi-source research
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from smolagents import Tool
from config import Config, AgentRole
from .base_agent import BaseResearchAgent, AgentWithMemory

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Structured research result"""
    query: str
    findings: List[str]
    sources: List[str]
    confidence: float
    metadata: Dict[str, Any]


class ResearchAgent(AgentWithMemory):
    """
    Specialized agent for conducting web research
    Uses Perplexity, Tavily, and browser tools for comprehensive information gathering
    """
    
    def __init__(self, tools: List[Tool], model=None, name: str = "researcher"):
        """
        Initialize research agent
        
        Args:
            tools: Research tools (perplexity, tavily, browser, firecrawl)
            model: LLM model (defaults to Perplexity for research)
            name: Agent name
        """
        super().__init__(
            name=name,
            role=AgentRole.RESEARCHER,
            tools=tools,
            model=model,
            use_code_agent=True
        )
    
    def _get_description(self) -> str:
        """Get agent description"""
        return """Expert research agent specializing in:
        - Multi-source web research using Perplexity AI and Tavily
        - Real-time information gathering with recency awareness
        - Comprehensive fact-finding and source verification
        - Deep-dive research on complex topics
        - Citation and reference management
        
        Always prioritizes accuracy, uses multiple sources, and provides proper citations."""
    
    def _get_authorized_imports(self) -> List[str]:
        """Get authorized imports for code execution"""
        return ["json", "datetime", "time", "re", "functools", "typing"]
    
    async def research_topic(
        self,
        topic: str,
        depth: int = 1,
        max_sources: int = 5,
        recency: str = "month"
    ) -> ResearchResult:
        """
        Conduct comprehensive research on a topic
        
        Args:
            topic: Research topic or question
            depth: Research depth (1=basic, 2=intermediate, 3=deep)
            max_sources: Maximum number of sources to consult
            recency: Recency filter for search (day, week, month, year)
            
        Returns:
            ResearchResult with findings and sources
        """
        task = f"""Research the topic: "{topic}"
        
        Requirements:
        - Search depth: {depth} (1=basic overview, 2=detailed analysis, 3=comprehensive deep-dive)
        - Consult at least {max_sources} different sources
        - Prioritize information from the last {recency}
        - Verify facts across multiple sources
        - Extract key findings with proper citations
        - Rate your confidence in the findings (0-1 scale)
        
        Provide results in this format:
        FINDINGS:
        - [Finding 1]
        - [Finding 2]
        ...
        
        SOURCES:
        - [URL 1]: Brief description
        - [URL 2]: Brief description
        ...
        
        CONFIDENCE: [0-1 score]
        """
        
        result = await self.run(task)
        
        # Parse result into structured format
        return self._parse_research_result(topic, str(result))
    
    def _parse_research_result(self, query: str, raw_result: str) -> ResearchResult:
        """Parse raw research result into structured format"""
        # Simple parsing - can be enhanced with DSPy
        findings = []
        sources = []
        confidence = 0.8  # Default
        
        lines = raw_result.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'FINDINGS:' in line:
                current_section = 'findings'
            elif 'SOURCES:' in line:
                current_section = 'sources'
            elif 'CONFIDENCE:' in line:
                try:
                    confidence = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('-') and current_section:
                if current_section == 'findings':
                    findings.append(line[1:].strip())
                elif current_section == 'sources':
                    sources.append(line[1:].strip())
        
        return ResearchResult(
            query=query,
            findings=findings,
            sources=sources,
            confidence=confidence,
            metadata={"raw_result": raw_result}
        )


class RecursiveResearchAgent(ResearchAgent):
    """
    Advanced research agent with recursive topic breakdown
    Breaks complex topics into sub-topics for thorough research
    """
    
    def __init__(self, tools: List[Tool], model=None, max_depth: int = 3):
        """
        Initialize recursive research agent
        
        Args:
            tools: Research tools
            model: LLM model
            max_depth: Maximum recursion depth
        """
        super().__init__(tools=tools, model=model, name="recursive_researcher")
        self.max_depth = max_depth
    
    async def research_with_breakdown(
        self,
        topic: str,
        current_depth: int = 0
    ) -> Dict[str, Any]:
        """
        Recursively research topic by breaking it into sub-topics
        
        Args:
            topic: Main research topic
            current_depth: Current recursion depth
            
        Returns:
            Hierarchical research results
        """
        if current_depth >= self.max_depth:
            # Base case: simple research
            result = await self.research_topic(topic, depth=1)
            return {"topic": topic, "result": result, "sub_topics": []}
        
        # Step 1: Break topic into sub-topics
        breakdown_task = f"""Analyze this research topic and break it into 3-5 key sub-topics:
        Topic: {topic}
        
        List the sub-topics that need to be researched to fully understand this topic.
        Format: One sub-topic per line, starting with '-'
        """
        
        breakdown_result = await self.run(breakdown_task)
        sub_topics = [
            line.strip()[1:].strip()
            for line in str(breakdown_result).split('\n')
            if line.strip().startswith('-')
        ][:5]  # Limit to 5 sub-topics
        
        # Step 2: Research each sub-topic recursively
        if Config.ENABLE_ASYNC:
            sub_results = await asyncio.gather(*[
                self.research_with_breakdown(st, current_depth + 1)
                for st in sub_topics
            ])
        else:
            sub_results = []
            for st in sub_topics:
                sub_results.append(await self.research_with_breakdown(st, current_depth + 1))
        
        # Step 3: Research main topic with context from sub-topics
        main_result = await self.research_topic(topic, depth=2)
        
        return {
            "topic": topic,
            "result": main_result,
            "sub_topics": sub_results,
            "depth": current_depth
        }


# Factory functions for backward compatibility
def create_researcher_agent(tools=None, model=None):
    """Create standard research agent"""
    if tools is None:
        tools = []
    return ResearchAgent(tools=tools, model=model)


def create_recursive_researcher(tools=None, model=None, max_depth=3):
    """Create recursive research agent"""
    if tools is None:
        tools = []
    return RecursiveResearchAgent(tools=tools, model=model, max_depth=max_depth)

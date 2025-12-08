"""
Multi-Agent Orchestrator
Coordinates multiple specialized agents for complex research workflows
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from config import Config
from .base_agent import BaseResearchAgent, AgentMessage
from .researcher import ResearchAgent, RecursiveResearchAgent
from .writer import WriterAgent, EditorAgent, FactCheckerAgent

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Research workflow stages"""
    PLANNING = "planning"
    RESEARCH = "research"
    VERIFICATION = "verification"
    WRITING = "writing"
    EDITING = "editing"
    COMPLETE = "complete"


@dataclass
class WorkflowState:
    """Tracks workflow execution state"""
    current_stage: WorkflowStage = WorkflowStage.PLANNING
    completed_stages: List[WorkflowStage] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    stage_times: Dict[str, float] = field(default_factory=dict)


class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents for research workflows
    Manages agent communication, task delegation, and result aggregation
    """
    
    def __init__(
        self,
        research_agent: Optional[ResearchAgent] = None,
        writer_agent: Optional[WriterAgent] = None,
        fact_checker: Optional[FactCheckerAgent] = None,
        editor_agent: Optional[EditorAgent] = None
    ):
        """
        Initialize orchestrator with agents
        
        Args:
            research_agent: Agent for research tasks
            writer_agent: Agent for writing tasks
            fact_checker: Agent for verification tasks
            editor_agent: Agent for editing tasks
        """
        self.agents: Dict[str, BaseResearchAgent] = {}
        
        if research_agent:
            self.agents['researcher'] = research_agent
        if writer_agent:
            self.agents['writer'] = writer_agent
        if fact_checker:
            self.agents['fact_checker'] = fact_checker
        if editor_agent:
            self.agents['editor'] = editor_agent
        
        self.state = WorkflowState()
        self.message_queue: List[AgentMessage] = []
        
        logger.info(f"Orchestrator initialized with {len(self.agents)} agents")
    
    def add_agent(self, name: str, agent: BaseResearchAgent):
        """Add agent to orchestrator"""
        self.agents[name] = agent
        logger.info(f"Added agent: {name}")
    
    async def execute_workflow(
        self,
        topic: str,
        research_depth: int = 2,
        output_filename: str = "research_output.docx",
        style: str = "academic"
    ) -> Dict[str, Any]:
        """
        Execute complete research workflow
        
        Args:
            topic: Research topic
            research_depth: Research depth (1-3)
            output_filename: Output document filename
            style: Writing style
            
        Returns:
            Workflow results with document path
        """
        logger.info(f"Starting workflow for topic: {topic}")
        
        try:
            # Stage 1: Planning
            if Config.ENABLE_PLANNING:
                await self._run_stage(WorkflowStage.PLANNING, 
                                     lambda: self._planning_stage(topic))
            
            # Stage 2: Research
            if Config.ENABLE_RESEARCH:
                research_results = await self._run_stage(
                    WorkflowStage.RESEARCH,
                    lambda: self._research_stage(topic, research_depth)
                )
                self.state.artifacts['research'] = research_results
            
            # Stage 3: Verification
            if Config.ENABLE_VERIFICATION and 'fact_checker' in self.agents:
                verification = await self._run_stage(
                    WorkflowStage.VERIFICATION,
                    lambda: self._verification_stage(
                        self.state.artifacts.get('research', {})
                    )
                )
                self.state.artifacts['verification'] = verification
            
            # Stage 4: Writing
            if Config.ENABLE_WRITING and 'writer' in self.agents:
                document = await self._run_stage(
                    WorkflowStage.WRITING,
                    lambda: self._writing_stage(
                        topic,
                        self.state.artifacts.get('research', {}),
                        output_filename,
                        style
                    )
                )
                self.state.artifacts['document'] = document
            
            # Stage 5: Editing
            if 'editor' in self.agents:
                edited = await self._run_stage(
                    WorkflowStage.EDITING,
                    lambda: self._editing_stage(
                        self.state.artifacts.get('document', '')
                    )
                )
                self.state.artifacts['review'] = edited
            
            self.state.current_stage = WorkflowStage.COMPLETE
            
            return {
                'status': 'success',
                'topic': topic,
                'document_path': output_filename,
                'artifacts': self.state.artifacts,
                'execution_time': self._get_total_time(),
                'stage_times': self.state.stage_times
            }
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            self.state.errors.append(str(e))
            return {
                'status': 'failed',
                'error': str(e),
                'completed_stages': [s.value for s in self.state.completed_stages]
            }
    
    async def _run_stage(self, stage: WorkflowStage, stage_func) -> Any:
        """Run a workflow stage with timing"""
        logger.info(f"Starting stage: {stage.value}")
        self.state.current_stage = stage
        stage_start = datetime.now()
        
        try:
            result = await stage_func()
            self.state.completed_stages.append(stage)
            
            duration = (datetime.now() - stage_start).total_seconds()
            self.state.stage_times[stage.value] = duration
            logger.info(f"Completed stage {stage.value} in {duration:.2f}s")
            
            return result
        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {e}")
            raise
    
    async def _planning_stage(self, topic: str) -> Dict[str, Any]:
        """Planning stage - create research plan"""
        # For now, simple planning logic
        # Can be enhanced with dedicated PlannerAgent
        return {
            'topic': topic,
            'outline': ['Introduction', 'Background', 'Analysis', 'Conclusion'],
            'research_questions': [
                f'What is {topic}?',
                f'What are the key aspects of {topic}?',
                f'What are the implications of {topic}?'
            ]
        }
    
    async def _research_stage(self, topic: str, depth: int) -> Dict[str, Any]:
        """Research stage - gather information"""
        if 'researcher' not in self.agents:
            logger.warning("No research agent available")
            return {'findings': [], 'sources': []}
        
        researcher = self.agents['researcher']
        
        if isinstance(researcher, RecursiveResearchAgent) and depth > 1:
            # Use recursive research for deep topics
            result = await researcher.research_with_breakdown(topic)
        else:
            # Standard research
            result = await researcher.research_topic(topic, depth=depth)
        
        return result
    
    async def _verification_stage(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verification stage - fact-check findings"""
        fact_checker = self.agents['fact_checker']
        
        # Extract claims from research
        findings = research_data.get('findings', [])
        if hasattr(research_data, 'findings'):
            findings = research_data.findings
        
        if not findings:
            return {'verified': True, 'confidence': 1.0}
        
        # Verify claims
        verification = await fact_checker.verify_claims(findings)
        return verification
    
    async def _writing_stage(
        self,
        topic: str,
        research_data: Dict[str, Any],
        filename: str,
        style: str
    ) -> str:
        """Writing stage - create document"""
        writer = self.agents['writer']
        
        document = await writer.create_document(
            topic=topic,
            research_data=research_data,
            filename=filename,
            style=style
        )
        
        return document
    
    async def _editing_stage(self, document_content: str) -> Dict[str, Any]:
        """Editing stage - review and improve document"""
        editor = self.agents['editor']
        
        review = await editor.review_document(document_content)
        return review
    
    def _get_total_time(self) -> float:
        """Get total workflow execution time"""
        return (datetime.now() - self.state.start_time).total_seconds()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            'current_stage': self.state.current_stage.value,
            'completed_stages': [s.value for s in self.state.completed_stages],
            'errors': self.state.errors,
            'execution_time': self._get_total_time(),
            'agents': {name: agent.get_state_summary() 
                      for name, agent in self.agents.items()}
        }
    
    async def parallel_research(
        self,
        topics: List[str],
        depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Research multiple topics in parallel
        
        Args:
            topics: List of research topics
            depth: Research depth for each topic
            
        Returns:
            List of research results
        """
        if 'researcher' not in self.agents:
            raise ValueError("No research agent available")
        
        researcher = self.agents['researcher']
        
        if Config.ENABLE_ASYNC:
            tasks = [researcher.research_topic(topic, depth=depth) for topic in topics]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for topic in topics:
                result = await researcher.research_topic(topic, depth=depth)
                results.append(result)
        
        return [r for r in results if not isinstance(r, Exception)]


async def create_full_orchestrator(
    research_tools: List,
    writing_tools: List
) -> MultiAgentOrchestrator:
    """
    Create orchestrator with all agents initialized
    
    Args:
        research_tools: Tools for research agents
        writing_tools: Tools for writing/editing agents
        
    Returns:
        Configured MultiAgentOrchestrator
    """
    # Create agents
    researcher = ResearchAgent(tools=research_tools)
    writer = WriterAgent(tools=writing_tools)
    fact_checker = FactCheckerAgent(tools=research_tools)
    editor = EditorAgent(tools=writing_tools)
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(
        research_agent=researcher,
        writer_agent=writer,
        fact_checker=fact_checker,
        editor_agent=editor
    )
    
    return orchestrator

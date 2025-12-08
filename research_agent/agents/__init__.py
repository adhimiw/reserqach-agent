"""
Agents Package - Specialized Research Agents
Multi-agent system for comprehensive research workflows
"""

# Base classes
from .base_agent import (
    BaseResearchAgent,
    AgentWithMemory,
    AgentMessage,
    AgentState,
    AgentMemory
)

# Specialized agents
from .researcher import (
    ResearchAgent,
    RecursiveResearchAgent,
    ResearchResult,
    create_researcher_agent,
    create_recursive_researcher
)

from .writer import (
    WriterAgent,
    EditorAgent,
    FactCheckerAgent,
    DocumentSection,
    create_writer_agent,
    create_verifier_agent
)

# Orchestration
from .orchestrator import (
    MultiAgentOrchestrator,
    WorkflowStage,
    WorkflowState,
    create_full_orchestrator
)

__all__ = [
    # Base classes
    "BaseResearchAgent",
    "AgentWithMemory",
    "AgentMessage",
    "AgentState",
    "AgentMemory",
    
    # Research agents
    "ResearchAgent",
    "RecursiveResearchAgent",
    "ResearchResult",
    
    # Writing/Editing agents
    "WriterAgent",
    "EditorAgent",
    "FactCheckerAgent",
    "DocumentSection",
    
    # Orchestration
    "MultiAgentOrchestrator",
    "WorkflowStage",
    "WorkflowState",
    "create_full_orchestrator",
    
    # Factory functions (backward compatibility)
    "create_researcher_agent",
    "create_recursive_researcher",
    "create_writer_agent",
    "create_verifier_agent"
]


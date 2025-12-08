"""
Workflow Steps and Research Orchestration
"""

from typing import List, Dict, Any
from config import Config


# Define standard research workflow steps
RESEARCH_WORKFLOW = [
    {
        "step": 1,
        "name": "Planning & Outline Generation",
        "type": "planning",
        "description": "Break down topic into research questions and create outline",
        "agent": "planner"
    },
    {
        "step": 2,
        "name": "Research & Information Gathering",
        "type": "research",
        "description": "Conduct web research and gather relevant information",
        "agent": "researcher"
    },
    {
        "step": 3,
        "name": "Fact-Checking & Verification",
        "type": "verification",
        "description": "Verify claims and validate citations",
        "agent": "verifier"
    },
    {
        "step": 4,
        "name": "Paper Generation",
        "type": "writing",
        "description": "Generate academic paper with proper formatting",
        "agent": "writer"
    },
    {
        "step": 5,
        "name": "Quality Assurance",
        "type": "qa",
        "description": "Final review and formatting check",
        "agent": "verifier"
    }
]


class ResearchWorkflow:
    """Orchestrates the research workflow"""
    
    def __init__(self, agents: Dict[str, Any]):
        """
        Initialize workflow with agents
        
        Args:
            agents: Dictionary of agent_type -> agent_instance
        """
        self.agents = agents
        self.results = {}
        self.current_step = 0
    
    def get_step(self, step_num: int) -> Dict[str, Any]:
        """Get step definition by number"""
        for step in RESEARCH_WORKFLOW:
            if step["step"] == step_num:
                return step
        return None
    
    async def execute_step(self, step_num: int, context: str) -> str:
        """
        Execute a specific workflow step
        
        Args:
            step_num: Step number (1-5)
            context: Context/input for the step
        
        Returns:
            Result from the step
        """
        step = self.get_step(step_num)
        if not step:
            raise ValueError(f"Invalid step number: {step_num}")
        
        agent_type = step["agent"]
        if agent_type not in self.agents:
            raise ValueError(f"Agent '{agent_type}' not found")
        
        agent = self.agents[agent_type]
        print(f"\n[STEP {step_num}] {step['name']}")
        print(f"  Description: {step['description']}")
        
        result = agent.run(context)
        self.results[step_num] = result
        return result
    
    def get_results(self) -> Dict[int, str]:
        """Get all accumulated results"""
        return self.results


# Define workflow configurations for different research types
SIMPLE_WORKFLOW = [1, 2, 4]  # Planning, Research, Writing
DETAILED_WORKFLOW = [1, 2, 3, 4, 5]  # Full workflow
RECURSIVE_WORKFLOW = [1, 2, 3, 4, 5]  # Same as detailed, but with recursive research


def create_workflow_context(topic: str, depth: int = 1) -> Dict[str, Any]:
    """
    Create initial context for research workflow
    
    Args:
        topic: Research topic
        depth: Research depth for recursive research
    
    Returns:
        Context dictionary
    """
    return {
        "topic": topic,
        "depth": depth,
        "max_depth": Config.MAX_RESEARCH_DEPTH,
        "research_level": 0,
        "sub_topics": [],
        "findings": {},
        "citations": []
    }

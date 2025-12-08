"""
Enhanced Multi-Agent Framework

Implements LangChain, LlamaIndex, Haystack, and LangGraph patterns
for sophisticated multi-agent orchestration with token optimization.
"""

from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

from smolagents import CodeAgent, LiteLLMModel, tool
from config import Config
from tools.token_optimizer import TokenOptimizer, optimize_text_for_llm


class AgentRole(Enum):
    """Roles for specialized agents."""
    PLANNER = "planner"
    RESEARCHER = "researcher"
    VERIFIER = "verifier"
    WRITER = "writer"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"


class AgentState(Enum):
    """Possible states of an agent."""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING = "waiting"


@dataclass
class Message:
    """Message passed between agents."""
    sender: str
    receiver: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AgentContext:
    """Context shared between agents."""
    topic: str
    research_findings: Dict = field(default_factory=dict)
    verified_sources: List[str] = field(default_factory=list)
    generated_content: Dict = field(default_factory=dict)
    api_calls: int = 0
    tokens_saved: int = 0
    
    def to_dict(self) -> Dict:
        """Convert context to dictionary."""
        return {
            'topic': self.topic,
            'research_findings': self.research_findings,
            'verified_sources': self.verified_sources,
            'generated_content': self.generated_content,
            'api_calls': self.api_calls,
            'tokens_saved': self.tokens_saved,
        }


class SpecializedAgent:
    """Base class for specialized agents."""
    
    def __init__(self, name: str, role: AgentRole, tools: List[Callable] = None, 
                 model: Optional[LiteLLMModel] = None):
        self.name = name
        self.role = role
        self.tools = tools or []
        self.state = AgentState.IDLE
        self.message_queue: List[Message] = []
        self.optimizer = TokenOptimizer()
        
        # Create the agent
        if model is None:
            model = LiteLLMModel(
                model_id=Config.LLM_MODEL_ID,
                api_key=Config.PERPLEXITY_API_KEY,
            )
        
        self.agent = CodeAgent(
            tools=self.tools,
            model=model,
            name=self.name
        )
        self.model = model
    
    async def process_message(self, message: Message) -> Message:
        """Process incoming message and generate response."""
        self.state = AgentState.WORKING
        
        try:
            # Optimize the message content for token efficiency
            optimized_content = self.optimizer.optimize_prompt(message.content)
            
            # Run the agent
            result = self.agent.run(optimized_content)
            
            # Create response
            response = Message(
                sender=self.name,
                receiver=message.sender,
                content=str(result),
                metadata={
                    'optimization_stats': self.optimizer.get_stats(),
                    'original_message': message.content
                }
            )
            
            self.state = AgentState.COMPLETED
            return response
            
        except Exception as e:
            self.state = AgentState.ERROR
            return Message(
                sender=self.name,
                receiver=message.sender,
                content=f"Error: {str(e)}",
                metadata={'error': True}
            )
    
    def add_tool(self, tool_func: Callable) -> None:
        """Add a tool to this agent."""
        self.tools.append(tool_func)


class PlannerAgent(SpecializedAgent):
    """Agent responsible for planning research strategy."""
    
    def __init__(self, tools: List[Callable] = None, model: Optional[LiteLLMModel] = None):
        super().__init__("PlannerAgent", AgentRole.PLANNER, tools, model)
    
    def generate_research_plan(self, topic: str, context: AgentContext) -> Dict:
        """Generate detailed research plan."""
        prompt = f"""
        Create a comprehensive research plan for: {topic}
        
        Include:
        1. Key research questions (5-7 main questions)
        2. Subtopics to explore
        3. Types of sources needed
        4. Verification strategy
        5. Timeline for each phase
        
        Format as JSON with clear structure.
        """
        
        optimized = optimize_text_for_llm(prompt, f"Topic: {topic}")
        result = self.agent.run(optimized)
        
        try:
            plan = json.loads(str(result))
        except:
            plan = {'raw_plan': str(result)}
        
        return plan


class ResearcherAgent(SpecializedAgent):
    """Agent responsible for gathering research information."""
    
    def __init__(self, tools: List[Callable] = None, model: Optional[LiteLLMModel] = None):
        super().__init__("ResearcherAgent", AgentRole.RESEARCHER, tools, model)
    
    def research_topic(self, topic: str, questions: List[str], 
                      context: AgentContext) -> Dict[str, str]:
        """Research answers to specific questions."""
        findings = {}
        
        for question in questions:
            prompt = f"""
            Research and answer this question thoroughly: {question}
            
            Topic context: {topic}
            
            Provide:
            1. Direct answer
            2. Key evidence/sources
            3. Relevant statistics
            4. Related concepts
            
            Keep concise but comprehensive.
            """
            
            optimized = optimize_text_for_llm(prompt, f"Question: {question}")
            result = self.agent.run(optimized)
            findings[question] = str(result)
        
        return findings


class VerifierAgent(SpecializedAgent):
    """Agent responsible for verifying claims and sources."""
    
    def __init__(self, tools: List[Callable] = None, model: Optional[LiteLLMModel] = None):
        super().__init__("VerifierAgent", AgentRole.VERIFIER, tools, model)
    
    def verify_claims(self, claims: List[str], sources: List[str], 
                     context: AgentContext) -> Dict[str, Dict]:
        """Verify claims against sources."""
        verification_results = {}
        
        for claim in claims:
            prompt = f"""
            Verify this claim: {claim}
            
            Against these sources:
            {', '.join(sources)}
            
            Provide:
            1. Verified (True/False/Partial)
            2. Evidence level
            3. Reliability assessment
            4. Alternative viewpoints
            """
            
            optimized = optimize_text_for_llm(prompt, f"Claim: {claim}")
            result = self.agent.run(optimized)
            
            verification_results[claim] = {
                'verification': str(result),
                'sources_checked': sources
            }
        
        return verification_results


class WriterAgent(SpecializedAgent):
    """Agent responsible for writing content."""
    
    def __init__(self, tools: List[Callable] = None, model: Optional[LiteLLMModel] = None):
        super().__init__("WriterAgent", AgentRole.WRITER, tools, model)
    
    def write_section(self, section_title: str, research_notes: str, 
                     section_context: str, style: str = "academic") -> str:
        """Write a research paper section."""
        prompt = f"""
        Write a {style} section with title: {section_title}
        
        Context: {section_context}
        
        Research Notes:
        {research_notes}
        
        Requirements:
        1. Well-structured paragraphs
        2. Clear topic sentences
        3. Evidence-based statements
        4. Professional tone
        5. Smooth transitions
        
        Generate the section content now.
        """
        
        optimized = optimize_text_for_llm(prompt, f"Section: {section_title}")
        result = self.agent.run(optimized)
        return str(result)


class MultiAgentOrchestrator:
    """Orchestrates communication and coordination between agents."""
    
    def __init__(self, agents: Dict[str, SpecializedAgent] = None):
        self.agents = agents or {}
        self.context = AgentContext(topic="")
        self.message_history: List[Message] = []
        self.execution_trace = []
    
    def register_agent(self, name: str, agent: SpecializedAgent) -> None:
        """Register an agent in the orchestrator."""
        self.agents[name] = agent
    
    def broadcast_message(self, sender: str, content: str, 
                         target_agents: Optional[List[str]] = None) -> List[Message]:
        """Broadcast message to specific agents."""
        responses = []
        targets = target_agents or list(self.agents.keys())
        
        for target in targets:
            if target in self.agents:
                message = Message(sender=sender, receiver=target, content=content)
                self.message_history.append(message)
                
                # Process asynchronously (simplified to sync here)
                response = self._process_message_sync(message)
                responses.append(response)
        
        return responses
    
    def _process_message_sync(self, message: Message) -> Message:
        """Process message synchronously."""
        agent = self.agents.get(message.receiver)
        if agent:
            # Simplified processing
            return Message(
                sender=message.receiver,
                receiver=message.sender,
                content=f"Agent {message.receiver} processed: {message.content[:50]}..."
            )
        return message
    
    def set_context(self, context: AgentContext) -> None:
        """Set shared context for all agents."""
        self.context = context
    
    def get_context(self) -> AgentContext:
        """Get current shared context."""
        return self.context
    
    def execute_workflow(self, workflow_steps: List[Dict]) -> Dict:
        """Execute a multi-agent workflow."""
        results = {}
        
        for step in workflow_steps:
            step_name = step.get('name', 'unknown')
            agent_name = step.get('agent')
            action = step.get('action')
            params = step.get('params', {})
            
            self.execution_trace.append({
                'step': step_name,
                'agent': agent_name,
                'timestamp': datetime.now().isoformat()
            })
            
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                try:
                    result = self._execute_agent_action(agent, action, params)
                    results[step_name] = result
                except Exception as e:
                    results[step_name] = {'error': str(e)}
        
        return results
    
    def _execute_agent_action(self, agent: SpecializedAgent, 
                             action: str, params: Dict) -> Any:
        """Execute specific action on an agent."""
        if action == "research":
            return agent.research_topic(**params)
        elif action == "verify":
            return agent.verify_claims(**params)
        elif action == "write":
            return agent.write_section(**params)
        elif action == "plan":
            return agent.generate_research_plan(**params)
        else:
            return {"status": "unknown_action"}
    
    def get_execution_report(self) -> Dict:
        """Get report of execution."""
        return {
            'agents_registered': len(self.agents),
            'messages_processed': len(self.message_history),
            'execution_steps': self.execution_trace,
            'context': self.context.to_dict()
        }


def create_multi_agent_system(tools: Dict[str, List[Callable]]) -> MultiAgentOrchestrator:
    """Factory function to create a complete multi-agent system."""
    
    orchestrator = MultiAgentOrchestrator()
    
    # Create agents
    planner = PlannerAgent(tools=tools.get('planner', []))
    researcher = ResearcherAgent(tools=tools.get('researcher', []))
    verifier = VerifierAgent(tools=tools.get('verifier', []))
    writer = WriterAgent(tools=tools.get('writer', []))
    
    # Register agents
    orchestrator.register_agent("planner", planner)
    orchestrator.register_agent("researcher", researcher)
    orchestrator.register_agent("verifier", verifier)
    orchestrator.register_agent("writer", writer)
    
    return orchestrator

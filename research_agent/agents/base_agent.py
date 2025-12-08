"""
Base Agent Classes
Foundation for all specialized research agents with shared functionality
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from smolagents import CodeAgent, ToolCallingAgent, Tool
from config import Config, AgentRole

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    recipient: str
    content: Any
    message_type: str  # 'request', 'response', 'notification'
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Tracks agent execution state"""
    current_task: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    iterations: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


class BaseResearchAgent(ABC):
    """
    Base class for all research agents
    Provides common functionality and interface
    """
    
    def __init__(
        self,
        name: str,
        role: AgentRole,
        tools: List[Tool],
        model=None,
        use_code_agent: bool = True,
        max_iterations: int = None
    ):
        """
        Initialize base agent
        
        Args:
            name: Agent name/identifier
            role: Agent role from AgentRole enum
            tools: List of tools available to agent
            model: LLM model (defaults to role-specific model)
            use_code_agent: Whether to use CodeAgent (True) or ToolCallingAgent (False)
            max_iterations: Maximum iterations for agent execution
        """
        self.name = name
        self.role = role
        self.tools = tools or []
        self.model = model or Config.get_model(role=role)
        self.max_iterations = max_iterations or Config.MAX_AGENT_ITERATIONS
        self.state = AgentState()
        
        # Convert tools to list if needed
        if hasattr(self.tools, '__iter__') and not isinstance(self.tools, (list, str)):
            tool_list = list(self.tools)
        else:
            tool_list = self.tools if isinstance(self.tools, list) else []
        
        # Create underlying smolagent
        agent_class = CodeAgent if use_code_agent else ToolCallingAgent
        self.agent = agent_class(
            tools=tool_list,
            model=self.model,
            name=name,
            description=self._get_description(),
            additional_authorized_imports=self._get_authorized_imports(),
            max_steps=self.max_iterations
        )
        
        logger.info(f"Initialized {self.role.value} agent: {name}")
    
    @abstractmethod
    def _get_description(self) -> str:
        """Get agent description - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _get_authorized_imports(self) -> List[str]:
        """Get authorized Python imports for CodeAgent"""
        pass
    
    async def run(self, task: str, **kwargs) -> Any:
        """
        Execute agent task asynchronously
        
        Args:
            task: Task description
            **kwargs: Additional task parameters
            
        Returns:
            Task result
        """
        self.state.current_task = task
        self.state.start_time = datetime.now()
        self.state.iterations = 0
        
        try:
            logger.info(f"{self.name} starting task: {task}")
            
            # Execute task
            if Config.ENABLE_ASYNC:
                result = await asyncio.to_thread(self.agent.run, task, **kwargs)
            else:
                result = self.agent.run(task, **kwargs)
            
            self.state.completed_tasks.append(task)
            logger.info(f"{self.name} completed task successfully")
            
            return result
            
        except Exception as e:
            error_msg = f"Agent {self.name} failed: {str(e)}"
            self.state.errors.append(error_msg)
            logger.error(error_msg)
            raise
            
        finally:
            self.state.end_time = datetime.now()
            self.state.current_task = None
    
    def run_sync(self, task: str, **kwargs) -> Any:
        """
        Execute agent task synchronously
        
        Args:
            task: Task description
            **kwargs: Additional task parameters
            
        Returns:
            Task result
        """
        self.state.current_task = task
        self.state.start_time = datetime.now()
        
        try:
            logger.info(f"{self.name} starting task: {task}")
            result = self.agent.run(task, **kwargs)
            self.state.completed_tasks.append(task)
            return result
        except Exception as e:
            error_msg = f"Agent {self.name} failed: {str(e)}"
            self.state.errors.append(error_msg)
            logger.error(error_msg)
            raise
        finally:
            self.state.end_time = datetime.now()
            self.state.current_task = None
    
    def send_message(self, recipient: str, content: Any, message_type: str = "request") -> AgentMessage:
        """
        Send message to another agent
        
        Args:
            recipient: Recipient agent name
            content: Message content
            message_type: Type of message
            
        Returns:
            AgentMessage object
        """
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            content=content,
            message_type=message_type
        )
        logger.debug(f"Message from {self.name} to {recipient}: {message_type}")
        return message
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of agent state"""
        duration = None
        if self.state.start_time and self.state.end_time:
            duration = (self.state.end_time - self.state.start_time).total_seconds()
        
        return {
            "name": self.name,
            "role": self.role.value,
            "current_task": self.state.current_task,
            "completed_tasks": len(self.state.completed_tasks),
            "errors": len(self.state.errors),
            "iterations": self.state.iterations,
            "duration_seconds": duration
        }
    
    def reset_state(self):
        """Reset agent state"""
        self.state = AgentState()
        logger.info(f"Reset state for {self.name}")


class AgentMemory:
    """
    Simple memory system for agents
    Stores conversation history and artifacts
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize memory
        
        Args:
            max_history: Maximum conversation history items to keep
        """
        self.max_history = max_history
        self.conversation_history: List[Dict[str, Any]] = []
        self.artifacts: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add_interaction(self, role: str, content: str, metadata: Dict = None):
        """Add interaction to conversation history"""
        interaction = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        self.conversation_history.append(interaction)
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def store_artifact(self, key: str, value: Any):
        """Store artifact in memory"""
        self.artifacts[key] = value
    
    def get_artifact(self, key: str) -> Optional[Any]:
        """Retrieve artifact from memory"""
        return self.artifacts.get(key)
    
    def get_recent_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent interactions"""
        return self.conversation_history[-n:] if self.conversation_history else []
    
    def clear(self):
        """Clear all memory"""
        self.conversation_history.clear()
        self.artifacts.clear()
        self.metadata.clear()


class AgentWithMemory(BaseResearchAgent):
    """
    Extended base agent with memory capabilities
    """
    
    def __init__(self, *args, memory_size: int = 100, **kwargs):
        """Initialize agent with memory"""
        super().__init__(*args, **kwargs)
        self.memory = AgentMemory(max_history=memory_size)
    
    async def run(self, task: str, **kwargs) -> Any:
        """Run with memory tracking"""
        self.memory.add_interaction("user", task)
        result = await super().run(task, **kwargs)
        self.memory.add_interaction("assistant", str(result))
        return result
    
    def run_sync(self, task: str, **kwargs) -> Any:
        """Run synchronously with memory tracking"""
        self.memory.add_interaction("user", task)
        result = super().run_sync(task, **kwargs)
        self.memory.add_interaction("assistant", str(result))
        return result

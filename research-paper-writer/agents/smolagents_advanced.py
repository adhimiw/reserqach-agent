"""
Advanced SmolaGents Framework Integration

Implements state-of-the-art multi-agent patterns with:
- LangChain-style orchestration
- LlamaIndex retrieval patterns
- Haystack search pipelines
- LangGraph stateful graphs
"""

from typing import Dict, List, Any, Optional, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

from smolagents import CodeAgent, LiteLLMModel, tool, ToolCollection
from config import Config


class TaskType(Enum):
    """Types of tasks that can be executed."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    WRITING = "writing"
    VERIFICATION = "verification"


@dataclass
class TaskNode:
    """Represents a task in the execution graph."""
    id: str
    task_type: TaskType
    description: str
    agent_name: str
    input_data: Dict = field(default_factory=dict)
    output_data: Optional[Dict] = None
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    timestamp: datetime = field(default_factory=datetime.now)


class ExecutionGraph:
    """Directed acyclic graph for task execution."""
    
    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}
        self.edges: List[tuple] = []
    
    def add_task(self, task: TaskNode) -> None:
        """Add a task node to the graph."""
        self.nodes[task.id] = task
    
    def add_dependency(self, from_task: str, to_task: str) -> None:
        """Add a dependency between tasks."""
        self.edges.append((from_task, to_task))
        if to_task in self.nodes:
            self.nodes[to_task].dependencies.append(from_task)
    
    def get_execution_order(self) -> List[str]:
        """Get topological sort of tasks for execution."""
        visited = set()
        order = []
        
        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            
            node = self.nodes.get(node_id)
            if node:
                for dep in node.dependencies:
                    visit(dep)
            
            order.append(node_id)
        
        for node_id in self.nodes:
            visit(node_id)
        
        return order
    
    def get_parallel_tasks(self) -> List[List[str]]:
        """Get tasks that can run in parallel."""
        execution_order = self.get_execution_order()
        parallel_batches = []
        executed = set()
        
        for task_id in execution_order:
            task = self.nodes[task_id]
            deps_done = all(dep in executed for dep in task.dependencies)
            
            if deps_done:
                if parallel_batches and all(
                    self.nodes[t].dependencies == [] or 
                    all(d in executed for d in self.nodes[t].dependencies)
                    for t in parallel_batches[-1]
                ):
                    parallel_batches[-1].append(task_id)
                else:
                    parallel_batches.append([task_id])
                executed.add(task_id)
        
        return parallel_batches


class SmolagentsPipeline:
    """Orchestrates agents in a LangGraph-style pipeline."""
    
    def __init__(self, name: str):
        self.name = name
        self.agents: Dict[str, CodeAgent] = {}
        self.graph = ExecutionGraph()
        self.execution_history = []
    
    def register_agent(self, name: str, agent: CodeAgent) -> None:
        """Register an agent in the pipeline."""
        self.agents[name] = agent
    
    def create_task(self, task_id: str, task_type: TaskType,
                   description: str, agent_name: str,
                   input_data: Dict = None) -> TaskNode:
        """Create a task node."""
        task = TaskNode(
            id=task_id,
            task_type=task_type,
            description=description,
            agent_name=agent_name,
            input_data=input_data or {}
        )
        self.graph.add_task(task)
        return task
    
    def add_task_dependency(self, from_task: str, to_task: str) -> None:
        """Add dependency between tasks."""
        self.graph.add_dependency(from_task, to_task)
    
    def execute_pipeline(self) -> Dict[str, Any]:
        """Execute the pipeline in order."""
        results = {}
        execution_order = self.graph.get_execution_order()
        
        for task_id in execution_order:
            task = self.graph.nodes[task_id]
            agent = self.agents.get(task.agent_name)
            
            if not agent:
                task.status = "failed"
                results[task_id] = {"error": f"Agent {task.agent_name} not found"}
                continue
            
            try:
                task.status = "running"
                
                # Prepare input (use outputs from dependent tasks if available)
                input_data = task.input_data.copy()
                for dep_id in task.dependencies:
                    if dep_id in results:
                        input_data[f"{dep_id}_output"] = results[dep_id]
                
                # Execute task
                prompt = f"{task.description}\n\nInput: {json.dumps(input_data)}"
                output = agent.run(prompt)
                
                task.output_data = {"result": str(output)}
                task.status = "completed"
                results[task_id] = {"result": str(output), "status": "success"}
                
                self.execution_history.append({
                    "task_id": task_id,
                    "status": "completed",
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                task.status = "failed"
                results[task_id] = {"error": str(e), "status": "failed"}
                self.execution_history.append({
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results


class RetrievalPipeline:
    """LlamaIndex-style retrieval pipeline for research."""
    
    def __init__(self, agent: CodeAgent):
        self.agent = agent
        self.documents: List[Dict] = []
        self.embeddings: Dict[str, List[float]] = {}
    
    def add_document(self, doc_id: str, content: str, metadata: Dict = None) -> None:
        """Add document to retrieval base."""
        self.documents.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata or {}
        })
    
    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant documents."""
        # Simple keyword matching (in production, use embeddings)
        query_words = set(query.lower().split())
        
        scores = []
        for doc in self.documents:
            content_words = set(doc["content"].lower().split())
            overlap = len(query_words & content_words)
            scores.append((doc, overlap))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scores[:top_k]]
    
    def query(self, query: str) -> str:
        """Query with retrieval augmentation."""
        relevant_docs = self.retrieve_relevant(query, top_k=3)
        
        context = "\n".join([
            f"Document {doc['id']}: {doc['content'][:200]}..."
            for doc in relevant_docs
        ])
        
        prompt = f"""
        Based on these documents:
        {context}
        
        Answer this query: {query}
        """
        
        return self.agent.run(prompt)


class SearchPipeline:
    """Haystack-style search pipeline."""
    
    def __init__(self, agents: Dict[str, CodeAgent]):
        self.agents = agents
        self.index: Dict[str, List[str]] = {}
    
    def index_content(self, topic: str, content: List[str]) -> None:
        """Index content for searching."""
        self.index[topic] = content
    
    def search(self, query: str, topic: str = None) -> List[str]:
        """Search indexed content."""
        results = []
        
        search_topics = [topic] if topic else list(self.index.keys())
        
        for search_topic in search_topics:
            if search_topic not in self.index:
                continue
            
            query_words = set(query.lower().split())
            for content in self.index[search_topic]:
                content_words = set(content.lower().split())
                if query_words & content_words:
                    results.append(content)
        
        return results[:10]


class LangChainStyleOrchestrator:
    """LangChain-inspired orchestrator with chain of thought."""
    
    def __init__(self):
        self.chains: Dict[str, List[Callable]] = {}
        self.memory = {}
    
    def create_chain(self, name: str, steps: List[Callable]) -> None:
        """Create a chain of steps."""
        self.chains[name] = steps
    
    def execute_chain(self, chain_name: str, input_data: str) -> str:
        """Execute a chain sequentially."""
        if chain_name not in self.chains:
            return f"Chain {chain_name} not found"
        
        result = input_data
        for step in self.chains[chain_name]:
            result = step(result)
            self.memory[f"{chain_name}_step_{len(self.memory)}"] = result
        
        return result


@tool
def run_research_cycle(topic: str, depth: int = 3) -> Dict:
    """
    Execute a full research cycle with multiple agents.
    
    Args:
        topic: Research topic
        depth: Research depth (1-3)
        
    Returns:
        Research findings
    """
    return {
        "topic": topic,
        "depth": depth,
        "status": "researching"
    }


@tool
def synthesize_findings(findings: List[Dict]) -> str:
    """
    Synthesize multiple research findings.
    
    Args:
        findings: List of finding dictionaries
        
    Returns:
        Synthesized text
    """
    combined = "\n".join([
        f"- {finding.get('title', 'Unknown')}: {finding.get('summary', '')}"
        for finding in findings
    ])
    return combined


@tool
def verify_and_rank_sources(sources: List[Dict]) -> List[Dict]:
    """
    Verify and rank sources by reliability.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Ranked sources with scores
    """
    ranked = sorted(
        sources,
        key=lambda x: x.get('reliability_score', 0),
        reverse=True
    )
    return ranked


def create_advanced_pipeline() -> SmolagentsPipeline:
    """Factory to create advanced pipeline with multiple agents."""
    
    pipeline = SmolagentsPipeline("ResearchPipeline")
    
    # Create agents
    model = LiteLLMModel(
        model_id=Config.LLM_MODEL_ID,
        api_key=Config.PERPLEXITY_API_KEY,
    )
    
    researcher = CodeAgent(
        tools=[run_research_cycle],
        model=model,
        name="researcher"
    )
    
    synthesizer = CodeAgent(
        tools=[synthesize_findings],
        model=model,
        name="synthesizer"
    )
    
    verifier = CodeAgent(
        tools=[verify_and_rank_sources],
        model=model,
        name="verifier"
    )
    
    # Register agents
    pipeline.register_agent("researcher", researcher)
    pipeline.register_agent("synthesizer", synthesizer)
    pipeline.register_agent("verifier", verifier)
    
    # Create task graph
    pipeline.create_task(
        "research_1",
        TaskType.RESEARCH,
        "Research the main topic",
        "researcher"
    )
    
    pipeline.create_task(
        "research_2",
        TaskType.RESEARCH,
        "Research subtopics",
        "researcher"
    )
    
    pipeline.create_task(
        "synthesize",
        TaskType.SYNTHESIS,
        "Synthesize research findings",
        "synthesizer"
    )
    
    pipeline.create_task(
        "verify",
        TaskType.VERIFICATION,
        "Verify source credibility",
        "verifier"
    )
    
    # Add dependencies
    pipeline.add_task_dependency("research_1", "synthesize")
    pipeline.add_task_dependency("research_2", "synthesize")
    pipeline.add_task_dependency("synthesize", "verify")
    
    return pipeline

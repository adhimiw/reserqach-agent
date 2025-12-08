"""
Complete Example: Using Token Optimization & Multi-Agent Framework

This script demonstrates:
1. Token optimization for cost reduction
2. Multi-agent collaboration
3. Advanced smolagents pipeline
4. Integration with MCP servers
"""

import asyncio
from typing import Dict, List
import json

from config import Config
from tools.token_optimizer import TokenOptimizer, optimize_text_for_llm, get_optimization_stats
from agents.multi_agent_framework import (
    create_multi_agent_system, AgentContext, Message
)
from agents.smolagents_advanced import (
    SmolagentsPipeline, TaskType, RetrievalPipeline, SearchPipeline
)


def example_1_token_optimization():
    """Example 1: Basic Token Optimization"""
    print("=" * 60)
    print("EXAMPLE 1: Token Optimization")
    print("=" * 60)
    
    optimizer = TokenOptimizer()
    
    # Original prompt
    original = """
    I am requesting that you perform a comprehensive and thorough analysis of the 
    impact of artificial intelligence on modern software engineering practices. 
    Please provide a detailed examination of how AI technologies have fundamentally 
    changed the way software engineers approach their daily tasks and challenges.
    """
    
    print(f"Original length: {len(original.split())} tokens")
    print(f"Original: {original[:100]}...")
    
    # Optimize
    optimized = optimizer.optimize_prompt(original, use_compression=True)
    print(f"\nOptimized length: {len(optimized.split())} tokens")
    print(f"Optimized: {optimized[:100]}...")
    
    # Show stats
    stats = optimizer.get_stats()
    print(f"\nOptimization Stats:")
    print(f"  - Total prompts processed: {stats['total_prompts']}")
    print(f"  - Tokens saved: {stats['tokens_saved']}")
    print(f"  - Compression ratio: {stats['compression_ratio']:.1f}%")
    print()


def example_2_caching():
    """Example 2: Prompt Caching"""
    print("=" * 60)
    print("EXAMPLE 2: Prompt Caching")
    print("=" * 60)
    
    optimizer = TokenOptimizer()
    
    # Same prompt twice
    prompt1 = "What is machine learning?"
    prompt2 = "What is machine learning?"
    
    print("Processing prompt 1...")
    result1 = optimizer.optimize_prompt(prompt1, use_cache=True)
    stats1 = optimizer.get_stats()
    
    print(f"  Cache hits: {stats1['cached_hits']}")
    
    print("\nProcessing prompt 2 (identical)...")
    result2 = optimizer.optimize_prompt(prompt2, use_cache=True)
    stats2 = optimizer.get_stats()
    
    print(f"  Cache hits: {stats2['cached_hits']}")
    print(f"  Result from cache: {result1 == result2}")
    print()


def example_3_multi_agent_system():
    """Example 3: Multi-Agent System"""
    print("=" * 60)
    print("EXAMPLE 3: Multi-Agent System")
    print("=" * 60)
    
    # Create context
    context = AgentContext(topic="Quantum Computing")
    
    print(f"Topic: {context.topic}")
    print(f"Initial state: {context.to_dict()}")
    
    # Add research findings
    context.research_findings['quantum_basics'] = "Quantum computing uses qubits..."
    context.verified_sources.append("https://example.com/quantum")
    
    print(f"\nUpdated research findings: {len(context.research_findings)}")
    print(f"Verified sources: {len(context.verified_sources)}")
    print()


def example_4_prompt_compression_techniques():
    """Example 4: Different Compression Techniques"""
    print("=" * 60)
    print("EXAMPLE 4: Compression Techniques")
    print("=" * 60)
    
    from tools.token_optimizer import PromptCompressor
    
    text = """
    I am writing to you today to kindly request that you please provide 
    a comprehensive analysis. In my opinion, I believe that we should 
    basically focus on the key aspects. It seems to be important that 
    we essentially address the fundamental issues that appear to be 
    critical for our success.
    """
    
    print("Original text:")
    print(text)
    print(f"Length: {len(text.split())} tokens\n")
    
    # Remove redundancy
    step1 = PromptCompressor.remove_redundancy(text)
    print("After removing redundancy:")
    print(step1)
    print(f"Length: {len(step1.split())} tokens\n")
    
    # Extract keywords
    keywords = PromptCompressor.extract_keywords(text, max_keywords=5)
    print(f"Key keywords: {keywords}\n")


def example_5_workflow_execution():
    """Example 5: Workflow Execution"""
    print("=" * 60)
    print("EXAMPLE 5: Workflow Execution")
    print("=" * 60)
    
    # Define workflow steps
    workflow = [
        {
            "name": "planning",
            "agent": "planner",
            "action": "plan",
            "params": {"topic": "AI Ethics", "context": None}
        },
        {
            "name": "initial_research",
            "agent": "researcher",
            "action": "research",
            "params": {
                "topic": "AI Ethics",
                "questions": [
                    "What is AI ethics?",
                    "Why is it important?"
                ]
            }
        },
        {
            "name": "verification",
            "agent": "verifier",
            "action": "verify",
            "params": {
                "claims": ["AI ethics is crucial"],
                "sources": []
            }
        }
    ]
    
    print("Workflow steps:")
    for i, step in enumerate(workflow, 1):
        print(f"{i}. {step['name']} ({step['agent']})")
    print()


def example_6_execution_graph():
    """Example 6: Task Execution Graph"""
    print("=" * 60)
    print("EXAMPLE 6: Execution Graph")
    print("=" * 60)
    
    from agents.smolagents_advanced import ExecutionGraph, TaskNode, TaskType
    
    graph = ExecutionGraph()
    
    # Create tasks
    task1 = TaskNode(
        id="research_1",
        task_type=TaskType.RESEARCH,
        description="Initial research",
        agent_name="researcher"
    )
    
    task2 = TaskNode(
        id="research_2",
        task_type=TaskType.RESEARCH,
        description="Secondary research",
        agent_name="researcher"
    )
    
    task3 = TaskNode(
        id="synthesize",
        task_type=TaskType.SYNTHESIS,
        description="Synthesize findings",
        agent_name="synthesizer"
    )
    
    # Add to graph
    graph.add_task(task1)
    graph.add_task(task2)
    graph.add_task(task3)
    
    # Add dependencies
    graph.add_dependency("research_1", "synthesize")
    graph.add_dependency("research_2", "synthesize")
    
    # Get execution order
    order = graph.get_execution_order()
    print(f"Execution order: {order}")
    
    # Get parallel tasks
    parallel = graph.get_parallel_tasks()
    print(f"Parallel batches: {parallel}")
    print()


def example_7_retrieval_pipeline():
    """Example 7: Retrieval Pipeline (LlamaIndex style)"""
    print("=" * 60)
    print("EXAMPLE 7: Retrieval Pipeline")
    print("=" * 60)
    
    print("Setting up retrieval pipeline...")
    
    documents = [
        {
            "id": "doc1",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence..."
        },
        {
            "id": "doc2",
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning uses neural networks with multiple layers..."
        },
        {
            "id": "doc3",
            "title": "Natural Language Processing",
            "content": "NLP focuses on processing and understanding human language..."
        }
    ]
    
    print(f"Documents loaded: {len(documents)}\n")
    
    # Simulate retrieval
    query = "neural networks"
    
    print(f"Query: '{query}'")
    print("\nRelevant documents:")
    
    query_words = set(query.lower().split())
    for doc in documents:
        content_words = set(doc["content"].lower().split())
        overlap = len(query_words & content_words)
        if overlap > 0:
            print(f"  - {doc['title']} (relevance score: {overlap})")
    print()


def example_8_cost_calculation():
    """Example 8: Cost Calculation"""
    print("=" * 60)
    print("EXAMPLE 8: Cost Analysis")
    print("=" * 60)
    
    # Assuming costs
    gpt4_cost_per_1k = 0.03  # $0.03 per 1K tokens
    
    original_tokens = 5000
    optimized_tokens = 2000  # 40% of original
    
    original_cost = (original_tokens / 1000) * gpt4_cost_per_1k
    optimized_cost = (optimized_tokens / 1000) * gpt4_cost_per_1k
    savings = original_cost - optimized_cost
    savings_pct = (savings / original_cost) * 100
    
    print(f"GPT-4 API Cost Analysis:")
    print(f"  Original tokens: {original_tokens}")
    print(f"  Optimized tokens: {optimized_tokens}")
    print(f"  Original cost: ${original_cost:.4f}")
    print(f"  Optimized cost: ${optimized_cost:.4f}")
    print(f"  Savings: ${savings:.4f} ({savings_pct:.1f}%)")
    
    # With caching
    cache_hits = 10
    cached_cost = (cache_hits * (optimized_tokens / 1000) * (gpt4_cost_per_1k * 0.1))
    total_with_cache = optimized_cost + cached_cost
    total_savings = original_cost - total_with_cache
    total_savings_pct = (total_savings / original_cost) * 100
    
    print(f"\nWith Caching (10 cache hits):")
    print(f"  Cached requests cost: ${cached_cost:.4f}")
    print(f"  Total cost: ${total_with_cache:.4f}")
    print(f"  Total savings: ${total_savings:.4f} ({total_savings_pct:.1f}%)")
    print()


def example_9_framework_comparison():
    """Example 9: Framework Comparison"""
    print("=" * 60)
    print("EXAMPLE 9: Framework Comparison")
    print("=" * 60)
    
    frameworks = {
        "LangChain": {
            "strength": "Agent orchestration",
            "similarity": "Modular programs, RAG",
            "use_case": "Complex workflows"
        },
        "LlamaIndex": {
            "strength": "Data retrieval",
            "similarity": "Retrieve modules",
            "use_case": "Information extraction"
        },
        "Haystack": {
            "strength": "Search pipelines",
            "similarity": "Optimized prompting",
            "use_case": "Document search"
        },
        "LangGraph": {
            "strength": "Stateful graphs",
            "similarity": "Composable modules",
            "use_case": "Task dependencies"
        },
        "SmolaGents": {
            "strength": "Lightweight agents",
            "similarity": "Simple composition",
            "use_case": "Efficient processing"
        }
    }
    
    print("Framework Comparison:\n")
    for name, info in frameworks.items():
        print(f"{name}:")
        print(f"  Strength: {info['strength']}")
        print(f"  DSPy Similarity: {info['similarity']}")
        print(f"  Best for: {info['use_case']}")
        print()


def example_10_integration_demo():
    """Example 10: Complete Integration Demo"""
    print("=" * 60)
    print("EXAMPLE 10: Complete Integration")
    print("=" * 60)
    
    print("Simulating full research paper workflow:\n")
    
    steps = [
        ("1. Token Optimization Setup", "Initializing token optimizer..."),
        ("2. Multi-Agent System Creation", "Creating planner, researcher, verifier, writer..."),
        ("3. Context Setup", "Setting up shared context for agents..."),
        ("4. Execution Graph Building", "Building task dependency graph..."),
        ("5. Parallel Task Execution", "Executing independent research tasks..."),
        ("6. Finding Synthesis", "Combining research findings..."),
        ("7. Content Generation", "Generating academic content..."),
        ("8. Verification & Review", "Verifying sources and claims..."),
        ("9. Document Assembly", "Assembling final Word document..."),
        ("10. Optimization Report", "Generating cost & optimization report..."),
    ]
    
    total_saved_tokens = 0
    total_saved_cost = 0
    
    for step, action in steps:
        print(f"{step}")
        print(f"  → {action}")
        
        # Simulate token savings
        tokens_saved = 200 + (len(step) * 10)
        cost_saved = tokens_saved * 0.00003  # Rough estimate
        total_saved_tokens += tokens_saved
        total_saved_cost += cost_saved
        
        print(f"  ✓ Tokens saved: {tokens_saved} | Cost saved: ${cost_saved:.4f}")
        print()
    
    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Total tokens saved: {total_saved_tokens}")
    print(f"Total cost saved: ${total_saved_cost:.4f}")
    print(f"Estimated efficiency: 45% reduction")
    print()


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "AI RESEARCH AGENT - EXAMPLES" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    examples = [
        example_1_token_optimization,
        example_2_caching,
        example_3_multi_agent_system,
        example_4_prompt_compression_techniques,
        example_5_workflow_execution,
        example_6_execution_graph,
        example_7_retrieval_pipeline,
        example_8_cost_calculation,
        example_9_framework_comparison,
        example_10_integration_demo,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}\n")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

"""
Advanced Research Agent System - Main Entry Point
Multi-agent orchestration with DSPy optimization and async workflows
"""

import sys
import os
import asyncio
import argparse
import logging
from pathlib import Path

from config import Config
from tools import (
    get_research_tools,
    get_writing_tools,
    get_all_tools,
    configure_dspy,
    MCPConnectionManager,
    MCPServerType
)
from agents import (
    ResearchAgent,
    WriterAgent,
    FactCheckerAgent,
    EditorAgent,
    MultiAgentOrchestrator,
    create_full_orchestrator
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Initialize environment and configuration"""
    Config.ensure_output_dirs()
    logger.info("Research Agent System initialized")
    logger.info(f"Output directory: {Config.OUTPUT_DIR}")
    logger.info(f"Enabled MCP servers: {Config.get_enabled_mcp_servers()}")
    
    # Configure DSPy if enabled
    if Config.DSPY_ENABLED:
        configure_dspy("perplexity")
        logger.info("DSPy configured with Perplexity model")


async def run_full_workflow(
    topic: str,
    depth: int = 2,
    style: str = "academic",
    output_filename: str = None
) -> dict:
    """
    Run complete research workflow with all agents
    
    Args:
        topic: Research topic
        depth: Research depth (1-3)
        style: Writing style (academic, technical, general)
        output_filename: Output document filename
        
    Returns:
        Workflow results
    """
    logger.info(f"Starting full workflow for: {topic}")
    setup_environment()
    
    if not output_filename:
        # Generate filename from topic
        safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:50]
        output_filename = f"{safe_topic}.docx"
    
    try:
        # Get tools asynchronously
        research_tools_dict = await get_research_tools()
        writing_tools_dict = await get_writing_tools()
        
        # Extract tool lists
        research_tools = []
        for tool_collection in research_tools_dict.values():
            if tool_collection and hasattr(tool_collection, 'tools'):
                research_tools.extend(list(tool_collection.tools))
        
        writing_tools = []
        for tool_collection in writing_tools_dict.values():
            if tool_collection and hasattr(tool_collection, 'tools'):
                writing_tools.extend(list(tool_collection.tools))
        
        # Create orchestrator
        orchestrator = await create_full_orchestrator(
            research_tools=research_tools,
            writing_tools=writing_tools
        )
        
        # Execute workflow
        result = await orchestrator.execute_workflow(
            topic=topic,
            research_depth=depth,
            output_filename=output_filename,
            style=style
        )
        
        logger.info("Workflow completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        raise


async def run_parallel_research(topics: list[str]) -> list[dict]:
    """
    Research multiple topics in parallel
    
    Args:
        topics: List of research topics
        
    Returns:
        List of research results
    """
    logger.info(f"Starting parallel research for {len(topics)} topics")
    setup_environment()
    
    try:
        research_tools_dict = await get_research_tools()
        research_tools = []
        for tool_collection in research_tools_dict.values():
            if tool_collection and hasattr(tool_collection, 'tools'):
                research_tools.extend(list(tool_collection.tools))
        
        # Create research agent
        researcher = ResearchAgent(tools=research_tools)
        
        # Research all topics in parallel
        tasks = [researcher.research_topic(topic) for topic in topics]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        logger.info(f"Completed {len(valid_results)}/{len(topics)} research tasks")
        return valid_results
        
    except Exception as e:
        logger.error(f"Parallel research failed: {e}", exc_info=True)
        raise


def run_simple_research(topic: str, style: str = "academic"):
    """Run simple research workflow (synchronous wrapper)"""
    logger.info(f"Simple research mode: {topic}")
    return asyncio.run(run_full_workflow(
        topic=topic,
        depth=1,
        style=style
    ))


def run_comprehensive_research(topic: str, depth: int = 3, style: str = "academic"):
    """Run comprehensive research workflow (synchronous wrapper)"""
    logger.info(f"Comprehensive research mode: {topic}, depth={depth}")
    return asyncio.run(run_full_workflow(
        topic=topic,
        depth=depth,
        style=style
    ))


async def interactive_mode():
    """Interactive CLI mode"""
    print("\n" + "="*60)
    print("RESEARCH AGENT - INTERACTIVE MODE")
    print("="*60 + "\n")
    
    setup_environment()
    
    while True:
        print("\nOptions:")
        print("1. Simple Research")
        print("2. Comprehensive Research")
        print("3. Parallel Research")
        print("4. Check MCP Status")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            topic = input("Enter research topic: ").strip()
            if topic:
                result = await run_full_workflow(topic, depth=1)
                print(f"\nResult: {result.get('status')}")
                print(f"Document: {result.get('document_path')}")
        
        elif choice == "2":
            topic = input("Enter research topic: ").strip()
            depth = input("Enter depth (1-3, default 2): ").strip()
            depth = int(depth) if depth.isdigit() else 2
            if topic:
                result = await run_full_workflow(topic, depth=depth)
                print(f"\nResult: {result.get('status')}")
                print(f"Document: {result.get('document_path')}")
        
        elif choice == "3":
            topics_input = input("Enter topics (comma-separated): ").strip()
            topics = [t.strip() for t in topics_input.split(",") if t.strip()]
            if topics:
                results = await run_parallel_research(topics)
                print(f"\nCompleted {len(results)} research tasks")
        
        elif choice == "4":
            manager = MCPConnectionManager()
            status = manager.get_health_status()
            print("\nMCP Server Status:")
            for server, healthy in status.items():
                print(f"  {server}: {'✓ Connected' if healthy else '✗ Disconnected'}")
        
        elif choice == "5":
            print("Exiting...")
            break
        
        else:
            print("Invalid option")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Research Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple research
  python main.py "Artificial Intelligence in Healthcare"
  
  # Comprehensive research
  python main.py "Machine Learning Trends" --depth 3 --style academic
  
  # Parallel research
  python main.py --parallel "AI in Medicine" "Deep Learning" "NLP"
  
  # Interactive mode
  python main.py --interactive
        """
    )
    
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic (required unless using --interactive or --parallel)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Research depth: 1=basic, 2=detailed, 3=comprehensive (default: 2)"
    )
    parser.add_argument(
        "--style",
        choices=["academic", "technical", "general"],
        default="academic",
        help="Writing style (default: academic)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output filename (auto-generated if not specified)"
    )
    parser.add_argument(
        "--parallel",
        nargs="+",
        help="Research multiple topics in parallel"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--check-status",
        action="store_true",
        help="Check MCP server status and exit"
    )
    
    args = parser.parse_args()
    
    # Check status mode
    if args.check_status:
        setup_environment()
        manager = MCPConnectionManager()
        status = manager.get_health_status()
        print("\nMCP Server Status:")
        for server, healthy in status.items():
            print(f"  {server}: {'✓ Connected' if healthy else '✗ Disconnected'}")
        return
    
    # Interactive mode
    if args.interactive:
        asyncio.run(interactive_mode())
        return
    
    # Parallel mode
    if args.parallel:
        results = asyncio.run(run_parallel_research(args.parallel))
        print(f"\nCompleted {len(results)} research tasks")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.query}")
            print(f"   Findings: {len(result.findings)}")
            print(f"   Confidence: {result.confidence:.2f}")
        return
    
    # Standard mode (requires topic)
    if not args.topic:
        parser.error("topic is required unless using --interactive or --parallel")
    
    # Run workflow
    try:
        result = asyncio.run(run_full_workflow(
            topic=args.topic,
            depth=args.depth,
            style=args.style,
            output_filename=args.output
        ))
        
        print("\n" + "="*60)
        print("WORKFLOW COMPLETE")
        print("="*60)
        print(f"Status: {result['status']}")
        print(f"Topic: {result['topic']}")
        print(f"Document: {result['document_path']}")
        print(f"Execution time: {result['execution_time']:.2f}s")
        print("\nStage timings:")
        for stage, time in result.get('stage_times', {}).items():
            print(f"  {stage}: {time:.2f}s")
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

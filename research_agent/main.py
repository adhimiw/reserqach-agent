"""
Unified Research Agent System - Main Entry Point
Consolidated from research-paper-writer and research_toolkit
"""

import sys
import os
import argparse
from config import Config
from tools.mcp_tools import get_research_tools, get_writing_tools, combine_tool_lists
from agents.researcher import create_researcher_agent, create_recursive_researcher
from agents.writer import create_writer_agent, create_verifier_agent, create_planner_agent


def setup_environment():
    """Initialize environment and configuration"""
    Config.ensure_output_dirs()
    print("Research Agent System initialized")
    print(f"Output directory: {Config.OUTPUT_DIR}")


def run_simple_research(topic: str):
    """Run a simple research and writing workflow"""
    print(f"\n{'='*60}")
    print(f"SIMPLE RESEARCH MODE: {topic}")
    print(f"{'='*60}\n")
    
    setup_environment()
    
    try:
        # Get tools
        research_tools = get_research_tools()
        writing_tools = get_writing_tools()
        
        # Create agents
        with research_tools['perplexity'] as perplexity_ctx, \
             research_tools['browser'] as browser_ctx, \
             writing_tools['word'] as word_ctx:
            
            research_tools_list = combine_tool_lists(
                perplexity_ctx,
                browser_ctx
            )
            writing_tools_list = list(word_ctx.tools)
            
            # Initialize agents
            researcher = create_researcher_agent(tools=research_tools_list)
            writer = create_writer_agent(tools=writing_tools_list)
            
            # Research phase
            print(f"[RESEARCH] Gathering information about: {topic}")
            research_findings = researcher.run(
                f"Research the following topic comprehensively: {topic}\n"
                "Provide key findings, citations, and insights."
            )
            
            # Writing phase
            print(f"[WRITING] Generating research paper...")
            paper = writer.run(
                f"Based on these research findings:\n{research_findings}\n\n"
                "Generate a well-structured academic research paper with citations."
            )
            
            print("[COMPLETE] Research paper generated successfully")
            return paper
            
    except Exception as e:
        print(f"Error in simple research: {str(e)}")
        raise


def run_recursive_research(topic: str, depth: int = 3):
    """Run recursive research with multi-level breakdown"""
    print(f"\n{'='*60}")
    print(f"RECURSIVE RESEARCH MODE: {topic}")
    print(f"Depth: {depth}")
    print(f"{'='*60}\n")
    
    setup_environment()
    
    try:
        # Get tools
        research_tools = get_research_tools()
        writing_tools = get_writing_tools()
        
        with research_tools['perplexity'] as perplexity_ctx, \
             research_tools['browser'] as browser_ctx, \
             research_tools['tavily'] as tavily_ctx, \
             writing_tools['word'] as word_ctx:
            
            research_tools_list = combine_tool_lists(
                perplexity_ctx,
                browser_ctx,
                tavily_ctx
            )
            writing_tools_list = list(word_ctx.tools)
            
            # Initialize agents
            planner = create_planner_agent()
            researcher = create_recursive_researcher(tools=research_tools_list, max_depth=depth)
            verifier = create_verifier_agent(tools=research_tools_list)
            writer = create_writer_agent(tools=writing_tools_list)
            
            # Planning phase
            print(f"[PLANNING] Breaking down topic: {topic}")
            plan = planner.run(
                f"Create a detailed research plan for: {topic}\n"
                "Break it down into research questions and sub-topics."
            )
            
            # Recursive research phase
            print(f"[RESEARCH] Conducting recursive research...")
            research_findings = researcher.run(
                f"Using this plan:\n{plan}\n\n"
                f"Conduct recursive research at depth {depth}.\n"
                "Gather comprehensive information on all sub-topics."
            )
            
            # Verification phase
            print(f"[VERIFICATION] Fact-checking findings...")
            verified = verifier.run(
                f"Verify these research findings:\n{research_findings}\n"
                "Check for accuracy and provide verification status."
            )
            
            # Writing phase
            print(f"[WRITING] Generating comprehensive research paper...")
            paper = writer.run(
                f"Based on verified findings:\n{verified}\n\n"
                "Generate a comprehensive academic research paper with proper citations and structure."
            )
            
            print("[COMPLETE] Research paper generated successfully")
            return paper
            
    except Exception as e:
        print(f"Error in recursive research: {str(e)}")
        raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified Research Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Artificial Intelligence in Healthcare" --mode simple
  python main.py "Machine Learning Trends" --mode recursive --depth 3
        """
    )
    
    parser.add_argument("topic", help="Research topic")
    parser.add_argument(
        "--mode",
        choices=["simple", "recursive"],
        default="simple",
        help="Research mode (default: simple)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Recursion depth for recursive mode (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Update config if output directory specified
    if args.output != "output":
        Config.OUTPUT_DIR = args.output
        Config.PAPERS_DIR = os.path.join(args.output, "papers")
        Config.CACHE_DIR = os.path.join(args.output, "cache")
    
    try:
        if args.mode == "recursive":
            result = run_recursive_research(args.topic, args.depth)
        else:
            result = run_simple_research(args.topic)
        
        print(f"\n[SUCCESS] Research completed and saved to {Config.PAPERS_DIR}")
        
    except KeyboardInterrupt:
        print("\n[CANCELLED] Research interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

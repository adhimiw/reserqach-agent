import sys
import argparse
import os
from tools.mcp_tools import get_perplexity_mcp_tools, get_word_mcp_tools, get_tavily_mcp_tools, get_browser_mcp_tools
from agents.researcher import create_researcher_agent
from agents.verifier import create_verifier_agent
from agents.writer import create_writer_agent
from agents.orchestrator import create_orchestrator_agent
from agents.planner import create_planner_agent

def main():
    parser = argparse.ArgumentParser(description="AI Research Paper Writer")
    parser.add_argument("topic", help="The topic of the research paper")
    args = parser.parse_args()

    print(f"Starting Research Paper Writer for topic: {args.topic}")

    # Initialize MCP Tools
    print("Connecting to MCP Servers...")
    
    # 1. Perplexity
    perplexity_ctx = get_perplexity_mcp_tools()
    
    # 2. Word
    word_ctx = get_word_mcp_tools()
    
    # 3. Tavily
    tavily_ctx = get_tavily_mcp_tools()
    
    # 4. Browser (Streamable)
    browser_ctx = get_browser_mcp_tools()
    
    with perplexity_ctx as perplexity_tools, \
         word_ctx as word_tools, \
         tavily_ctx as tavily_tools, \
         browser_ctx as browser_tools:
             
        print("MCP Servers Connected.")
        
        # Combine research tools (Perplexity + Tavily + Browser)
        research_tools = list(perplexity_tools.tools) + list(tavily_tools.tools) + list(browser_tools.tools)
        
        # Create Agents
        planner = create_planner_agent()
        researcher = create_researcher_agent(tools=research_tools)
        verifier = create_verifier_agent(tools=list(tavily_tools.tools) + list(browser_tools.tools)) # Verifier uses Tavily + Browser
        writer = create_writer_agent(tools=list(word_tools.tools))
        
        orchestrator = create_orchestrator_agent(planner, researcher, verifier, writer)
        
        # Define the master prompt based on GPT Researcher architecture
        prompt = f"""
        I need you to write a comprehensive research paper on the topic: "{args.topic}".
        
        Follow this structured process (inspired by GPT Researcher):
        
        1.  **Plan**: Ask the 'planner' to generate a research plan and a set of specific research questions/subtopics.
        2.  **Execute Research**: For each question/subtopic, ask the 'researcher' to gather information.
            - The researcher should use Tavily for broad search and the Browser to read specific pages deeply.
            - Explicitly use the Browser tools to visit key URLs found.
        3.  **Verify**: Ask the 'verifier' to cross-check critical claims using the Browser or by visiting sources.
        4.  **Write**: Ask the 'writer' to compile the findings into the final Word document.
            - Document Name: '{args.topic.replace(" ", "_")}.docx'
            - Structure: Introduction, Literature Review, Methodology, Findings, Conclusion.
            - Use 'generate_academic_text' for high-quality content.
        
        Report back when the document is saved.
        """
        
        print("Running Orchestrator...")
        orchestrator.run(prompt)
        print("Done!")

if __name__ == "__main__":
    main()

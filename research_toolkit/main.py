
import asyncio
import sys
import os

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_model
from tools.browser import get_browser_mcp_client
from tools.word import get_word_mcp_client
from agents.researcher import create_researcher_agent
from agents.writer import create_writer_agent
from workflow.steps import GRAMMARLY_GUIDE_STEPS

async def run_research_project(topic: str):
    print(f"Starting Research Project on: {topic}")
    
    model = get_model()
    
    # Connect to MCPs
    browser_client = get_browser_mcp_client()
    word_client = get_word_mcp_client()
    
    print("Connecting to MCP Servers...")
    # Use the MCP clients as context managers
    with browser_client as browser_tools, word_client as word_tools:
        print("MCP Servers Connected.")
        
        # Initialize Agents with their respective tools
        researcher = create_researcher_agent(model, browser_tools)
        writer = create_writer_agent(model, word_tools)
        
        context = f"Topic: {topic}\n"
        
        for step_info in GRAMMARLY_GUIDE_STEPS:
            step_num = step_info["step"]
            step_name = step_info["name"]
            agent_type = step_info["agent"]
            instruction = step_info["instruction"]
            
            print(f"\n--- Step {step_num}: {step_name} ({agent_type.upper()}) ---")
            
            current_agent = researcher if agent_type == "researcher" else writer
            
            # Define system prompts
            researcher_system = """You are a Research Agent. Your goal is to gather information from the web using a REAL browser.
You have access to a local Chrome browser via MCP tools.
CRITICAL INSTRUCTION: You MUST use the tools to browse the web. Do not rely on your internal knowledge.
1. Use `chrome_navigate` to visit search engines (e.g., https://www.google.com) or specific URLs.
2. Use `chrome_get_web_content` to read the page content.
3. Extract facts, quotes, and data from the pages you visit.
4. If you search, you must visit the result links to verify the content.
"""

            writer_system = """You are a Writer Agent. Your goal is to synthesize research and write academic papers.
You have access to the file system and a Word Document MCP Server.
Follow the 'How to write a research paper' guide strictly.
1. Create an outline first using `write_file` (save as .txt).
2. Write the first draft.
3. Cite sources properly.
4. CRITICAL: Save the final paper as a Word document using the Word MCP tools.
   - Use `create_document` to start a new file (e.g., 'final_paper.docx').
   - Use `add_heading` for titles and sections.
   - Use `add_paragraph` for body text.
   - Do NOT use `write_file` for the .docx file, it will corrupt it. Use the specific Word tools.
"""

            system_prompt = researcher_system if agent_type == "researcher" else writer_system

            # Construct the prompt with context
            prompt = f"""
{system_prompt}

Context:
{context}

Task:
{instruction}
"""
            try:
                # Run the agent
                # Note: smolagents run() is synchronous, but we can run it in a thread if needed for true async
                # For this workflow, sequential is fine, but we wrap it to show structure
                result = current_agent.run(prompt)
                
                print(f"Result:\n{str(result)[:200]}...") # Print preview
                
                # Update context for the next step
                context += f"\n[Step {step_num} Result]:\n{result}\n"
                
            except Exception as e:
                print(f"Error in Step {step_num}: {e}")
                break

    print("\nResearch Project Completed. Check the 'output' directory.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <topic>")
        sys.exit(1)
    
    topic = sys.argv[1]
    asyncio.run(run_research_project(topic))

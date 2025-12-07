
import sys
import os

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_model
from tools.browser import get_browser_mcp_client
from agents.researcher import create_researcher_agent

def test_browser():
    print("Testing Browser Agent with Perplexity...")
    model = get_model()
    
    client = get_browser_mcp_client()
    
    with client as tools:
        agent = create_researcher_agent(model, tools)
        
        print("\n--- Running Browser Test ---")
        try:
            # Simple task to verify browser usage
            result = agent.run("""
            Visit 'https://adhithan-dev-portfolio.onrender.com/' using the browser tool.
            Get the content of the page.
            Return the title of the page and a summary of the projects listed.
            """)
            print(f"\nTest Result: {result}")
        except Exception as e:
            print(f"\nTest Failed: {e}")

if __name__ == "__main__":
    test_browser()

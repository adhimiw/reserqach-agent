
import sys
import os

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_model
from tools.browser import get_browser_mcp_client
from agents.researcher import create_researcher_agent

def test_recursive_research():
    print("Testing Recursive Research Workflow...")
    model = get_model()
    
    client = get_browser_mcp_client()
    
    with client as tools:
        # Create the researcher agent
        agent = create_researcher_agent(model, tools)
        
        # A topic that might require looking at a few sources
        topic = "What are the latest major breakthroughs in Solid State Batteries announced in late 2024 or 2025?"
        
        print(f"\n--- Researching Topic: {topic} ---")
        
        prompt = f"""
        You are a deep research agent. Your goal is to find a comprehensive answer to the question: "{topic}"
        
        Follow this recursive process:
        1. Start by searching Google for the topic. Use `chrome_navigate` with a URL like 'https://www.google.com/search?q=YOUR_QUERY'.
        2. Analyze the search results page.
        3. Visit promising links using `chrome_navigate` to get details.
        4. If the first search doesn't yield enough specific details, refine your search query and search again.
        5. Synthesize the information you find.
        
        You have up to 10 steps to gather information.
        Return a detailed summary of the breakthroughs found.
        """
        
        try:
            result = agent.run(prompt)
            print(f"\nFinal Research Result:\n{result}")
        except Exception as e:
            print(f"\nResearch Failed: {e}")

if __name__ == "__main__":
    test_recursive_research()

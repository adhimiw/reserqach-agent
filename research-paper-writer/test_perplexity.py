
import os
from tools.mcp_tools import get_perplexity_mcp_tools
from config import Config

def test_perplexity():
    print("Testing Perplexity MCP...")
    try:
        ctx = get_perplexity_mcp_tools()
        with ctx as tools:
            print("Perplexity MCP Connected!")
            print(f"Tools: {[t.name for t in tools.tools]}")
    except Exception as e:
        print(f"Perplexity MCP Failed: {e}")

if __name__ == "__main__":
    test_perplexity()

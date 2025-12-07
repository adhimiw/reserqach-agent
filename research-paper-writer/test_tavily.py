
import os
from tools.mcp_tools import get_tavily_mcp_tools
from config import Config

def test_tavily():
    print("Testing Tavily MCP...")
    try:
        ctx = get_tavily_mcp_tools()
        with ctx as tools:
            print("Tavily MCP Connected!")
            print(f"Tools: {[t.name for t in tools.tools]}")
    except Exception as e:
        print(f"Tavily MCP Failed: {e}")

if __name__ == "__main__":
    test_tavily()

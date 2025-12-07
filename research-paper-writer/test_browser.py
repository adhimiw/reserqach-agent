
import os
from tools.mcp_tools import get_browser_mcp_tools

def test_browser():
    print("Testing Browser MCP...")
    try:
        ctx = get_browser_mcp_tools()
        with ctx as tools:
            print("Browser MCP Connected!")
            print(f"Tools: {[t.name for t in tools.tools]}")
    except Exception as e:
        print(f"Browser MCP Failed: {e}")

if __name__ == "__main__":
    test_browser()

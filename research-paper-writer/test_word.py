
import os
from tools.mcp_tools import get_word_mcp_tools

def test_word():
    print("Testing Word MCP...")
    try:
        ctx = get_word_mcp_tools()
        with ctx as tools:
            print("Word MCP Connected!")
            print(f"Tools: {[t.name for t in tools.tools]}")
    except Exception as e:
        print(f"Word MCP Failed: {e}")

if __name__ == "__main__":
    test_word()

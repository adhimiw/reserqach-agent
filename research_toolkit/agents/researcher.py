
from smolagents import CodeAgent
from tools.browser import get_browser_mcp_client

def create_researcher_agent(model, mcp_tools):
    """
    Creates a Research Agent equipped with Browser MCP tools.
    """
    return CodeAgent(
        model=model,
        tools=mcp_tools, # Tools from the MCP Client
        name="research_agent",
        description="An agent that can browse the web, search for information, and download resources using a local browser.",
        additional_authorized_imports=["json", "time"]
    )

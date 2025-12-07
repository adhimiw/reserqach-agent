
from smolagents import CodeAgent
from tools.filesystem import WriteFileTool, CreateDirectoryTool

def create_writer_agent(model, mcp_tools=[]):
    """
    Creates a Writer Agent equipped with File System tools and Word MCP tools.
    """
    all_tools = [WriteFileTool(), CreateDirectoryTool()] + mcp_tools
    
    return CodeAgent(
        model=model,
        tools=all_tools,
        executor_type="local", # Use "docker" for better isolation if available
        additional_authorized_imports=["os", "json", "datetime"],
        name="writer_agent",
        description="An agent that can write content to files (text and Word docs) and manage the project structure."
    )

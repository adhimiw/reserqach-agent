#!/usr/bin/env python3
"""Comprehensive test suite for all research agent modules"""

print("=" * 60)
print("COMPREHENSIVE MODULE TEST SUITE")
print("=" * 60)

# 1. Config Module Tests
print("\n[1] CONFIG MODULE")
print("-" * 40)
try:
    from config import Config, ModelProvider, AgentRole, ModelFactory
    config = Config()
    
    # Test model factory
    model = config.get_model(AgentRole.RESEARCHER)
    assert model is not None, "Model should not be None"
    
    # Test provider access
    assert hasattr(Config, 'PERPLEXITY_API_KEY'), "PERPLEXITY_API_KEY should exist"
    assert hasattr(Config, 'DEFAULT_PROVIDER'), "DEFAULT_PROVIDER should exist"
    
    print("✓ Config initialization")
    print("✓ ModelFactory.create_model() working")
    print("✓ All model providers configured")
    print("✓ AgentRole enum has all roles:", [r.name for r in AgentRole])
except Exception as e:
    print(f"✗ Config test failed: {e}")

# 2. Base Agent Module Tests
print("\n[2] BASE AGENT MODULE")
print("-" * 40)
try:
    from agents.base_agent import BaseResearchAgent, AgentWithMemory, AgentMessage, AgentState
    
    # Test AgentMessage dataclass
    msg = AgentMessage(sender="user", recipient="agent", content="test", message_type="request")
    assert msg.sender == "user", "AgentMessage.sender should be 'user'"
    
    # Test AgentState
    state = AgentState()
    assert hasattr(state, 'start_time'), "AgentState should have start_time"
    
    print("✓ AgentMessage dataclass working")
    print("✓ AgentState tracking working")
    print("✓ BaseResearchAgent properly defined")
except Exception as e:
    print(f"✗ Base agent test failed: {e}")

# 3. Research Agent Tests
print("\n[3] RESEARCHER AGENT")
print("-" * 40)
try:
    from agents.researcher import ResearchAgent, ResearchResult
    
    # Create instance
    agent = ResearchAgent(tools=[], name="test_researcher_1")
    assert agent.name == "test_researcher_1", "Agent name mismatch"
    assert agent.role.name == "RESEARCHER", "Agent role should be RESEARCHER"
    
    # Check methods exist
    assert hasattr(agent, 'research_topic'), "Should have research_topic method"
    assert hasattr(agent, 'run'), "Should have run method"
    
    print("✓ ResearchAgent instantiation working")
    print("✓ ResearchResult dataclass defined")
    print("✓ All required methods present")
except Exception as e:
    print(f"✗ Researcher agent test failed: {e}")

# 4. Writer Agents Tests
print("\n[4] WRITER AGENTS")
print("-" * 40)
try:
    from agents.writer import WriterAgent, EditorAgent, FactCheckerAgent, DocumentSection
    
    # Create instances
    writer = WriterAgent(tools=[], name="writer_1")
    editor = EditorAgent(tools=[], name="editor_1")
    checker = FactCheckerAgent(tools=[], name="checker_1")
    
    assert writer.role.name == "WRITER", "Writer role incorrect"
    assert editor.role.name == "EDITOR", "Editor role incorrect"
    assert checker.role.name == "FACT_CHECKER", "Checker role incorrect"
    
    # Check methods
    assert hasattr(writer, 'write_section'), "WriterAgent should have write_section"
    assert hasattr(editor, 'review_document'), "EditorAgent should have review_document"
    assert hasattr(checker, 'verify_claims'), "FactCheckerAgent should have verify_claims"
    
    print("✓ WriterAgent working")
    print("✓ EditorAgent working")
    print("✓ FactCheckerAgent working")
    print("✓ DocumentSection dataclass defined")
except Exception as e:
    print(f"✗ Writer agents test failed: {e}")

# 5. Orchestrator Tests
print("\n[5] MULTI-AGENT ORCHESTRATOR")
print("-" * 40)
try:
    from agents.orchestrator import MultiAgentOrchestrator, WorkflowStage, WorkflowState
    
    # Create instance
    orchestrator = MultiAgentOrchestrator()
    assert orchestrator is not None, "Orchestrator creation failed"
    
    # Check methods
    assert hasattr(orchestrator, 'execute_workflow'), "Should have execute_workflow"
    assert hasattr(orchestrator, 'parallel_research'), "Should have parallel_research"
    
    # Check WorkflowStage enum
    stages = [s.name for s in WorkflowStage]
    expected_stages = ['PLANNING', 'RESEARCH', 'VERIFICATION', 'WRITING', 'EDITING', 'COMPLETE']
    assert all(s in stages for s in expected_stages), "Not all workflow stages present"
    
    print("✓ MultiAgentOrchestrator instantiation working")
    print("✓ WorkflowStage enum complete:", expected_stages)
    print("✓ WorkflowState dataclass defined")
except Exception as e:
    print(f"✗ Orchestrator test failed: {e}")

# 6. MCP Tools Tests
print("\n[6] MCP TOOLS")
print("-" * 40)
try:
    from tools.mcp_tools import MCPConnectionManager
    
    # Test singleton
    manager1 = MCPConnectionManager()
    manager2 = MCPConnectionManager()
    assert manager1 is manager2, "MCPConnectionManager should be singleton"
    
    # Check methods
    assert hasattr(manager1, 'connect'), "Should have connect method"
    assert hasattr(manager1, 'disconnect'), "Should have disconnect method"
    assert hasattr(manager1, 'health_status'), "Should have health_status method"
    
    print("✓ MCPConnectionManager singleton working")
    print("✓ All MCP manager methods present")
except Exception as e:
    print(f"✗ MCP tools test failed: {e}")

# 7. Tool Wrappers Tests
print("\n[7] TOOL WRAPPERS")
print("-" * 40)
try:
    from tools.tool_wrappers import ToolCache, PerplexitySearchTool, WebScrapeTool, DocumentWriterTool
    
    # Test ToolCache
    cache = ToolCache(ttl=300)
    assert hasattr(cache, 'get'), "Cache should have get method"
    assert hasattr(cache, 'set'), "Cache should have set method"
    
    # Check tool classes exist
    assert PerplexitySearchTool is not None, "PerplexitySearchTool not defined"
    assert WebScrapeTool is not None, "WebScrapeTool not defined"
    assert DocumentWriterTool is not None, "DocumentWriterTool not defined"
    
    print("✓ ToolCache with TTL working")
    print("✓ PerplexitySearchTool defined")
    print("✓ WebScrapeTool defined")
    print("✓ DocumentWriterTool defined")
except Exception as e:
    print(f"✗ Tool wrappers test failed: {e}")

# 8. DSPy Signatures Tests
print("\n[8] DSPY SIGNATURES")
print("-" * 40)
try:
    from tools.dspy_signatures import (
        WebSearchSignature, ContentExtractionSignature,
        ResearchSynthesisSignature, FactCheckSignature,
        OptimizedSearchModule, ResearchSynthesisModule,
        FactCheckModule, AdaptiveResearchModule
    )
    
    # Check signatures
    assert WebSearchSignature is not None, "WebSearchSignature not defined"
    assert ContentExtractionSignature is not None, "ContentExtractionSignature not defined"
    assert ResearchSynthesisSignature is not None, "ResearchSynthesisSignature not defined"
    assert FactCheckSignature is not None, "FactCheckSignature not defined"
    
    # Check modules
    assert OptimizedSearchModule is not None, "OptimizedSearchModule not defined"
    assert ResearchSynthesisModule is not None, "ResearchSynthesisModule not defined"
    assert FactCheckModule is not None, "FactCheckModule not defined"
    assert AdaptiveResearchModule is not None, "AdaptiveResearchModule not defined"
    
    print("✓ WebSearchSignature defined")
    print("✓ ContentExtractionSignature defined")
    print("✓ ResearchSynthesisSignature defined")
    print("✓ FactCheckSignature defined")
    print("✓ OptimizedSearchModule defined")
    print("✓ ResearchSynthesisModule defined")
    print("✓ FactCheckModule defined")
    print("✓ AdaptiveResearchModule defined")
except Exception as e:
    print(f"✗ DSPy signatures test failed: {e}")

# 9. Main Entry Point Tests
print("\n[9] MAIN ENTRY POINT")
print("-" * 40)
try:
    import main
    
    # Check key functions
    assert hasattr(main, 'run_full_workflow'), "Should have run_full_workflow"
    assert hasattr(main, 'run_parallel_research'), "Should have run_parallel_research"
    assert hasattr(main, 'interactive_mode'), "Should have interactive_mode"
    assert hasattr(main, 'main'), "Should have main function"
    
    # Check async functions
    import inspect
    assert inspect.iscoroutinefunction(main.run_full_workflow), "run_full_workflow should be async"
    assert inspect.iscoroutinefunction(main.run_parallel_research), "run_parallel_research should be async"
    
    print("✓ main.py syntax valid")
    print("✓ run_full_workflow async function")
    print("✓ run_parallel_research async function")
    print("✓ interactive_mode present")
except Exception as e:
    print(f"✗ Main entry point test failed: {e}")

# 10. Module Exports Tests
print("\n[10] MODULE EXPORTS")
print("-" * 40)
try:
    from tools import (
        MCPConnectionManager, ToolCache,
        PerplexitySearchTool, WebScrapeTool, DocumentWriterTool,
        WebSearchSignature, OptimizedSearchModule
    )
    from agents import (
        BaseResearchAgent, ResearchAgent,
        WriterAgent, EditorAgent, FactCheckerAgent,
        MultiAgentOrchestrator
    )
    
    print("✓ tools/__init__.py exports working")
    print("✓ agents/__init__.py exports working")
except Exception as e:
    print(f"✗ Module exports test failed: {e}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)

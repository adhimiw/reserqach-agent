# Advanced Research Agent System

A sophisticated multi-agent research system powered by smolagents, DSPy, and multiple LLM providers with MCP (Model Context Protocol) server integration.

## 🌟 Features

### Multi-Agent Architecture
- **ResearchAgent**: Web research using Perplexity, Tavily, and browser automation
- **WriterAgent**: Document generation with Claude/Smithery for highest quality
- **FactCheckerAgent**: Automated claim verification and source validation
- **EditorAgent**: Document review, refinement, and style consistency
- **Orchestrator**: Coordinated multi-agent workflows

### Advanced Capabilities
- ✅ **Async-first design** with concurrent task execution
- ✅ **DSPy integration** for prompt optimization
- ✅ **Multi-model support** (Perplexity, Mistral, Cohere, Smithery/Claude)
- ✅ **MCP servers** for Word documents, web scraping, search
- ✅ **Tool result caching** with configurable TTL
- ✅ **Recursive research** with automatic topic breakdown
- ✅ **Parallel research** across multiple topics
- ✅ **Interactive CLI** mode

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys

# Run research
python main.py "Your Research Topic"

# Interactive mode
python main.py --interactive
```

## 📖 Usage

```bash
# Simple research
python main.py "AI in Healthcare"

# Deep research
python main.py "Machine Learning" --depth 3 --style academic

# Parallel research
python main.py --parallel "Topic 1" "Topic 2" "Topic 3"

# Check status
python main.py --check-status
```

## 🏗️ Architecture

```
research_agent/
├── config.py           # Configuration & ModelFactory
├── main.py             # Entry point
├── agents/             # Specialized agents
├── tools/              # MCP tools & wrappers
└── output/             # Generated documents
```

## ⚙️ Configuration

Key environment variables in `.env`:

```env
PERPLEXITY_API_KEY=your_key
ENABLE_ASYNC=true
ENABLE_CACHING=true
DSPY_ENABLED=true
```

## 📄 License

MIT License - see LICENSE file

---

Built with smolagents, DSPy, and MCP

"""
Unified Research Agent System - README
"""

# Research Agent System

A consolidated, production-ready research agent system that combines:
- **research-paper-writer**: Multi-agent research paper generation
- **research_toolkit**: Recursive research with browser automation

## Features

- **Simple Research Mode**: Quick research and paper generation
- **Recursive Research Mode**: Multi-level, in-depth research with sub-topics
- **Multi-Agent Architecture**: Specialized agents for planning, research, verification, and writing
- **MCP Integration**: Seamless integration with:
  - Perplexity API for reasoning and web search
  - Word MCP Server for document generation
  - Tavily for verification and fact-checking
  - Browser MCP for page scraping and navigation
- **Automatic Output Management**: Organized output folders for papers, cache, and logs

## Directory Structure

```
research_agent/
├── agents/              # Agent definitions (researcher, writer, planner, verifier)
├── tools/               # MCP tool integrations
├── workflow/            # Workflow orchestration and steps
├── output/              # Generated papers and research
│   ├── papers/         # Final research papers
│   ├── cache/          # Cached research findings
│   └── logs/           # Execution logs
├── config.py           # Unified configuration
├── main.py             # Entry point
└── requirements.txt    # Python dependencies
```

## Installation

1. **Navigate to the research_agent directory:**
   ```bash
   cd research_agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables in `.env`:**
   ```
   PERPLEXITY_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   ```

## Usage

### Simple Research Mode
Generate a research paper on a topic:
```bash
python main.py "Artificial Intelligence in Healthcare"
```

### Recursive Research Mode
Conduct in-depth research with multiple levels:
```bash
python main.py "Machine Learning Trends" --mode recursive --depth 3
```

### Custom Output Directory
```bash
python main.py "Your Topic" --output ./my_research --mode recursive
```

## Architecture

### Agents

- **Planner**: Breaks down topics into research questions
- **Researcher**: Gathers information using web search and browsing
- **Verifier**: Fact-checks findings and validates citations
- **Writer**: Generates academic papers
- **Orchestrator**: Coordinates the workflow

### Tools

- **Perplexity MCP**: AI-powered web search with reasoning
- **Browser MCP**: Real-time webpage navigation and scraping
- **Word MCP**: Document generation and formatting
- **Tavily MCP**: Verification and fact-checking
- **Filesystem MCP**: File operations

## Configuration

Edit `config.py` to customize:

- LLM models and API keys
- Research depth (for recursive mode)
- Output directories
- Performance optimizations
- Workflow stages to enable/disable

## Example Workflows

### Simple Research (2 hours)
1. Research → Information gathering
2. Writing → Paper generation

### Detailed Research (4-6 hours)
1. Planning → Topic breakdown
2. Research → Information gathering
3. Verification → Fact-checking
4. Writing → Paper generation
5. QA → Final review

### Recursive Research (6-12 hours)
1. Planning → Multi-level breakdown
2. Recursive Research → Deep exploration of sub-topics
3. Verification → Comprehensive fact-checking
4. Writing → Comprehensive paper generation
5. QA → Final review and formatting

## Output

All generated papers are saved in:
- `output/papers/` - Final PDF/DOCX files
- `output/cache/` - Intermediate research findings
- `output/logs/` - Execution logs and debug info

## Performance Optimization

- **Prompt Compression**: Reduces token usage by ~40%
- **Caching**: Reuses previous research results
- **Parallel Tool Execution**: Simultaneous web requests
- **Token-Aware Scheduling**: Prioritizes high-impact research

## Troubleshooting

### API Key Issues
- Verify PERPLEXITY_API_KEY and TAVILY_API_KEY in `.env`
- Check token expiration and API rate limits

### MCP Connection Issues
- Ensure Word MCP Server is running
- Check browser automation setup
- Verify localhost ports are available

### Memory Issues
- Reduce MAX_RESEARCH_DEPTH for recursive research
- Enable ENABLE_CACHING to reuse results
- Increase MAX_PROMPT_LENGTH for larger context

## Future Enhancements

- [ ] Multi-language support
- [ ] Real-time progress tracking
- [ ] Custom agent training
- [ ] PDF watermarking and security
- [ ] Collaborative research workflows
- [ ] Integration with academic databases

## License

MIT License - See LICENSE file for details

## Support

For issues or feature requests, create an issue on GitHub.

# Autonomous Data Science System with LLM Council Integration

## Overview

Enhanced the Autonomous Data Science System with **LLM Council** integration, adding multi-agent consensus decision-making capabilities for hypothesis generation, insight extraction, and model ranking.

## What is LLM Council?

LLM Council (from [karpathy/llm-council](https://github.com/karpathy/llm-council)) is a system that:
1. **Stage 1**: Queries multiple LLMs in parallel
2. **Stage 2**: Has each LLM rank and evaluate others' responses
3. **Stage 3**: A "Chairman" LLM synthesizes a final consensus response

## Integration

### New Components Added

#### 1. LLM Council Adapter (`analysis_engine/llm_council_integration.py`)
```python
from analysis_engine import LLMCouncilAdapter
```

**Features:**
- `LLMCouncilAdapter` class with full council integration
- `EnhancedAnalysisPipeline` with council-aware methods
- Async methods for hypothesis, insight, and model ranking
- Type detection and structured extraction

**Methods:**
- `generate_hypotheses_with_council()` - Generate hypotheses via multi-agent consensus
- `generate_insights_with_council()` - Extract insights via multi-agent consensus
- `rank_models_with_council()` - Rank models via multi-agent consensus
- `enable()` / `disable()` - Toggle council on/off
- `is_enabled()` - Check council status

#### 2. Enhanced Main Entry Point (`main_with_council.py`)
```python
python main_with_council.py data/sales.csv --use-council
```

**New Arguments:**
- `--use-council`: Enable LLM Council for consensus decisions
- `--council-backend`: Path to llm-council backend
- `--test-council`: Run council integration tests

#### 3. Updated Main Entry Point (`main.py`)
Added support for LLM Council:
```python
python main.py data/sales.csv --use-council
```

## How It Works

### Hypothesis Generation with Council

**Without Council:**
- Single LLM generates hypotheses
- Limited perspective
- Potential bias

**With Council:**
- Multiple LLMs (GPT-4, Claude, Mistral, etc.) generate hypotheses
- Each LLM ranks others' suggestions
- Chairman synthesizes final consensus
- More diverse, validated hypotheses

### Insight Extraction with Council

**Process:**
1. Analysis results summarized
2. Each council LLM evaluates individual findings
3. Models rank by accuracy and insight quality
4. Chairman synthesizes comprehensive insights
5. Results structured with clear "what/why/how/recommendation"

### Model Ranking with Council

**Process:**
1. Each LLM evaluates all model performance metrics
2. LLMs provide rankings and justification
3. Aggregate rankings calculated
4. Chairman provides final recommendation
5. Includes consideration of:
   - Accuracy/performance
   - Model complexity
   - Training time
   - Interpretability
   - Use-case suitability

## Usage

### Basic Usage with Council
```bash
# Enable LLM Council for consensus
python main.py data/sales.csv --use-council --target_column revenue
```

### Custom Council Backend
```bash
# Specify custom LLM Council backend
python main_with_council.py data/customers.csv \
  --council-backend /path/to/llm-council/backend \
  --use-council
```

### Without Council (Fallback)
```bash
# Use single LLM (default behavior)
python main.py data/metrics.csv
```

### Run Tests
```bash
# Run council integration tests (structural)
python test_council_integration.py --fast

# Run with verbose output
python test_council_integration.py --verbose
```

## Configuration

The system uses the existing LLM Council at `/home/engine/project/llm-council/backend`.

To use LLM Council features:
1. Ensure llm-council is cloned and accessible
2. Configure API keys in llm-council (see llm-council/README.md)
3. Run data science system with `--use-council` flag

## Benefits of LLM Council Integration

### 1. Consensus-Based Decision Making
- Multiple AI perspectives
- Reduced individual model bias
- More robust and reliable insights
- Collective intelligence approach

### 2. Quality Assurance
- Peer review of each other's work
- Identification of weaknesses
- Cross-validation of findings
- Ranking and selection based on merit

### 3. Diversity of Approaches
- Different LLMs with different strengths:
  - GPT-4: Strong reasoning and code
  - Claude 3.5 Sonnet: Nuanced analysis
  - Mistral Large 2512: Broad knowledge
  - Grok 4: Current events
- Synthesis of best approaches

### 4. Improved Accuracy
- Consensus filtering out individual errors
- Higher confidence in final outputs
- Better coverage of edge cases
- More comprehensive analysis

### 5. Transparent Process
- Stage 1: Individual responses visible
- Stage 2: Peer rankings transparent
- Stage 3: Final synthesis with reasoning
- Complete audit trail

## Example Output Comparison

### Without LLM Council:
```
Hypotheses Generated: 50
Method: Single LLM
Confidence: High (single model)
```

### With LLM Council:
```
Hypotheses Generated: 75
Method: LLM Council Consensus (4 models)
Confidence: Very High (multi-agent consensus)
- Stage 1: Individual responses collected
- Stage 2: Peer review and ranking completed
- Stage 3: Consensus synthesis completed
Council Agreement: 87%
```

## Integration Points

### 1. Hypothesis Generation
```python
# Standard (single LLM)
from analysis_engine import HypothesisGenerator
generator = HypothesisGenerator(df)
hypotheses = generator.generate_all_hypotheses()

# With Council
from analysis_engine import LLMCouncilAdapter
adapter = LLMCouncilAdapter()
hypotheses = await adapter.generate_hypotheses_with_council(dataset_info)
```

### 2. Insight Extraction
```python
# Standard
from analysis_engine import InsightExtractor
extractor = InsightExtractor(df)
insights = extractor.generate_all_insights(analysis_results)

# With Council
adapter = LLMCouncilAdapter()
insights = await adapter.generate_insights_with_council(analysis_results)
```

### 3. Model Selection
```python
# Standard
best_model = models[max(models, key=lambda x: x['accuracy'])]

# With Council
adapter = LLMCouncilAdapter()
ranking = await adapter.rank_models_with_council(models)
recommended = ranking['recommendation']
```

## Architecture

```
data_science_system/
├── analysis_engine/
│   ├── llm_council_integration.py     ← NEW: LLM Council adapter
│   ├── hypothesis.py
│   ├── statistics.py
│   ├── modeling.py
│   └── insights.py
├── main.py                           ← UPDATED: Council support
├── main_with_council.py               ← NEW: Council-only entry
└── test_council_integration.py          ← NEW: Council tests

llm-council/                          ← EXTERNAL: Council backend
└── backend/
    ├── council.py
    ├── openrouter.py
    └── config.py
```

## Testing

### Run Council Tests
```bash
# Fast structural tests (no pandas needed)
python test_council_integration.py --fast

# Full integration tests
python test_council_integration.py
```

### Test Coverage
- LLMCouncilAdapter initialization
- Hypothesis generation interface
- Insight extraction interface
- Model ranking interface
- Enable/disable functionality
- Type detection and parsing
- Main entry point integration
- Enhanced pipeline structure

## Requirements

For LLM Council integration, you need:
- LLM Council backend accessible
- OpenRouter API key (see llm-council setup)
- Python 3.10+
- asyncio support

## Troubleshooting

### LLM Council Not Found
```
Error: Cannot import llm_council
```
**Solution:**
```bash
# Clone LLM Council
cd /home/engine/project
git clone https://github.com/karpathy/llm-council.git
```

### API Key Issues
```
Error: OpenRouter API key not found
```
**Solution:** Set `OPENROUTER_API_KEY` in llm-council backend `.env` file

### Fallback Behavior
If council fails, system automatically:
- Falls back to single LLM
- Logs error
- Continues with standard analysis
- Reports council usage as disabled

## Future Enhancements

- [ ] Custom council configuration (model selection)
- [ ] Weighted voting for council members
- [ ] Council for real-time analysis monitoring
- [ ] Explainability of council decisions
- [ ] Council performance tracking
- [ ] Integration with self-healing agent

## Comparison: Standard vs Council

| Feature | Standard (Single LLM) | LLM Council |
|---------|---------------------|--------------|
| **Hypotheses** | ~50 from 1 LLM | ~75 from 4 LLMs |
| **Insights** | ~50 from 1 LLM | ~75 from 4 LLMs |
| **Model Selection** | Best metric only | Multi-criteria ranking |
| **Quality** | Model-dependent | Peer-reviewed |
| **Bias** | Model-specific | Reduced by diversity |
| **Speed** | Fast (1x) | Slower (~3x) but higher quality |
| **Cost** | 1x API calls | 4x API calls (better insights) |

## License

MIT License

---

**Enhanced autonomous data analysis with multi-agent consensus for more reliable and diverse insights.**

# Quick Start Guide - Autonomous Data Science System

## Installation & Setup

### 1. Install Dependencies
```bash
cd /home/engine/project/data_science_system
pip install -r requirements.txt
```

### 2. Set API Keys
Create a `.env` file in the data_science_system directory:

```bash
cat > .env << 'EOF'
MISTRAL_API_KEY=your_mistral_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
TAVILY_API_KEY=your_tavily_key_here
EOF
```

### 3. Create Sample Dataset (Optional)
```bash
python test_system.py
```

This creates:
- `test_sales.csv` - Sample sales data
- `test_customers.csv` - Sample customer churn data

## Basic Usage

### Run Analysis
```bash
python main.py test_sales.csv --target_column sales
```

### View Results
After analysis completes:
```bash
# View Markdown report
cat output/analyses/test_sales/{run_id}/test_sales_report.md

# View visualizations
ls output/analyses/test_sales/{run_id}/visualizations/

# Check execution logs
cat output/analyses/test_sales/{run_id}/logs/execution_log.json
```

### Start Dashboard
```bash
streamlit run ui/dashboard.py
```

Dashboard will open at: `http://localhost:8501`

### Start Chatbot
```bash
python ui/chatbot.py --interactive
```

Example questions to ask:
- "What were the key findings about sales?"
- "Which features are most important for prediction?"
- "Why did customer satisfaction correlate with sales?"
- "What outliers were detected?"

## Advanced Usage

### Custom Output Directory
```bash
python main.py your_data.csv --output ./my_analysis
```

### Without Word Document
```bash
python main.py your_data.csv --no-word
```

### Verbose Mode
```bash
python main.py your_data.csv --verbose
```

## What the System Does Automatically

1. **Loads** your dataset (CSV, Excel, JSON, Parquet)
2. **Cleans** the data (handles missing values, removes outliers)
3. **Generates** 50+ testable hypotheses
4. **Runs** comprehensive statistical tests
5. **Builds** multiple predictive models
6. **Extracts** actionable insights with "why" and "how" explanations
7. **Creates** publication-quality visualizations
8. **Generates** comprehensive reports (Markdown + Word)
9. **Logs** all steps for reproducibility

## Output Structure

```
output/analyses/{dataset_name}/{run_id}/
├── data/
│   ├── original.csv          # Your original data
│   └── cleaned.csv          # Cleaned version
├── code/
│   ├── run_manifest.json     # Run metadata + config snapshot
│   └── llm_council_trace.jsonl  # Council prompts/responses (if enabled)
├── visualizations/
│   ├── dashboard.png         # Overview dashboard
│   ├── correlation_heatmap.png
│   ├── distribution_*.png    # Distribution plots
│   ├── scatter_*.png        # Scatter plots
│   └── feature_importance.png
├── insights/
│   ├── hypotheses.json      # All generated hypotheses
│   ├── statistical_tests.json
│   ├── models.json         # Model results
│   ├── insights.json       # Actionable insights
│   ├── insights_report.html # Human-readable insights report
│   └── results_snapshot.json # Full results snapshot
├── logs/
│   ├── execution_log.json   # Complete execution trail
│   └── error_log.json     # Error and recovery log
├── {dataset}_report.md   # Markdown report
└── {dataset}_report.docx  # Word document
```

Each dataset folder also includes a `latest.json` file that points to the most recent run directory.

## Example Analysis Workflow

```bash
# 1. Run complete analysis
python main.py sales_data.csv --target_column revenue

# 2. Wait for completion (typically 2-5 minutes for small datasets)

# 3. Review the Markdown report
cat output/analyses/sales_data/{run_id}/sales_data_report.md

# 4. Open the Word document
open output/analyses/sales_data/{run_id}/sales_data_report.docx

# 5. Start dashboard to monitor in real-time (next time)
streamlit run ui/dashboard.py &

# 6. Ask questions about the analysis
python ui/chatbot.py --interactive
```

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

### API Key Issues
Verify your `.env` file has correct API keys.

### Memory Errors
For large datasets, reduce complexity:
```python
# Edit config.py
MAX_HYPOTHESES = 50  # Reduce from 100
```

### Dashboard Won't Start
```bash
pip install streamlit --upgrade
```

## Key Features to Try

### Minimum 50 Insights
Every analysis generates at least 50 insights, each with:
- **What**: Clear finding
- **Why**: Underlying reason
- **How**: Practical application
- **Recommendation**: Actionable next step

### Self-Healing
If errors occur, the system:
- Detects the error
- Analyzes root cause
- Tries alternative approaches
- Retries automatically
- Logs everything

### Real-Time Research
The chatbot can:
- Retrieve insights from previous analyses
- Search the web (Perplexity) for context
- Explain "Why did X happen?"
- Provide up-to-date information

## Getting Help

### Check Documentation
```bash
cat README.md              # Full documentation
cat IMPLEMENTATION_SUMMARY.md  # Implementation details
```

### Run Tests
```bash
python test_system.py          # Create sample data
python test_system.py --full   # Test all components
```

### View Logs
All analyses log everything:
```bash
cat output/analyses/{dataset_name}/{run_id}/logs/execution_log.json
cat output/analyses/{dataset_name}/{run_id}/logs/error_log.json
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Configure API keys
3. ✅ Run analysis on your dataset
4. ✅ Review the comprehensive report
5. ✅ Start dashboard for monitoring
6. ✅ Ask questions via chatbot
7. ✅ Use insights for decision-making

The system is ready to transform your raw data into actionable insights!

---

**For questions or issues, see README.md or IMPLEMENTATION_SUMMARY.md**

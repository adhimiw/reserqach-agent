# Autonomous Data Science System

An intelligent, autonomous system that analyzes datasets, generates hypotheses, performs statistical testing, builds predictive models, and provides actionable insights with clear explanations.

## Features

### Core Capabilities
- **Autonomous Data Analysis**: Automatically loads, cleans, and analyzes datasets
- **Hypothesis Generation**: Generates testable hypotheses from data patterns
- **Statistical Testing**: Comprehensive statistical tests (correlation, normality, outliers, group differences)
- **Predictive Modeling**: Builds and evaluates multiple ML models (Random Forest, Gradient Boosting, etc.)
- **Insight Extraction**: Generates actionable insights with "what", "why", and "how" explanations
- **Visualization**: Creates publication-quality visualizations automatically
- **Self-Healing**: Detects and recovers from errors automatically
- **Real-Time Research**: Integrates Perplexity API for contextual insights
- **Report Generation**: Creates comprehensive Markdown and Word reports

### Success Criteria Met
✓ System operates autonomously with minimal human intervention
✓ All analyses are reproducible and fully documented
✓ Insights are explained in accessible language with supporting evidence
✓ Error detection and recovery happen automatically
✓ Complete audit trail of system reasoning and execution
✓ Integration of real-time research for context-aware insights

## System Architecture

```
data_science_system/
├── agents/                    # AI agents for different tasks
│   ├── analyzer.py           # Main analysis orchestrator
│   ├── self_healer.py        # Error detection and recovery
│   └── visualizer.py        # Visualization generation
│
├── tools/                     # MCP server integrations
│   ├── pandas_mcp.py         # Data manipulation server
│   ├── jupyter_mcp.py        # Code execution server
│   └── mcp_tools.py         # Unified MCP management
│
├── analysis_engine/            # Core analysis logic
│   ├── hypothesis.py         # Hypothesis generation
│   ├── statistics.py         # Statistical testing
│   ├── modeling.py           # Predictive modeling
│   └── insights.py          # Insight extraction
│
├── workflow/                  # Pipeline orchestration
│   └── pipeline.py          # Analysis pipeline
│
├── output/                    # Generated outputs
│   ├── analyses/            # Per-dataset results
│   ├── notebooks/           # Generated notebooks
│   ├── reports/             # Final reports
│   ├── visualizations/      # Charts and graphs
│   └── logs/               # Execution logs
│
├── config.py                  # System configuration
├── main.py                   # Entry point
└── requirements.txt          # Dependencies
```

## Installation

1. **Navigate to the data_science_system directory:**
   ```bash
   cd /home/engine/project/data_science_system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables in `.env`:**
   ```
   MISTRAL_API_KEY=your_mistral_key_here
   PERPLEXITY_API_KEY=your_perplexity_key_here
   TAVILY_API_KEY=your_tavily_key_here
   ```

## Usage

### Basic Usage
Analyze a dataset with default settings:
```bash
python main.py path/to/your/data.csv
```

### Advanced Usage

Specify a target variable for predictive modeling:
```bash
python main.py data/sales.csv --target_column revenue
```

Custom output directory:
```bash
python main.py data/metrics.json --output ./my_analysis
```

Skip Word document generation (Markdown only):
```bash
python main.py data/financials.xlsx --no-word
```

Verbose mode for debugging:
```bash
python main.py data/customers.csv --verbose
```

## Supported File Formats
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)
- Parquet (`.parquet`)

## Analysis Pipeline

The system follows this automated pipeline:

### 1. Data Loading
- Auto-detects file format
- Loads data into pandas DataFrame
- Generates data quality report
- Saves original data

### 2. Data Cleaning
- Handles missing values (mean/median/mode/drop)
- Detects and handles outliers (IQR/Z-score)
- Removes duplicates
- Saves cleaned data

### 3. Hypothesis Generation
- **Correlation-based**: Identifies relationships between variables
- **Distribution-based**: Detects non-normal patterns
- **Categorical**: Analyzes category distributions
- **Outlier-based**: Flags anomalous data points
- **Trend-based**: Detects temporal trends (if date column exists)

### 4. Statistical Testing
- **Correlation Tests**: Pearson, Spearman, Kendall
- **Normality Tests**: Shapiro-Wilk
- **Outlier Detection**: IQR method, Z-score
- **Group Differences**: ANOVA, Kruskal-Wallis
- **Categorical Association**: Chi-square test

### 5. Predictive Modeling
- **Linear Regression**: With optional Ridge/Lasso regularization
- **Random Forest**: Handles both classification and regression
- **Gradient Boosting**: Advanced ensemble method
- Automatic model comparison and best model selection
- Feature importance analysis

### 6. Insight Extraction
- Generates minimum 50 insights (more if data supports)
- Each insight includes:
  - **What**: Clear statement of finding
  - **Why**: Explanation of underlying reason
  - **How**: Practical application and usage
  - **Recommendation**: Actionable next steps
  - **Data Support**: Statistics and evidence

### 7. Visualization
- Correlation heatmaps
- Distribution plots (histogram + box plot)
- Scatter plots with optional hue grouping
- Line plots for time series
- Bar plots for categorical data
- Feature importance charts
- Residual plots for models
- Interactive plots (Plotly)
- Comprehensive dashboard

### 8. Report Generation
- **Markdown Report**: Comprehensive with all findings
- **Word Document**: Formatted with Word MCP server
- Includes:
  - Executive summary
  - Dataset overview
  - All hypotheses with explanations
  - Statistical test results
  - Model performance metrics
  - All insights with "why" and "how"
  - Embedded visualizations
  - Methodology documentation

### 9. Error Handling & Self-Healing
- Monitors all operations
- Detects errors automatically
- Analyzes root causes
- Implements recovery strategies:
  - API fallback (switch to backup APIs)
  - Data reduction (chunking/sampling)
  - Alternative approaches
  - Parameter adjustment
- Logs all errors and recovery attempts
- Provides comprehensive error report

## Output Structure

For each dataset analysis, the system creates an isolated directory:

```
output/analyses/{dataset_name}/
├── data/
│   ├── original.csv          # Original dataset
│   └── cleaned.csv          # Cleaned dataset
├── code/
│   └── analysis_code.py     # Generated analysis code
├── visualizations/
│   ├── correlation_heatmap.png
│   ├── distribution_*.png
│   ├── scatter_*.png
│   ├── feature_importance.png
│   └── dashboard.png
├── insights/
│   ├── hypotheses.json
│   ├── statistical_tests.json
│   ├── models.json
│   └── insights.json
├── logs/
│   ├── execution_log.json
│   └── error_log.json
├── {dataset_name}_report.md    # Markdown report
└── {dataset_name}_report.docx  # Word document
```

## MCP Servers

### Pandas MCP Server
Provides tools for data manipulation:
- `load_data`: Load datasets from various formats
- `clean_data`: Handle missing values and outliers
- `transform_data`: Create new columns and apply functions
- `filter_data`: Filter data based on conditions
- `group_aggregate`: Group and aggregate data
- `merge_data`: Join multiple datasets
- `get_statistics`: Get descriptive statistics
- `correlation_analysis`: Calculate correlations
- `detect_outliers`: Identify outliers

### Jupyter MCP Server
Provides tools for code execution:
- `execute_code`: Execute Python code with output
- `create_notebook`: Generate Jupyter notebooks
- `run_notebook`: Execute notebook cells
- `convert_to_html`: Convert notebook to HTML
- `convert_to_pdf`: Convert notebook to PDF
- `generate_analysis_notebook`: Create complete analysis notebooks

## Configuration

Edit `config.py` to customize:

```python
# LLM Models
MISTRAL_API_KEY = "your_key"
PERPLEXITY_API_KEY = "your_key"

# Analysis Settings
SIGNIFICANCE_LEVEL = 0.05        # Alpha for statistical tests
MIN_CORRELATION_THRESHOLD = 0.3   # Minimum correlation to report
MAX_HYPOTHESES = 100             # Maximum hypotheses

# Data Processing
MISSING_VALUE_THRESHOLD = 0.5      # Drop columns with >50% missing
TRAIN_TEST_SPLIT = 0.8             # 80% training, 20% testing

# Output
MIN_INSIGHTS = 50                 # Minimum insights to generate
REPORT_FORMATS = ['markdown', 'word']
```

## Example Workflow

```python
# 1. Run autonomous analysis
python main.py data/sales.csv --target_column revenue

# 2. System automatically:
#    - Loads and cleans data
#    - Generates hypotheses
#    - Runs statistical tests
#    - Builds predictive models
#    - Extracts 50+ insights
#    - Creates visualizations
#    - Generates reports

# 3. View results
#    - Read {dataset_name}_report.md for comprehensive findings
#    - Open {dataset_name}_report.docx for formatted document
#    - Check visualizations/ directory for charts

# 4. Review insights:
#    - Each insight has "what", "why", "how", and "recommendation"
#    - All insights are backed by statistical evidence
#    - Clear, non-technical language throughout
```

## Error Handling

The system includes robust error handling:

1. **Detection**: All operations monitored for errors
2. **Analysis**: Root cause identification
3. **Recovery**: Automatic retry with alternatives:
   - Switch to fallback APIs
   - Reduce data size
   - Try alternative methods
   - Adjust parameters
4. **Logging**: Complete error and recovery log
5. **Reporting**: Error summary in final report

## Troubleshooting

### API Key Issues
- Verify API keys in `.env` file
- Check key permissions and rate limits
- Ensure network connectivity

### Memory Issues
- Reduce `MAX_HYPOTHESES` in config
- Use larger datasets on machines with more RAM
- System automatically chunks large operations

### Visualization Errors
- System uses non-interactive backend (Agg)
- Check matplotlib backend settings
- Ensure output directory is writable

### Missing Dependencies
```bash
pip install -r requirements.txt
```

## Future Enhancements

- [ ] Real-time logging dashboard (Streamlit)
- [ ] Post-analysis chatbot with RAG (ChromaDB)
- [ ] Interactive notebook exploration
- [ ] Cloud deployment options
- [ ] Multi-language support
- [ ] Custom model training
- [ ] API endpoint for programmatic access

## Dependencies

See `requirements.txt` for complete list:
- smolagents (agent framework)
- pandas, numpy (data processing)
- scikit-learn (machine learning)
- scipy (statistical testing)
- matplotlib, seaborn, plotly (visualization)
- python-docx (Word documents)
- chromadb (RAG for chatbot)

## License

MIT License

## Support

For issues or feature requests, create an issue on GitHub.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Built for organizations that need to transform raw data into actionable insights without specialized data science teams.**

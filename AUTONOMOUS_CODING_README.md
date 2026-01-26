# Autonomous Code Execution for Data Science System

## Overview

The Autonomous Coding System empowers your agent to write, save, and execute its own Python code with full terminal access. The agent is no longer limited to pre-coded logic - it can generate custom solutions for each analysis task.

## Key Features

### 🤖 Autonomous Code Generation
- Agent generates Python code using LLMs
- No pre-coded logic for analysis tasks
- Custom solutions for each dataset
- Smart code generation with best practices

### 💾 Code Persistence
- All generated code saved as `.py` files
- Optional Jupyter notebook generation
- Complete code manifests and logs
- Timestamped and organized code storage

### 🔧 Environment Control
- Isolated Python environment for execution
- Support for venv, conda, and UV virtualenv
- Automatic dependency installation
- Full terminal command access

### 🚀 Execution Power
- Execute any Python code directly
- Run terminal commands with shell access
- Install packages on demand
- Handle errors and timeouts gracefully

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Autonomous Coding System                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐    ┌──────────────────────────────────┐│
│  │  LLM Council    │───▶│   AutonomousCoderAgent           ││
│  │  (Guidance)     │    │   - Generates code using LLMs    ││
│  └─────────────────┘    │   - Orchestrates analysis tasks  ││
│                         └──────────────────┬───────────────┘│
│                                            │                 │
│                         ┌──────────────────▼───────────────┐│
│                         │   CodeExecutionAgent              ││
│                         │   - Saves code to .py files       ││
│                         │   - Manages Python environments   ││
│                         │   - Executes code with subprocess││
│                         │   - Full terminal access          ││
│                         └──────────────────┬───────────────┘│
│                                            │                 │
│                         ┌──────────────────▼───────────────┐│
│                         │   Output Storage                  ││
│                         │   - generated_code/*.py           ││
│                         │   - notebooks/*.ipynb             ││
│                         │   - visualizations/*.png          ││
│                         │   - logs/*.json                   ││
│                         └───────────────────────────────────┘│
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### 1. Basic Autonomous Analysis

```bash
python main_autonomous.py your_dataset.csv
```

This will:
1. Create an isolated Python environment
2. Generate exploratory analysis code
3. Save code as `.py` file
4. Execute the generated code
5. Create a Jupyter notebook
6. Save all results and visualizations

### 2. With Target Column (Model Building)

```bash
python main_autonomous.py your_dataset.csv --target target_column
```

The agent will generate code for:
- Exploratory data analysis
- Feature engineering
- Model training and evaluation
- Feature importance visualization

### 3. Specify Task Type

```bash
# Classification
python main_autonomous.py your_dataset.csv --target churn --task-type classification

# Regression
python main_autonomous.py your_dataset.csv --target price --task-type regression
```

### 4. With LLM Council Guidance

```bash
python main_autonomous_with_council.py your_dataset.csv --target column
```

Combines autonomous coding with LLM Council consensus for:
- Better analysis strategies
- Smarter model selection
- More comprehensive insights

## Output Structure

```
output/analyses/{dataset_name}/{timestamp}/
├── generated_code/
│   ├── exploratory_analysis_20260126_105941.py     # EDA code
│   ├── feature_engineering_20260126_105941.py     # Feature engineering
│   ├── model_building_20260126_105941.py          # Model training
│   └── code_manifest.json                         # Code inventory
├── notebooks/
│   └── analysis_20260126_105941.ipynb             # Complete notebook
├── visualizations/
│   ├── distributions/
│   ├── correlations/
│   └── feature_importance.png
├── data/
│   └── original.csv
├── logs/
│   └── autonomous_execution_log.json
├── envs/
│   └── analysis_env/                              # Python environment
└── autonomous_analysis_results.json              # Complete results
```

## Generated Code Example

The agent generates complete, executable Python code like this:

```python
"""
Generated by Autonomous Code Execution Agent
Filename: exploratory_analysis_20260126_105941.py
Generated: 2026-01-26T10:59:41.123456
Description: Exploratory data analysis with visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Load dataset
df = pd.read_csv('data/original.csv')

print("="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Dataset info
print(f"\nDataset Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")

# Summary statistics
print(f"\nSummary Statistics:\n{df.describe()}")

# Missing values
missing = df.isnull().sum()
print(f"\nMissing Values:\n{missing[missing > 0]}")

# Visualizations
output_dir = 'visualizations'
os.makedirs(output_dir, exist_ok=True)

# Distribution plots
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols[:5]:  # First 5 numeric columns
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'{output_dir}/distribution_{col}.png')
    plt.close()

# Correlation heatmap
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()

# Save summary
summary = {
    'shape': df.shape,
    'columns': list(df.columns),
    'numeric_columns': list(numeric_cols),
    'missing_values': missing.to_dict()
}

with open('data/eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✓ Exploratory analysis complete")
print(f"✓ Visualizations saved to {output_dir}")
print(f"✓ Summary saved to data/eda_summary.json")
```

## Key Components

### CodeExecutionAgent

Handles the low-level code execution:

```python
from agents import CodeExecutionAgent

# Initialize
executor = CodeExecutionAgent(output_dir="output")

# Setup environment
env_path = executor.setup_environment(env_name="my_env")

# Save code
file_path = executor.generate_code_file(
    code="print('Hello, world!')",
    filename="my_script",
    description="Simple hello world script"
)

# Execute code
result = executor.execute_code_file(file_path)

# Run terminal commands
result = executor.run_terminal_command("ls -la")

# Install packages
result = executor.install_package("scikit-learn")
```

### AutonomousCoderAgent

Orchestrates autonomous analysis:

```python
from agents import AutonomousCoderAgent

# Initialize
coder = AutonomousCoderAgent(output_dir="output")

# Exploratory analysis
coder.analyze_dataset_exploratory("data.csv")

# Feature engineering
coder.perform_feature_engineering("data.csv", target_column="target")

# Model building
coder.build_models_autonomously("data.csv", target_column="target")

# Custom analysis
coder.generate_custom_analysis("Create a time series forecast", "data.csv")

# Generate notebook
coder.create_notebook_from_analysis(code_files, "my_analysis")
```

### AutonomousAnalysisPipeline

Complete pipeline orchestration:

```python
from workflow.autonomous_pipeline import AutonomousAnalysisPipeline

# Create pipeline
pipeline = AutonomousAnalysisPipeline("data.csv")

# Run full pipeline
results = pipeline.run_full_autonomous_pipeline(
    target_column="target",
    task_type="classification",
    create_notebook=True
)
```

## Environment Setup

The system supports three environment types:

### 1. UV Virtualenv (Recommended - Fastest)
```python
executor = CodeExecutionAgent(output_dir="output", environment_type="venv-uv")
```

### 2. Standard Virtualenv
```python
executor = CodeExecutionAgent(output_dir="output", environment_type="venv")
```

### 3. Conda
```python
executor = CodeExecutionAgent(output_dir="output", environment_type="conda")
```

## Advanced Usage

### Custom Tasks

```python
from agents import AutonomousCoderAgent

coder = AutonomousCoderAgent(output_dir="output")

# Any custom task description
result = coder.generate_custom_analysis(
    custom_task="""
    Perform advanced analysis:
    1. Detect anomalies using Isolation Forest
    2. Cluster data using K-means
    3. Generate SHAP values for model interpretability
    """,
    dataset_path="data.csv"
)
```

### Direct Code Execution

```python
from agents import CodeExecutionAgent

executor = CodeExecutionAgent(output_dir="output")

# Execute code string directly
result = executor.execute_code("""
import pandas as pd
df = pd.read_csv('data.csv')
print(df.describe())
""")
```

### Terminal Commands

```python
# Full shell access
result = executor.run_terminal_command("pip list | grep pandas")
result = executor.run_terminal_command("ls -lh *.csv")
result = executor.run_terminal_command("python -c 'import numpy as np; print(np.pi)'")
```

## Comparison: Old vs New

| Feature | Old System | New Autonomous System |
|---------|-----------|----------------------|
| Code Source | Pre-coded logic | Agent-generated code |
| Code Storage | Not saved | Saved as .py files |
| Environment | System Python | Isolated environment |
| Customization | Limited | Unlimited |
| Transparency | Hidden | Full code visibility |
| Reproducibility | Hard | Easy (code files) |
| Debugging | Difficult | Direct code access |

## Security Considerations

- All code executed in isolated environments
- Timeouts prevent infinite loops
- Error handling prevents crashes
- No system-level access by default
- Controlled terminal command execution

## Troubleshooting

### Environment Setup Fails
```bash
# Ensure UV is installed
pip install uv

# Or use standard venv
python main_autonomous.py data.csv --use-venv
```

### Code Execution Times Out
```python
# Increase timeout in code_executor.py
executor.execute_code_file(file_path, timeout=600)  # 10 minutes
```

### Package Installation Fails
```bash
# Check environment
output/analyses/{dataset}/{timestamp}/envs/analysis_env/bin/pip list

# Manually install
output/analyses/{dataset}/{timestamp}/envs/analysis_env/bin/pip install package_name
```

## Future Enhancements

- [ ] Integration with MCP Code Executor (bazinga012/mcp_code_executor)
- [ ] Code review and optimization by LLM Council
- [ ] Automatic unit test generation
- [ ] Docker containerization
- [ ] Parallel code execution
- [ ] Code versioning with Git

## License

MIT License - See LICENSE file for details

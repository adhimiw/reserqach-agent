"""
Jupyter MCP Server for Code Execution and Notebook Generation
Provides tools for executing Python code, generating Jupyter notebooks, and running analyses
"""

from mcp.server import Server
from mcp.types import Tool, TextContent
import json
import subprocess
import tempfile
import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from typing import Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JupyterMCP")

# Initialize MCP server
jupyter_server = Server("jupyter-execution-server")

# Temporary directory for notebooks
NOTEBOOK_DIR = tempfile.gettempdir()


@jupyter_server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available Jupyter tools"""
    return [
        Tool(
            name="execute_code",
            description="Execute Python code and return output, including matplotlib plots",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "save_output": {
                        "type": "boolean",
                        "description": "Whether to save outputs to files (default: false)"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save outputs (if save_output=true)"
                    }
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="create_notebook",
            description="Create a Jupyter notebook from cells",
            inputSchema={
                "type": "object",
                "properties": {
                    "cells": {
                        "type": "string",
                        "description": "JSON array of cells: [{'type': 'code'|'markdown', 'content': '...'}]"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the .ipynb file"
                    },
                    "metadata": {
                        "type": "string",
                        "description": "JSON metadata for the notebook (optional)"
                    }
                },
                "required": ["cells", "output_path"]
            }
        ),
        Tool(
            name="run_notebook",
            description="Execute a Jupyter notebook and capture outputs",
            inputSchema={
                "type": "object",
                "properties": {
                    "notebook_path": {
                        "type": "string",
                        "description": "Path to the .ipynb file"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the executed notebook"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Execution timeout in seconds (default: 600)"
                    }
                },
                "required": ["notebook_path", "output_path"]
            }
        ),
        Tool(
            name="convert_to_html",
            description="Convert a notebook to HTML format",
            inputSchema={
                "type": "object",
                "properties": {
                    "notebook_path": {
                        "type": "string",
                        "description": "Path to the .ipynb file"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the .html file"
                    }
                },
                "required": ["notebook_path", "output_path"]
            }
        ),
        Tool(
            name="convert_to_pdf",
            description="Convert a notebook to PDF format",
            inputSchema={
                "type": "object",
                "properties": {
                    "notebook_path": {
                        "type": "string",
                        "description": "Path to the .ipynb file"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the .pdf file"
                    }
                },
                "required": ["notebook_path", "output_path"]
            }
        ),
        Tool(
            name="generate_analysis_notebook",
            description="Generate a complete analysis notebook with data loading, analysis, and visualizations",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["exploratory", "statistical", "predictive", "time_series"],
                        "description": "Type of analysis to perform"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the notebook"
                    },
                    "custom_code": {
                        "type": "string",
                        "description": "Additional custom code to include (optional)"
                    }
                },
                "required": ["dataset_path", "analysis_type", "output_path"]
            }
        ),
        Tool(
            name="install_package",
            description="Install a Python package",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {
                        "type": "string",
                        "description": "Package name (e.g., 'scikit-learn', version optional: 'scikit-learn==1.3.0')"
                    },
                    "upgrade": {
                        "type": "boolean",
                        "description": "Upgrade if already installed (default: false)"
                    }
                },
                "required": ["package"]
            }
        ),
        Tool(
            name="check_package_version",
            description="Check version of an installed package",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {
                        "type": "string",
                        "description": "Package name"
                    }
                },
                "required": ["package"]
            }
        )
    ]


@jupyter_server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "execute_code":
            return await handle_execute_code(arguments)
        elif name == "create_notebook":
            return await handle_create_notebook(arguments)
        elif name == "run_notebook":
            return await handle_run_notebook(arguments)
        elif name == "convert_to_html":
            return await handle_convert_to_html(arguments)
        elif name == "convert_to_pdf":
            return await handle_convert_to_pdf(arguments)
        elif name == "generate_analysis_notebook":
            return await handle_generate_analysis_notebook(arguments)
        elif name == "install_package":
            return await handle_install_package(arguments)
        elif name == "check_package_version":
            return await handle_check_package_version(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        logger.error(f"Error in {name}: {str(e)}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_execute_code(arguments: dict) -> list[TextContent]:
    """Execute Python code"""
    code = arguments["code"]
    save_output = arguments.get("save_output", False)
    output_dir = arguments.get("output_dir", NOTEBOOK_DIR)
    
    # Add matplotlib inline to save plots
    import_matplotlib = """
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
"""
    
    save_plots_code = ""
    if save_output:
        save_plots_code = f"""
import os
os.makedirs('{output_dir}', exist_ok=True)
"""
        # Wrap plotting code to save figures
        code = code.replace("plt.show()", f"plt.savefig('{output_dir}/plot.png'); plt.close()")
    
    full_code = import_matplotlib + save_plots_code + code
    
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_file = f.name
    
    try:
        # Execute the code
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n\nSTDERR:\n" + result.stderr
        
        # List generated files if any
        generated_files = []
        if save_output and os.path.exists(output_dir):
            generated_files = os.listdir(output_dir)
        
        response = f"Execution completed!\n\nOutput:\n{output}"
        if generated_files:
            response += f"\n\nGenerated files:\n{json.dumps(generated_files, indent=2)}"
        
        return [TextContent(type="text", text=response)]
        
    finally:
        os.unlink(temp_file)


async def handle_create_notebook(arguments: dict) -> list[TextContent]:
    """Create a Jupyter notebook"""
    cells = json.loads(arguments["cells"])
    output_path = arguments["output_path"]
    metadata = json.loads(arguments.get("metadata", "{}"))
    
    # Create notebook
    nb = new_notebook()
    
    # Add cells
    for cell_data in cells:
        if cell_data["type"] == "code":
            nb.cells.append(new_code_cell(source=cell_data["content"]))
        elif cell_data["type"] == "markdown":
            nb.cells.append(new_markdown_cell(source=cell_data["content"]))
    
    # Add metadata
    if metadata:
        nb.metadata.update(metadata)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save notebook
    with open(output_path, 'w') as f:
        nbformat.write(nb, f)
    
    return [TextContent(
        type="text",
        text=f"Notebook created successfully at: {output_path}\n"
             f"Total cells: {len(nb.cells)}"
    )]


async def handle_run_notebook(arguments: dict) -> list[TextContent]:
    """Execute a Jupyter notebook"""
    notebook_path = arguments["notebook_path"]
    output_path = arguments["output_path"]
    timeout = arguments.get("timeout", 600)
    
    # Use nbconvert to execute the notebook
    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--allow-errors',
        f'--ExecutePreprocessor.timeout={timeout}',
        '--output', output_path,
        notebook_path
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        return [TextContent(
            type="text",
            text=f"Notebook executed successfully!\n"
                 f"Output saved to: {output_path}"
        )]
    else:
        return [TextContent(
            type="text",
            text=f"Error executing notebook:\n{result.stderr}"
        )]


async def handle_convert_to_html(arguments: dict) -> list[TextContent]:
    """Convert notebook to HTML"""
    notebook_path = arguments["notebook_path"]
    output_path = arguments["output_path"]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use nbconvert
    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'html',
        '--output', output_path,
        notebook_path
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        return [TextContent(
            type="text",
            text=f"Notebook converted to HTML successfully!\n"
                 f"Output saved to: {output_path}"
        )]
    else:
        return [TextContent(
            type="text",
            text=f"Error converting notebook:\n{result.stderr}"
        )]


async def handle_convert_to_pdf(arguments: dict) -> list[TextContent]:
    """Convert notebook to PDF"""
    notebook_path = arguments["notebook_path"]
    output_path = arguments["output_path"]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use nbconvert (requires pandoc and TeX)
    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'pdf',
        '--output', output_path,
        notebook_path
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        return [TextContent(
            type="text",
            text=f"Notebook converted to PDF successfully!\n"
                 f"Output saved to: {output_path}"
        )]
    else:
        return [TextContent(
            type="text",
            text=f"Error converting to PDF (requires pandoc and TeX):\n{result.stderr}"
        )]


async def handle_generate_analysis_notebook(arguments: dict) -> list[TextContent]:
    """Generate a complete analysis notebook"""
    dataset_path = arguments["dataset_path"]
    analysis_type = arguments["analysis_type"]
    output_path = arguments["output_path"]
    custom_code = arguments.get("custom_code", "")
    
    # Base setup cells
    setup_cells = [
        {
            "type": "markdown",
            "content": f"# Data Analysis Notebook\n\nAnalysis Type: {analysis_type}\nDataset: {dataset_path}"
        },
        {
            "type": "code",
            "content": """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Libraries imported successfully!")
"""
        },
        {
            "type": "code",
            "content": f"""
# Load dataset
df = pd.read_csv('{dataset_path}')
print(f"Dataset shape: {{df.shape}}")
print(f"\\nDataset info:")
print(df.info())
"""
        }
    ]
    
    # Analysis-specific cells
    if analysis_type == "exploratory":
        analysis_cells = [
            {
                "type": "markdown",
                "content": "## Exploratory Data Analysis"
            },
            {
                "type": "code",
                "content": """
# Display first few rows
print("First 5 rows:")
display(df.head())

# Display last few rows
print("\\nLast 5 rows:")
display(df.tail())
"""
            },
            {
                "type": "code",
                "content": """
# Statistical summary
print("Statistical Summary:")
print(df.describe())
"""
            },
            {
                "type": "code",
                "content": """
# Missing values analysis
print("Missing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
print(missing_df[missing_df['Count'] > 0])
"""
            },
            {
                "type": "code",
                "content": """
# Data types
print("Data Types:")
print(df.dtypes)
"""
            },
            {
                "type": "code",
                "content": """
# Correlation heatmap (for numerical columns)
numeric_df = df.select_dtypes(include=[np.number])
if len(numeric_df.columns) > 1:
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("Not enough numeric columns for correlation analysis")
"""
            },
            {
                "type": "code",
                "content": """
# Distribution of numerical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
n_cols = min(4, len(numeric_cols))
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

if len(numeric_cols) > 0:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
"""
            }
        ]
    
    elif analysis_type == "statistical":
        analysis_cells = [
            {
                "type": "markdown",
                "content": "## Statistical Analysis"
            },
            {
                "type": "code",
                "content": """
# Detailed statistics
print("Detailed Statistical Analysis:")
print("\\nNumerical Columns:")
print(df.describe().T)

print("\\nCategorical Columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\\n{col}:")
    print(df[col].value_counts())
"""
            },
            {
                "type": "code",
                "content": """
# Correlation analysis
numeric_df = df.select_dtypes(include=[np.number])
if len(numeric_df.columns) > 1:
    corr_matrix = numeric_df.corr()
    print("Correlation Matrix:")
    print(corr_matrix.round(3))
    
    # Strong correlations
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= 0.5:
                strong_corrs.append({
                    'col1': corr_matrix.columns[i],
                    'col2': corr_matrix.columns[j],
                    'correlation': round(corr_val, 3)
                })
    
    if strong_corrs:
        print("\\nStrong Correlations (|r| >= 0.5):")
        for corr in sorted(strong_corrs, key=lambda x: abs(x['correlation']), reverse=True):
            print(f"  {corr['col1']} - {corr['col2']}: {corr['correlation']}")
"""
            },
            {
                "type": "code",
                "content": """
# Outlier detection using IQR method
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"\\n{col}:")
    print(f"  Outliers detected: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    if len(outliers) > 0:
        print(f"  Range: [{outliers[col].min():.2f}, {outliers[col].max():.2f}]")
"""
            }
        ]
    
    elif analysis_type == "predictive":
        analysis_cells = [
            {
                "type": "markdown",
                "content": "## Predictive Modeling"
            },
            {
                "type": "code",
                "content": """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report

# Identify target variable
print("Available columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col} ({df[col].dtype})")

# Note: In practice, specify the target variable
# target_column = "your_target_column"
print("\\nPlease specify the target variable for predictive modeling")
"""
            },
            {
                "type": "code",
                "content": """
# Feature engineering example
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("Categorical variables encoded")
print(f"Encoded shape: {df_encoded.shape}")
"""
            },
            {
                "type": "code",
                "content": """
# Feature importance using Random Forest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

X = df_encoded.drop(columns=[df_encoded.columns[-1]])  # Assuming last column is target
y = df_encoded[df_encoded.columns[-1]]

# Determine task type
if len(y.unique()) <= 20:  # Classification
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Task: Classification")
else:  # Regression
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("Task: Regression")

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nFeature Importance:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
"""
            }
        ]
    
    elif analysis_type == "time_series":
        analysis_cells = [
            {
                "type": "markdown",
                "content": "## Time Series Analysis"
            },
            {
                "type": "code",
                "content": """
# Note: This analysis requires a datetime column
print("Time series analysis requires:")
print("  1. A datetime column for the index")
print("  2. A target variable to analyze over time")
print("\\nPlease ensure your data has these requirements")
"""
            },
            {
                "type": "code",
                "content": """
# Example time series setup
# Uncomment and modify as needed:
# date_column = "your_date_column"
# value_column = "your_value_column"
# 
# df[date_column] = pd.to_datetime(df[date_column])
# df = df.set_index(date_column).sort_index()
# 
# plt.figure(figsize=(14, 6))
# plt.plot(df.index, df[value_column])
# plt.title(f'Time Series: {value_column}')
# plt.xlabel('Date')
# plt.ylabel(value_column)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
print("Configure time series columns in the code above")
"""
            },
            {
                "type": "code",
                "content": """
# Rolling statistics (example)
# window = 7  # 7-day moving average
# 
# df['rolling_mean'] = df[value_column].rolling(window=window).mean()
# df['rolling_std'] = df[value_column].rolling(window=window).std()
# 
# plt.figure(figsize=(14, 6))
# plt.plot(df.index, df[value_column], label='Original', alpha=0.7)
# plt.plot(df.index, df['rolling_mean'], label=f'{window}-day Moving Average', linewidth=2)
# plt.fill_between(df.index, 
#                  df['rolling_mean'] - df['rolling_std'],
#                  df['rolling_mean'] + df['rolling_std'],
#                  alpha=0.2, label=f'{window}-day Std Dev')
# plt.title(f'{value_column} with Rolling Statistics')
# plt.legend()
# plt.tight_layout()
# plt.show()
print("Configure rolling statistics in the code above")
"""
            }
        ]
    
    # Add custom code if provided
    custom_cells = []
    if custom_code:
        custom_cells = [
            {
                "type": "markdown",
                "content": "## Custom Analysis"
            },
            {
                "type": "code",
                "content": custom_code
            }
        ]
    
    # Combine all cells
    all_cells = setup_cells + analysis_cells + custom_cells
    
    # Create notebook
    return await handle_create_notebook({
        "cells": json.dumps(all_cells),
        "output_path": output_path,
        "metadata": json.dumps({
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        })
    })


async def handle_install_package(arguments: dict) -> list[TextContent]:
    """Install a Python package"""
    package = arguments["package"]
    upgrade = arguments.get("upgrade", False)
    
    cmd = ["pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Try to get version
        version_check = subprocess.run(
            ["pip", "show", package.split('==')[0]],
            capture_output=True,
            text=True
        )
        
        version = "unknown"
        for line in version_check.stdout.split('\n'):
            if line.startswith('Version:'):
                version = line.split(':')[1].strip()
                break
        
        return [TextContent(
            type="text",
            text=f"Package '{package}' installed successfully!\nVersion: {version}"
        )]
    else:
        return [TextContent(
            type="text",
            text=f"Error installing package:\n{result.stderr}"
        )]


async def handle_check_package_version(arguments: dict) -> list[TextContent]:
    """Check version of an installed package"""
    package = arguments["package"]
    
    result = subprocess.run(
        ["pip", "show", package],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        info = {}
        for line in result.stdout.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()
        
        return [TextContent(
            type="text",
            text=json.dumps(info, indent=2)
        )]
    else:
        return [TextContent(
            type="text",
            text=f"Package '{package}' not found or not installed"
        )]


async def main():
    """Main entry point for the Jupyter MCP server"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await jupyter_server.run(
            read_stream,
            write_stream,
            jupyter_server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

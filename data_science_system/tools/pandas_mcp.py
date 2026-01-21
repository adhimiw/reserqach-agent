"""
Pandas MCP Server for Data Manipulation and Analysis
Provides tools for data loading, cleaning, transformation, and analysis using pandas
"""

from mcp.server import Server
from mcp.types import Tool, TextContent
import pandas as pd
import numpy as np
import io
import json
from typing import Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PandasMCP")

# Initialize MCP server
pandas_server = Server("pandas-data-server")

# Global storage for loaded datasets
datasets = {}


@pandas_server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available Pandas tools"""
    return [
        Tool(
            name="load_data",
            description="Load data from CSV, Excel, JSON, or Parquet file. Returns dataset info.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the data file"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["csv", "excel", "json", "parquet"],
                        "description": "File format (auto-detected if not specified)"
                    },
                    "dataset_name": {
                        "type": "string",
                        "description": "Name to assign to this dataset (default: filename)"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="get_data_info",
            description="Get comprehensive information about a loaded dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        Tool(
            name="clean_data",
            description="Clean dataset: handle missing values, remove duplicates, fix data types",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "missing_strategy": {
                        "type": "string",
                        "enum": ["drop", "mean", "median", "mode", "forward_fill", "backward_fill"],
                        "description": "Strategy for handling missing values"
                    },
                    "drop_duplicates": {
                        "type": "boolean",
                        "description": "Whether to remove duplicate rows (default: true)"
                    },
                    "missing_threshold": {
                        "type": "number",
                        "description": "Drop columns with missing values above this threshold (0-1, default: 0.5)"
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        Tool(
            name="transform_data",
            description="Transform data: create new columns, apply functions, normalize",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "transformations": {
                        "type": "string",
                        "description": "JSON string describing transformations (e.g., {'new_col': 'col1 + col2'})"
                    }
                },
                "required": ["dataset_name", "transformations"]
            }
        ),
        Tool(
            name="filter_data",
            description="Filter dataset based on conditions",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "condition": {
                        "type": "string",
                        "description": "Filter condition (e.g., 'col1 > 100 and col2 == \"category\"')"
                    },
                    "output_name": {
                        "type": "string",
                        "description": "Name for the filtered dataset (default: {name}_filtered)"
                    }
                },
                "required": ["dataset_name", "condition"]
            }
        ),
        Tool(
            name="group_aggregate",
            description="Group data by columns and perform aggregations",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "group_by": {
                        "type": "string",
                        "description": "Column(s) to group by (comma-separated)"
                    },
                    "aggregations": {
                        "type": "string",
                        "description": "Aggregations as JSON: {'col1': ['mean', 'sum'], 'col2': 'count'}"
                    }
                },
                "required": ["dataset_name", "group_by", "aggregations"]
            }
        ),
        Tool(
            name="merge_data",
            description="Merge two datasets on common columns",
            inputSchema={
                "type": "object",
                "properties": {
                    "left_dataset": {
                        "type": "string",
                        "description": "Name of left dataset"
                    },
                    "right_dataset": {
                        "type": "string",
                        "description": "Name of right dataset"
                    },
                    "on": {
                        "type": "string",
                        "description": "Column(s) to merge on (comma-separated)"
                    },
                    "how": {
                        "type": "string",
                        "enum": ["inner", "outer", "left", "right"],
                        "description": "Type of merge (default: inner)"
                    },
                    "output_name": {
                        "type": "string",
                        "description": "Name for merged dataset"
                    }
                },
                "required": ["left_dataset", "right_dataset", "on"]
            }
        ),
        Tool(
            name="get_statistics",
            description="Get statistical summary of dataset columns",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "columns": {
                        "type": "string",
                        "description": "Specific columns to analyze (default: all)"
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        Tool(
            name="correlation_analysis",
            description="Calculate correlation matrix for numerical columns",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["pearson", "spearman", "kendall"],
                        "description": "Correlation method (default: pearson)"
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        Tool(
            name="export_data",
            description="Export dataset to file",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["csv", "excel", "json", "parquet"],
                        "description": "Output format"
                    }
                },
                "required": ["dataset_name", "output_path"]
            }
        ),
        Tool(
            name="get_sample",
            description="Get sample rows from dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "n": {
                        "type": "number",
                        "description": "Number of rows to return (default: 10)"
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        Tool(
            name="detect_outliers",
            description="Detect outliers using IQR or Z-score method",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "column": {
                        "type": "string",
                        "description": "Column to analyze"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["iqr", "zscore"],
                        "description": "Detection method (default: iqr)"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Threshold (default: 1.5 for IQR, 3 for Z-score)"
                    }
                },
                "required": ["dataset_name", "column"]
            }
        )
    ]


@pandas_server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "load_data":
            return await handle_load_data(arguments)
        elif name == "get_data_info":
            return await handle_get_data_info(arguments)
        elif name == "clean_data":
            return await handle_clean_data(arguments)
        elif name == "transform_data":
            return await handle_transform_data(arguments)
        elif name == "filter_data":
            return await handle_filter_data(arguments)
        elif name == "group_aggregate":
            return await handle_group_aggregate(arguments)
        elif name == "merge_data":
            return await handle_merge_data(arguments)
        elif name == "get_statistics":
            return await handle_get_statistics(arguments)
        elif name == "correlation_analysis":
            return await handle_correlation_analysis(arguments)
        elif name == "export_data":
            return await handle_export_data(arguments)
        elif name == "get_sample":
            return await handle_get_sample(arguments)
        elif name == "detect_outliers":
            return await handle_detect_outliers(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        logger.error(f"Error in {name}: {str(e)}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_load_data(arguments: dict) -> list[TextContent]:
    """Load data from file"""
    file_path = arguments["file_path"]
    format_type = arguments.get("format")
    dataset_name = arguments.get("dataset_name")
    
    # Auto-detect format if not specified
    if not format_type:
        if file_path.endswith('.csv'):
            format_type = 'csv'
        elif file_path.endswith(('.xlsx', '.xls')):
            format_type = 'excel'
        elif file_path.endswith('.json'):
            format_type = 'json'
        elif file_path.endswith('.parquet'):
            format_type = 'parquet'
        else:
            format_type = 'csv'  # Default
    
    # Generate dataset name from filename if not provided
    if not dataset_name:
        dataset_name = file_path.split('/')[-1].split('.')[0]
    
    # Load data
    if format_type == 'csv':
        df = pd.read_csv(file_path)
    elif format_type == 'excel':
        df = pd.read_excel(file_path)
    elif format_type == 'json':
        df = pd.read_json(file_path)
    elif format_type == 'parquet':
        df = pd.read_parquet(file_path)
    
    # Store dataset
    datasets[dataset_name] = df
    
    info = {
        "dataset_name": dataset_name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "shape": df.shape
    }
    
    return [TextContent(
        type="text",
        text=f"Data loaded successfully!\n\n{json.dumps(info, indent=2)}"
    )]


async def handle_get_data_info(arguments: dict) -> list[TextContent]:
    """Get comprehensive data info"""
    dataset_name = arguments["dataset_name"]
    
    if dataset_name not in datasets:
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found")]
    
    df = datasets[dataset_name]
    
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "duplicate_rows": df.duplicated().sum(),
        "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
        "datetime_columns": list(df.select_dtypes(include=['datetime64']).columns)
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(info, indent=2)
    )]


async def handle_clean_data(arguments: dict) -> list[TextContent]:
    """Clean dataset"""
    dataset_name = arguments["dataset_name"]
    missing_strategy = arguments.get("missing_strategy", "drop")
    drop_duplicates = arguments.get("drop_duplicates", True)
    missing_threshold = arguments.get("missing_threshold", 0.5)
    
    if dataset_name not in datasets:
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found")]
    
    df = datasets[dataset_name].copy()
    
    # Drop columns with too many missing values
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Handle remaining missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if missing_strategy == "drop":
        df = df.dropna()
    elif missing_strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    elif missing_strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    elif missing_strategy == "mode":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    elif missing_strategy == "forward_fill":
        df = df.fillna(method='ffill')
    elif missing_strategy == "backward_fill":
        df = df.fillna(method='bfill')
    
    # Drop duplicates
    if drop_duplicates:
        df = df.drop_duplicates()
    
    # Update dataset
    datasets[dataset_name] = df
    
    return [TextContent(
        type="text",
        text=f"Data cleaned successfully!\n\n"
             f"Original shape: {datasets[dataset_name].shape}\n"
             f"Cleaned shape: {df.shape}\n"
             f"Rows removed: {datasets[dataset_name].shape[0] - df.shape[0]}\n"
             f"Columns dropped: {cols_to_drop if cols_to_drop else 'None'}"
    )]


async def handle_transform_data(arguments: dict) -> list[TextContent]:
    """Transform data"""
    dataset_name = arguments["dataset_name"]
    transformations = json.loads(arguments["transformations"])
    
    if dataset_name not in datasets:
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found")]
    
    df = datasets[dataset_name].copy()
    
    # Apply transformations
    for new_col, expression in transformations.items():
        df[new_col] = df.eval(expression)
    
    datasets[dataset_name] = df
    
    return [TextContent(
        type="text",
        text=f"Transformations applied successfully!\n\n"
             f"New shape: {df.shape}\n"
             f"New columns: {list(transformations.keys())}"
    )]


async def handle_filter_data(arguments: dict) -> list[TextContent]:
    """Filter data"""
    dataset_name = arguments["dataset_name"]
    condition = arguments["condition"]
    output_name = arguments.get("output_name", f"{dataset_name}_filtered")
    
    if dataset_name not in datasets:
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found")]
    
    df = datasets[dataset_name].copy()
    filtered_df = df.query(condition)
    
    datasets[output_name] = filtered_df
    
    return [TextContent(
        type="text",
        text=f"Data filtered successfully!\n\n"
             f"Original rows: {len(df)}\n"
             f"Filtered rows: {len(filtered_df)}\n"
             f"Dataset saved as: {output_name}"
    )]


async def handle_group_aggregate(arguments: dict) -> list[TextContent]:
    """Group and aggregate data"""
    dataset_name = arguments["dataset_name"]
    group_by = arguments["group_by"].split(',')
    aggregations = json.loads(arguments["aggregations"])
    
    if dataset_name not in datasets:
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found")]
    
    df = datasets[dataset_name]
    
    grouped = df.groupby(group_by).agg(aggregations)
    
    output_name = f"{dataset_name}_grouped"
    datasets[output_name] = grouped.reset_index()
    
    return [TextContent(
        type="text",
        text=f"Grouped aggregation completed!\n\n"
             f"Groups: {len(grouped)}\n"
             f"Output dataset: {output_name}\n\n"
             f"{grouped.to_string()}"
    )]


async def handle_merge_data(arguments: dict) -> list[TextContent]:
    """Merge two datasets"""
    left_dataset = arguments["left_dataset"]
    right_dataset = arguments["right_dataset"]
    on = arguments["on"].split(',')
    how = arguments.get("how", "inner")
    output_name = arguments.get("output_name", f"{left_dataset}_merged")
    
    if left_dataset not in datasets or right_dataset not in datasets:
        return [TextContent(type="text", text="One or both datasets not found")]
    
    left_df = datasets[left_dataset]
    right_df = datasets[right_dataset]
    
    merged = pd.merge(left_df, right_df, on=on, how=how)
    datasets[output_name] = merged
    
    return [TextContent(
        type="text",
        text=f"Datasets merged successfully!\n\n"
             f"Left shape: {left_df.shape}\n"
             f"Right shape: {right_df.shape}\n"
             f"Merged shape: {merged.shape}\n"
             f"Output dataset: {output_name}"
    )]


async def handle_get_statistics(arguments: dict) -> list[TextContent]:
    """Get statistical summary"""
    dataset_name = arguments["dataset_name"]
    columns = arguments.get("columns")
    
    if dataset_name not in datasets:
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found")]
    
    df = datasets[dataset_name]
    
    if columns:
        columns = columns.split(',')
        df = df[columns]
    
    # Only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    stats = {
        "count": numeric_df.count().to_dict(),
        "mean": numeric_df.mean().to_dict(),
        "std": numeric_df.std().to_dict(),
        "min": numeric_df.min().to_dict(),
        "25%": numeric_df.quantile(0.25).to_dict(),
        "50%": numeric_df.quantile(0.50).to_dict(),
        "75%": numeric_df.quantile(0.75).to_dict(),
        "max": numeric_df.max().to_dict()
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(stats, indent=2)
    )]


async def handle_correlation_analysis(arguments: dict) -> list[TextContent]:
    """Calculate correlation matrix"""
    dataset_name = arguments["dataset_name"]
    method = arguments.get("method", "pearson")
    
    if dataset_name not in datasets:
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found")]
    
    df = datasets[dataset_name]
    numeric_df = df.select_dtypes(include=[np.number])
    
    corr_matrix = numeric_df.corr(method=method)
    
    # Find strong correlations
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= 0.5:  # Strong correlation threshold
                strong_corrs.append({
                    "col1": corr_matrix.columns[i],
                    "col2": corr_matrix.columns[j],
                    "correlation": round(corr_val, 3)
                })
    
    result = {
        "correlation_matrix": corr_matrix.round(3).to_dict(),
        "strong_correlations": sorted(strong_corrs, key=lambda x: abs(x['correlation']), reverse=True)
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]


async def handle_export_data(arguments: dict) -> list[TextContent]:
    """Export dataset to file"""
    dataset_name = arguments["dataset_name"]
    output_path = arguments["output_path"]
    format_type = arguments["format"]
    
    if dataset_name not in datasets:
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found")]
    
    df = datasets[dataset_name]
    
    if format_type == 'csv':
        df.to_csv(output_path, index=False)
    elif format_type == 'excel':
        df.to_excel(output_path, index=False)
    elif format_type == 'json':
        df.to_json(output_path, orient='records')
    elif format_type == 'parquet':
        df.to_parquet(output_path)
    
    return [TextContent(
        type="text",
        text=f"Data exported successfully to: {output_path}"
    )]


async def handle_get_sample(arguments: dict) -> list[TextContent]:
    """Get sample rows"""
    dataset_name = arguments["dataset_name"]
    n = arguments.get("n", 10)
    
    if dataset_name not in datasets:
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found")]
    
    df = datasets[dataset_name]
    sample = df.head(n)
    
    return [TextContent(
        type="text",
        text=f"Sample data (first {n} rows):\n\n{sample.to_string()}"
    )]


async def handle_detect_outliers(arguments: dict) -> list[TextContent]:
    """Detect outliers"""
    dataset_name = arguments["dataset_name"]
    column = arguments["column"]
    method = arguments.get("method", "iqr")
    threshold = arguments.get("threshold", 1.5 if method == "iqr" else 3)
    
    if dataset_name not in datasets:
        return [TextContent(type="text", text=f"Dataset '{dataset_name}' not found")]
    
    df = datasets[dataset_name]
    
    if column not in df.columns:
        return [TextContent(type="text", text=f"Column '{column}' not found")]
    
    data = df[column].dropna()
    
    if method == "iqr":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers_mask = (data < lower_bound) | (data > upper_bound)
    else:  # zscore
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers_mask = z_scores > threshold
    
    outlier_indices = data[outliers_mask].index.tolist()
    outlier_values = data[outliers_mask].tolist()
    
    result = {
        "column": column,
        "method": method,
        "threshold": threshold,
        "outlier_count": len(outlier_indices),
        "outlier_percentage": round(len(outlier_indices) / len(data) * 100, 2),
        "outlier_indices": outlier_indices[:100],  # Limit output
        "outlier_values": outlier_values[:100]
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]


async def main():
    """Main entry point for the Pandas MCP server"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await pandas_server.run(
            read_stream,
            write_stream,
            pandas_server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

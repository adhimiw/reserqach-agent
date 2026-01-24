"""
Visualizer Agent
Creates publication-quality visualizations for analysis results
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Optional
import json


class DataVisualizer:
    """Creates various types of visualizations for data analysis"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir or "output/visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Color schemes
        self.color_palette = sns.color_palette("husl", 12)
    
    def create_correlation_heatmap(self, df: pd.DataFrame, title: str = "Correlation Heatmap", 
                                figsize: tuple = (12, 10)) -> str:
        """
        Create correlation heatmap
        
        Args:
            df: DataFrame with numeric columns
            title: Plot title
            figsize: Figure size
        
        Returns:
            Path to saved file
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate correlation
        corr = numeric_df.corr()
        
        # Create heatmap
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_distribution_plot(self, df: pd.DataFrame, column: str, 
                              title: str = None) -> str:
        """
        Create distribution plot (histogram with KDE)
        
        Args:
            df: DataFrame
            column: Column name to plot
            title: Plot title
        
        Returns:
            Path to saved file
        """
        if title is None:
            title = f"Distribution of {column}"
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with KDE
        df[column].hist(bins=30, ax=axes[0], alpha=0.7, color=self.color_palette[0])
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Histogram of {column}')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        df.boxplot(column=column, ax=axes[1])
        axes[1].set_title(f'Box Plot of {column}')
        axes[1].set_ylabel(column)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        safe_col = column.replace('/', '_').replace('\\', '_')
        filepath = os.path.join(self.output_dir, f"distribution_{safe_col}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                          hue_col: str = None, title: str = None) -> str:
        """
        Create scatter plot
        
        Args:
            df: DataFrame
            x_col: X-axis column
            y_col: Y-axis column
            hue_col: Column for color grouping
            title: Plot title
        
        Returns:
            Path to saved file
        """
        if title is None:
            title = f"{x_col} vs {y_col}"
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if hue_col and hue_col in df.columns:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, 
                           alpha=0.6, ax=ax)
            ax.legend(title=hue_col)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6, 
                           color=self.color_palette[0], ax=ax)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        safe_title = title.replace(' ', '_').replace('/', '_')[:50]
        filepath = os.path.join(self.output_dir, f"scatter_{safe_title}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_line_plot(self, df: pd.DataFrame, x_col: str, y_cols: List[str],
                       title: str = None) -> str:
        """
        Create line plot for time series or trends
        
        Args:
            df: DataFrame
            x_col: X-axis column
            y_cols: List of Y-axis columns
            title: Plot title
        
        Returns:
            Path to saved file
        """
        if title is None:
            title = f"Trends"
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, y_col in enumerate(y_cols):
            if y_col in df.columns:
                ax.plot(df[x_col], df[y_col], label=y_col, 
                        linewidth=2, color=self.color_palette[i % len(self.color_palette)])
        
        ax.set_xlabel(x_col)
        ax.set_ylabel('Value')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        safe_title = title.replace(' ', '_').replace('/', '_')[:50]
        filepath = os.path.join(self.output_dir, f"line_{safe_title}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_bar_plot(self, df: pd.DataFrame, x_col: str, y_col: str = None,
                       title: str = None, top_n: int = 20) -> str:
        """
        Create bar plot
        
        Args:
            df: DataFrame
            x_col: X-axis column (categorical)
            y_col: Y-axis column (if None, uses count)
            title: Plot title
            top_n: Number of top categories to show
        
        Returns:
            Path to saved file
        """
        if title is None:
            title = f"{x_col}"
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if y_col and y_col in df.columns:
            # Aggregate by x_col
            plot_data = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        else:
            # Count occurrences
            plot_data = df[x_col].value_counts().head(top_n)
        
        plot_data.plot(kind='barh', ax=ax, color=self.color_palette[0])
        ax.set_xlabel('Value' if y_col else 'Count')
        ax.set_ylabel(x_col)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        safe_title = title.replace(' ', '_').replace('/', '_')[:50]
        filepath = os.path.join(self.output_dir, f"bar_{safe_title}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_pairplot(self, df: pd.DataFrame, hue_col: str = None,
                      title: str = "Pairwise Relationships") -> str:
        """
        Create pairplot for multivariate analysis
        
        Args:
            df: DataFrame
            hue_col: Column for color grouping
            title: Plot title
        
        Returns:
            Path to saved file
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return None
        
        # Limit to 6 columns for readability
        if len(numeric_df.columns) > 6:
            numeric_df = numeric_df[numeric_df.columns[:6]]
        
        if hue_col and hue_col in df.columns:
            hue_data = df[hue_col]
        else:
            hue_data = None
        
        fig = sns.pairplot(numeric_df, hue=hue_data, diag_kind='kde', 
                          plot_kws={'alpha': 0.6}, height=2.5)
        fig.fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
        
        filepath = os.path.join(self.output_dir, "pairplot.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_feature_importance_plot(self, feature_importance: pd.DataFrame,
                                    title: str = "Feature Importance") -> str:
        """
        Create feature importance bar plot
        
        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            title: Plot title
        
        Returns:
            Path to saved file
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_features = feature_importance.head(15)
        
        ax.barh(top_features['feature'], top_features['importance'], 
                color=self.color_palette[0])
        ax.set_xlabel('Importance')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        safe_title = title.replace(' ', '_').replace('/', '_')[:50]
        filepath = os.path.join(self.output_dir, f"feature_importance_{safe_title}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_residual_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = "Residual Plot") -> str:
        """
        Create residual plot for model evaluation
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
        
        Returns:
            Path to saved file
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, color=self.color_palette[0])
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        safe_title = title.replace(' ', '_').replace('/', '_')[:50]
        filepath = os.path.join(self.output_dir, f"residual_{safe_title}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_interactive_plot(self, df: pd.DataFrame, plot_type: str, 
                               x_col: str = None, y_col: str = None,
                               title: str = None) -> str:
        """
        Create interactive plot using Plotly
        
        Args:
            df: DataFrame
            plot_type: Type of plot ('scatter', 'line', 'bar', 'histogram')
            x_col: X-axis column
            y_col: Y-axis column
            title: Plot title
        
        Returns:
            Path to saved HTML file
        """
        if plot_type == 'scatter':
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, title=title or f"{x_col} vs {y_col}",
                               hover_data=df.columns.tolist())
            else:
                fig = px.scatter_matrix(df.select_dtypes(include=[np.number]).iloc[:, :5],
                                      title=title or "Scatter Matrix")
        
        elif plot_type == 'line':
            fig = px.line(df, x=x_col or df.index, y=y_col, 
                          title=title or "Line Plot")
        
        elif plot_type == 'bar':
            if y_col:
                plot_data = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(plot_data, x=x_col, y=y_col, 
                             title=title or f"{x_col} vs {y_col}")
            else:
                plot_data = df[x_col].value_counts().reset_index()
                plot_data.columns = [x_col, 'count']
                fig = px.bar(plot_data, x=x_col, y='count', 
                             title=title or f"Count of {x_col}")
        
        elif plot_type == 'histogram':
            fig = px.histogram(df, x=x_col, title=title or f"Distribution of {x_col}")
        
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        safe_title = (title or f"{plot_type}_plot").replace(' ', '_').replace('/', '_')[:50]
        filepath = os.path.join(self.output_dir, f"interactive_{safe_title}.html")
        fig.write_html(filepath)
        
        return filepath
    
    def create_dashboard(self, df: pd.DataFrame, title: str = "Data Dashboard") -> str:
        """
        Create a multi-panel dashboard
        
        Args:
            df: DataFrame
            title: Dashboard title
        
        Returns:
            Path to saved file
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Create subplot grid
        n_plots = min(4, len(numeric_cols))
        if n_plots < 1:
            n_plots = 1
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Correlation heatmap (if enough columns)
        if len(numeric_cols) >= 2:
            ax1 = fig.add_subplot(gs[0, 0])
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.1f', cmap='coolwarm', 
                       center=0, square=True, ax=ax1, cbar=False)
            ax1.set_title('Correlation Matrix', fontweight='bold')
        
        # Plot 2: Distribution of first numeric column
        if numeric_cols:
            ax2 = fig.add_subplot(gs[0, 1])
            df[numeric_cols[0]].hist(bins=30, ax=ax2, alpha=0.7, 
                                      color=self.color_palette[0])
            ax2.set_title(f'Distribution: {numeric_cols[0]}', fontweight='bold')
            ax2.set_xlabel(numeric_cols[0])
            ax2.set_ylabel('Frequency')
        
        # Plot 3: Bar plot of first categorical column
        if categorical_cols:
            ax3 = fig.add_subplot(gs[1, 0])
            top_values = df[categorical_cols[0]].value_counts().head(10)
            top_values.plot(kind='bar', ax=ax3, color=self.color_palette[1])
            ax3.set_title(f'Top 10: {categorical_cols[0]}', fontweight='bold')
            ax3.set_xlabel(categorical_cols[0])
            ax3.set_ylabel('Count')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Summary statistics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # Create summary table
        summary_text = f"Dataset Summary\n\n"
        summary_text += f"Rows: {len(df):,}\n"
        summary_text += f"Columns: {len(df.columns)}\n"
        summary_text += f"Numeric: {len(numeric_cols)}\n"
        summary_text += f"Categorical: {len(categorical_cols)}\n"
        summary_text += f"Missing Values: {df.isnull().sum().sum():,}\n"
        summary_text += f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n"
        
        ax4.text(0.1, 0.5, summary_text, fontsize=12, 
                verticalalignment='center', family='monospace')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        safe_title = title.replace(' ', '_').replace('/', '_')[:50]
        filepath = os.path.join(self.output_dir, f"dashboard_{safe_title}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def generate_all_visualizations(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate all relevant visualizations for a dataset
        
        Args:
            df: DataFrame to visualize
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualizations = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Dashboard
        visualizations['dashboard'] = self.create_dashboard(df, "Data Overview Dashboard")
        
        # Correlation heatmap
        if len(numeric_cols) >= 2:
            visualizations['correlation_heatmap'] = self.create_correlation_heatmap(df)
        
        # Distribution plots for numeric columns (top 5)
        for col in numeric_cols[:5]:
            visualizations[f'distribution_{col}'] = self.create_distribution_plot(df, col)
        
        # Scatter plots for correlated pairs
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().abs()
            top_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    top_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
            
            top_pairs.sort(key=lambda x: x[2], reverse=True)
            
            for col1, col2, corr_val in top_pairs[:3]:
                visualizations[f'scatter_{col1}_vs_{col2}'] = self.create_scatter_plot(
                    df, col1, col2, 
                    title=f"{col1} vs {col2} (r={corr_val:.2f})"
                )
        
        # Bar plots for categorical columns (top 3)
        for col in categorical_cols[:3]:
            visualizations[f'bar_{col}'] = self.create_bar_plot(df, col)
        
        # Pairplot (if small enough)
        if len(numeric_cols) >= 2 and len(numeric_cols) <= 6:
            visualizations['pairplot'] = self.create_pairplot(df)
        
        # Remove None values
        visualizations = {k: v for k, v in visualizations.items() if v is not None}
        
        return visualizations
    
    def format_visualizations_for_report(self, visualizations: Dict[str, str]) -> str:
        """
        Format visualization information for report
        
        Args:
            visualizations: Dictionary mapping names to file paths
        
        Returns:
            Formatted Markdown string
        """
        report = "## Visualizations\n\n"
        
        for name, path in visualizations.items():
            filename = os.path.basename(path)
            display_name = name.replace('_', ' ').title()
            report += f"### {display_name}\n\n"
            report += f"![{display_name}]({filename})\n\n"
        
        return report

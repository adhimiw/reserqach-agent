"""
Hypothesis Generation Engine
Automatically generates testable hypotheses from dataset characteristics
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)

class HypothesisGenerator:
    """Generates testable hypotheses from data patterns"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize hypothesis generator
        
        Args:
            df: Input DataFrame to analyze
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        self.hypotheses = []
    
    def generate_correlation_hypotheses(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Generate hypotheses based on correlation analysis
        
        Args:
            threshold: Minimum absolute correlation to generate hypothesis
        
        Returns:
            List of correlation-based hypotheses
        """
        if len(self.numeric_cols) < 2:
            return []
        
        corr_matrix = self.df[self.numeric_cols].corr()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) >= threshold:
                    direction = "positive" if corr_val > 0 else "negative"
                    strength = "strong" if abs(corr_val) >= 0.7 else "moderate"
                    
                    hypothesis = {
                        "type": "correlation",
                        "id": f"corr_{len(self.hypotheses) + 1}",
                        "column1": col1,
                        "column2": col2,
                        "correlation_value": round(corr_val, 3),
                        "direction": direction,
                        "strength": strength,
                        "hypothesis": f"There is a {strength} {direction} correlation between {col1} and {col2} (r = {corr_val:.3f}).",
                        "test_method": "Pearson correlation test",
                        "reasoning": f"The correlation coefficient of {corr_val:.3f} indicates a {'significant' if abs(corr_val) >= 0.5 else 'moderate'} linear relationship. This suggests changes in {col1} are associated with changes in {col2}."
                    }
                    self.hypotheses.append(hypothesis)
        
        return [h for h in self.hypotheses if h["type"] == "correlation"]
    
    def generate_distribution_hypotheses(self) -> List[Dict[str, Any]]:
        """
        Generate hypotheses based on data distribution analysis
        
        Returns:
            List of distribution-based hypotheses
        """
        hypotheses = []
        
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            
            # Skewness hypothesis
            skewness = data.skew()
            if abs(skewness) > 1:
                skew_type = "right" if skewness > 0 else "left"
                hypothesis = {
                    "type": "distribution",
                    "id": f"dist_{len(self.hypotheses) + 1}",
                    "column": col,
                    "metric": "skewness",
                    "value": round(skewness, 3),
                    "hypothesis": f"The distribution of {col} is {skew_type}-skewed (skewness = {skewness:.3f}).",
                    "test_method": "Shapiro-Wilk normality test",
                    "reasoning": f"The skewness value of {skewness:.3f} indicates the data is {'not normally distributed and has a long tail to the right' if skew_type == 'right' else 'not normally distributed and has a long tail to the left'}. This suggests potential outliers or underlying factors affecting the distribution."
                }
                self.hypotheses.append(hypothesis)
                hypotheses.append(hypothesis)
            
            # Kurtosis hypothesis
            kurtosis = data.kurtosis()
            if abs(kurtosis) > 3:
                tail_type = "heavy-tailed" if kurtosis > 0 else "light-tailed"
                hypothesis = {
                    "type": "distribution",
                    "id": f"dist_{len(self.hypotheses) + 1}",
                    "column": col,
                    "metric": "kurtosis",
                    "value": round(kurtosis, 3),
                    "hypothesis": f"The distribution of {col} is {tail_type} (kurtosis = {kurtosis:.3f}).",
                    "test_method": "Kurtosis analysis",
                    "reasoning": f"The kurtosis value of {kurtosis:.3f} indicates {'more extreme outliers' if kurtosis > 0 else 'fewer extreme outliers'} than a normal distribution. This {tail_type} distribution suggests {'potential extreme events or data anomalies' if kurtosis > 0 else 'more stable or constrained behavior'}."
                }
                self.hypotheses.append(hypothesis)
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def generate_categorical_hypotheses(self) -> List[Dict[str, Any]]:
        """
        Generate hypotheses based on categorical variable analysis
        
        Returns:
            List of categorical-based hypotheses
        """
        hypotheses = []
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            total_count = len(self.df[col].dropna())
            
            # Dominant category hypothesis
            dominant_pct = (value_counts.iloc[0] / total_count) * 100
            if dominant_pct > 70:
                hypothesis = {
                    "type": "categorical",
                    "id": f"cat_{len(self.hypotheses) + 1}",
                    "column": col,
                    "dominant_category": value_counts.index[0],
                    "dominant_percentage": round(dominant_pct, 2),
                    "unique_values": len(value_counts),
                    "hypothesis": f"One category ({value_counts.index[0]}) dominates the {col} variable with {dominant_pct:.2f}% of all observations.",
                    "test_method": "Chi-square test for uniformity",
                    "reasoning": f"The concentration of {dominant_pct:.2f}% of observations in a single category indicates a strong imbalance. This {col} variable may need special handling in analysis, as the dominant category may drive most patterns."
                }
                self.hypotheses.append(hypothesis)
                hypotheses.append(hypothesis)
            
            # Rare categories hypothesis
            rare_categories = value_counts[value_counts < total_count * 0.05]
            if len(rare_categories) > 0:
                hypothesis = {
                    "type": "categorical",
                    "id": f"cat_{len(self.hypotheses) + 1}",
                    "column": col,
                    "rare_categories": len(rare_categories),
                    "rare_percentage": round((len(rare_categories) / len(value_counts)) * 100, 2),
                    "hypothesis": f"There are {len(rare_categories)} rare categories in {col} (each <5% of observations), representing {len(rare_categories)/len(value_counts)*100:.2f}% of all categories.",
                    "test_method": "Frequency analysis",
                    "reasoning": f"The presence of {len(rare_categories)} rare categories suggests high cardinality or niche segments. These rare categories {('may need to be grouped or handled separately' if len(rare_categories) > 10 else 'could indicate outliers or special cases')}."
                }
                self.hypotheses.append(hypothesis)
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def generate_outlier_hypotheses(self, method: str = "iqr") -> List[Dict[str, Any]]:
        """
        Generate hypotheses based on outlier detection
        
        Args:
            method: Outlier detection method ('iqr' or 'zscore')
        
        Returns:
            List of outlier-based hypotheses
        """
        hypotheses = []
        threshold = 1.5 if method == "iqr" else 3
        
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            
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
            
            outlier_count = outliers_mask.sum()
            outlier_pct = (outlier_count / len(data)) * 100
            
            if outlier_count > 0:
                severity = "severe" if outlier_pct > 5 else "moderate" if outlier_pct > 1 else "mild"
                
                # Get outlier range
                outlier_values = data[outliers_mask]
                
                hypothesis = {
                    "type": "outlier",
                    "id": f"out_{len(self.hypotheses) + 1}",
                    "column": col,
                    "outlier_count": int(outlier_count),
                    "outlier_percentage": round(outlier_pct, 2),
                    "severity": severity,
                    "outlier_range": {
                        "min": float(outlier_values.min()),
                        "max": float(outlier_values.max())
                    },
                    "hypothesis": f"{col} contains {outlier_count} outliers ({outlier_pct:.2f}% of data), indicating {severity} deviation from typical values.",
                    "test_method": f"{method.upper()} outlier detection",
                    "reasoning": f"The presence of {severity} outliers ({outlier_pct:.2f}% of data) in {col} suggests {('significant data quality issues or genuine extreme events' if outlier_pct > 5 else 'some unusual observations that may require investigation')}. These outliers range from {outlier_values.min():.2f} to {outlier_values.max():.2f}."
                }
                self.hypotheses.append(hypothesis)
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def generate_trend_hypotheses(self, date_col: str = None) -> List[Dict[str, Any]]:
        """
        Generate hypotheses based on trend analysis (requires datetime column)
        
        Args:
            date_col: Name of datetime column (auto-detected if not provided)
        
        Returns:
            List of trend-based hypotheses
        """
        hypotheses = []
        
        if not self.datetime_cols:
            return []
        
        date_col = date_col or self.datetime_cols[0]
        
        for col in self.numeric_cols:
            # Create a temporary dataframe with the date column
            temp_df = self.df[[date_col, col]].copy()
            temp_df = temp_df.dropna()
            temp_df = temp_df.sort_values(date_col)
            
            if len(temp_df) < 10:
                continue
            
            # Calculate trend using linear regression
            x = np.arange(len(temp_df))
            y = temp_df[col].values
            
            # Simple linear regression
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
            
            # Calculate R-squared
            y_pred = slope * x + np.mean(y)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            if abs(slope) > 0 and r_squared > 0.3:
                trend_type = "increasing" if slope > 0 else "decreasing"
                strength = "strong" if r_squared > 0.7 else "moderate"
                
                hypothesis = {
                    "type": "trend",
                    "id": f"trend_{len(self.hypotheses) + 1}",
                    "column": col,
                    "date_column": date_col,
                    "slope": round(slope, 6),
                    "r_squared": round(r_squared, 3),
                    "trend_type": trend_type,
                    "strength": strength,
                    "hypothesis": f"{col} shows a {strength} {trend_type} trend over time (R² = {r_squared:.3f}).",
                    "test_method": "Linear regression trend analysis",
                    "reasoning": f"The linear regression reveals a {trend_type} trend with a slope of {slope:.6f} and R² of {r_squared:.3f}. This {strength} trend suggests {('consistent growth' if trend_type == 'increasing' else 'consistent decline')} in {col} over time, which {('may indicate positive momentum' if trend_type == 'increasing' else 'could signal underlying issues or market saturation')}."
                }
                self.hypotheses.append(hypothesis)
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def generate_all_hypotheses(self, max_hypotheses: int = 100) -> List[Dict[str, Any]]:
        """
        Generate all types of hypotheses
        
        Args:
            max_hypotheses: Maximum number of hypotheses to generate
        
        Returns:
            List of all generated hypotheses
        """
        self.hypotheses = []
        
        # Generate hypotheses of different types
        self.generate_correlation_hypotheses(threshold=0.3)
        self.generate_distribution_hypotheses()
        self.generate_categorical_hypotheses()
        self.generate_outlier_hypotheses(method="iqr")
        
        if self.datetime_cols:
            self.generate_trend_hypotheses()
        
        # Sort hypotheses by type and limit
        self.hypotheses = sorted(self.hypotheses, key=lambda x: x["type"])
        
        if len(self.hypotheses) > max_hypotheses:
            # Take balanced sample from each type
            type_counts = {}
            balanced_hypotheses = []
            
            for h in self.hypotheses:
                htype = h["type"]
                if htype not in type_counts:
                    type_counts[htype] = 0
                if type_counts[htype] < max_hypotheses // len(set(h["type"] for h in self.hypotheses)):
                    balanced_hypotheses.append(h)
                    type_counts[htype] += 1
            
            self.hypotheses = balanced_hypotheses
        
        return self.hypotheses
    
    def format_hypotheses_for_report(self) -> str:
        """
        Format hypotheses for inclusion in a report
        
        Returns:
            Formatted Markdown string
        """
        if not self.hypotheses:
            return "No hypotheses were generated from the data."
        
        # Group by type
        by_type = {}
        for h in self.hypotheses:
            htype = h["type"]
            if htype not in by_type:
                by_type[htype] = []
            by_type[htype].append(h)
        
        report = "# Generated Hypotheses\n\n"
        report += f"Total hypotheses generated: {len(self.hypotheses)}\n\n"
        
        type_headers = {
            "correlation": "## Correlation-Based Hypotheses",
            "distribution": "## Distribution-Based Hypotheses",
            "categorical": "## Categorical Variable Hypotheses",
            "outlier": "## Outlier-Based Hypotheses",
            "trend": "## Trend-Based Hypotheses"
        }
        
        for htype, hypotheses in by_type.items():
            report += f"{type_headers.get(htype, f'## {htype.capitalize()} Hypotheses')}\n\n"
            
            for h in hypotheses:
                report += f"### {h.get('hypothesis', 'Unknown Hypothesis')}\n\n"
                report += f"**Why:** {h.get('reasoning', 'N/A')}\n\n"
                report += f"**Test Method:** {h.get('test_method', 'N/A')}\n\n"
                
                # Add relevant details based on type
                if htype == "correlation":
                    report += f"**Correlation Value:** {h.get('correlation_value', 'N/A')}\n"
                    report += f"**Strength:** {h.get('strength', 'N/A')}\n"
                    report += f"**Direction:** {h.get('direction', 'N/A')}\n\n"
                elif htype == "distribution":
                    report += f"**Metric:** {h.get('metric', 'N/A')}\n"
                    report += f"**Value:** {h.get('value', 'N/A')}\n\n"
                elif htype == "categorical":
                    report += f"**Unique Values:** {h.get('unique_values', 'N/A')}\n\n"
                elif htype == "outlier":
                    report += f"**Outlier Count:** {h.get('outlier_count', 'N/A')}\n"
                    report += f"**Percentage:** {h.get('outlier_percentage', 'N/A')}%\n"
                    report += f"**Severity:** {h.get('severity', 'N/A')}\n\n"
                elif htype == "trend":
                    report += f"**Trend Type:** {h.get('trend_type', 'N/A')}\n"
                    report += f"**Strength:** {h.get('strength', 'N/A')}\n"
                    report += f"**R-Squared:** {h.get('r_squared', 'N/A')}\n\n"
                
                report += "---\n\n"
        
        return report
    
    def save_hypotheses(self, filepath: str):
        """
        Save hypotheses to a JSON file
        
        Args:
            filepath: Path to save the hypotheses
        """
        with open(filepath, 'w') as f:
            json.dump(self.hypotheses, f, indent=2, cls=NumpyEncoder)

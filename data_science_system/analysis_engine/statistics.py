"""
Statistical Testing Engine
Performs statistical tests to validate hypotheses and analyze data patterns
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal
from typing import Dict, Any, List, Optional, Tuple
import json


class StatisticalTester:
    """Performs various statistical tests on data"""
    
    def __init__(self, df: pd.DataFrame, significance_level: float = 0.05):
        """
        Initialize statistical tester
        
        Args:
            df: Input DataFrame
            significance_level: Alpha value for hypothesis testing
        """
        self.df = df
        self.alpha = significance_level
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.results = []
    
    def test_correlation(self, col1: str, col2: str, method: str = 'pearson') -> Dict[str, Any]:
        """
        Test correlation between two numerical variables
        
        Args:
            col1: First column name
            col2: Second column name
            method: Correlation method ('pearson', 'spearman', 'kendall')
        
        Returns:
            Dictionary with test results
        """
        if col1 not in self.numeric_cols or col2 not in self.numeric_cols:
            return {"error": f"One or both columns are not numeric"}
        
        # Remove missing values
        data = self.df[[col1, col2]].dropna()
        
        if len(data) < 3:
            return {"error": "Insufficient data for correlation test"}
        
        # Calculate correlation
        if method == 'pearson':
            corr, p_value = stats.pearsonr(data[col1], data[col2])
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(data[col1], data[col2])
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(data[col1], data[col2])
        else:
            return {"error": f"Unknown correlation method: {method}"}
        
        # Determine significance
        significant = p_value < self.alpha
        
        result = {
            "test": "correlation",
            "method": method,
            "column1": col1,
            "column2": col2,
            "correlation_coefficient": round(corr, 4),
            "p_value": round(p_value, 6),
            "significant": significant,
            "interpretation": self._interpret_correlation(corr, p_value, significant),
            "sample_size": len(data)
        }
        
        self.results.append(result)
        return result
    
    def _interpret_correlation(self, corr: float, p_value: float, significant: bool) -> str:
        """Interpret correlation results"""
        strength = "very weak" if abs(corr) < 0.1 else "weak" if abs(corr) < 0.3 else "moderate" if abs(corr) < 0.5 else "strong"
        direction = "positive" if corr > 0 else "negative"
        
        interpretation = f"There is a {strength} {direction} correlation (r = {corr:.4f})"
        
        if significant:
            interpretation += f" that is statistically significant (p = {p_value:.6f}, α = {self.alpha}). "
            interpretation += "This suggests the relationship is unlikely due to random chance."
        else:
            interpretation += f" that is not statistically significant (p = {p_value:.6f}, α = {self.alpha}). "
            interpretation += "This suggests the relationship may be due to random chance."
        
        return interpretation
    
    def test_normality(self, column: str) -> Dict[str, Any]:
        """
        Test if a column follows a normal distribution using Shapiro-Wilk test
        
        Args:
            column: Column name to test
        
        Returns:
            Dictionary with test results
        """
        if column not in self.numeric_cols:
            return {"error": f"Column {column} is not numeric"}
        
        data = self.df[column].dropna()
        
        # Shapiro-Wilk test has a limit of 5000 samples
        if len(data) > 5000:
            data = data.sample(5000, random_state=42)
        
        if len(data) < 3:
            return {"error": "Insufficient data for normality test"}
        
        statistic, p_value = stats.shapiro(data)
        
        normal = p_value > self.alpha
        
        result = {
            "test": "normality",
            "method": "Shapiro-Wilk",
            "column": column,
            "statistic": round(statistic, 4),
            "p_value": round(p_value, 6),
            "normal": normal,
            "interpretation": self._interpret_normality(statistic, p_value, normal),
            "sample_size": len(data)
        }
        
        self.results.append(result)
        return result
    
    def _interpret_normality(self, statistic: float, p_value: float, normal: bool) -> str:
        """Interpret normality test results"""
        if normal:
            interpretation = f"The data appears to follow a normal distribution (p = {p_value:.6f} > α = {self.alpha}). "
            interpretation += "This means the data is symmetrically distributed around the mean, and parametric tests are appropriate."
        else:
            interpretation = f"The data does not follow a normal distribution (p = {p_value:.6f} ≤ α = {self.alpha}). "
            interpretation += "This means the data is skewed or has outliers, and non-parametric tests or transformations may be needed."
        
        return interpretation
    
    def test_outliers(self, column: str, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect and test for outliers
        
        Args:
            column: Column name to test
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            Dictionary with outlier analysis
        """
        if column not in self.numeric_cols:
            return {"error": f"Column {column} is not numeric"}
        
        data = self.df[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers_mask = (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers_mask = z_scores > threshold
        else:
            return {"error": f"Unknown method: {method}"}
        
        outliers = data[outliers_mask]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(data)) * 100
        
        result = {
            "test": "outliers",
            "method": method,
            "column": column,
            "threshold": threshold,
            "outlier_count": outlier_count,
            "outlier_percentage": round(outlier_pct, 2),
            "outlier_min": float(outliers.min()) if outlier_count > 0 else None,
            "outlier_max": float(outliers.max()) if outlier_count > 0 else None,
            "total_observations": len(data),
            "interpretation": self._interpret_outliers(outlier_count, outlier_pct, method)
        }
        
        self.results.append(result)
        return result
    
    def _interpret_outliers(self, count: int, pct: float, method: str) -> str:
        """Interpret outlier results"""
        severity = "severe" if pct > 5 else "moderate" if pct > 1 else "mild"
        
        if count == 0:
            return "No outliers were detected. The data appears clean with no extreme values."
        
        interpretation = f"Detected {count} outliers ({pct:.2f}% of data), indicating {severity} deviation from typical values. "
        
        if pct > 5:
            interpretation += "The high percentage of outliers suggests potential data quality issues or the presence of multiple distinct subgroups. Investigation is recommended."
        elif pct > 1:
            interpretation += "The moderate number of outliers may indicate some unusual observations that should be examined individually."
        else:
            interpretation += "The small number of outliers is typical and may represent natural variation or rare but valid cases."
        
        return interpretation
    
    def test_group_differences(self, numerical_col: str, categorical_col: str, method: str = 'auto') -> Dict[str, Any]:
        """
        Test if there are significant differences between groups
        
        Args:
            numerical_col: Numerical column to test
            categorical_col: Categorical column defining groups
            method: Test method ('auto', 'anova', 'kruskal', 'ttest')
        
        Returns:
            Dictionary with test results
        """
        if numerical_col not in self.numeric_cols:
            return {"error": f"Column {numerical_col} is not numeric"}
        if categorical_col not in self.categorical_cols:
            return {"error": f"Column {categorical_col} is not categorical"}
        
        # Get data by groups
        groups = [group[numerical_col].dropna() for name, group in self.df.groupby(categorical_col)]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            return {"error": "Need at least 2 groups for comparison"}
        
        # Auto-select test
        if method == 'auto':
            # Check normality of each group
            normal_groups = 0
            for group in groups:
                if len(group) >= 3 and len(group) <= 5000:
                    _, p = stats.shapiro(group.sample(min(5000, len(group)), random_state=42))
                    if p > self.alpha:
                        normal_groups += 1
            
            # If most groups are normal, use ANOVA, else Kruskal-Wallis
            method = 'anova' if normal_groups / len(groups) >= 0.7 else 'kruskal'
        
        # Perform test
        if method == 'anova':
            statistic, p_value = f_oneway(*groups)
            test_name = "One-way ANOVA"
        elif method == 'kruskal':
            statistic, p_value = kruskal(*groups)
            test_name = "Kruskal-Wallis"
        else:
            return {"error": f"Unknown method: {method}"}
        
        significant = p_value < self.alpha
        
        result = {
            "test": "group_differences",
            "method": test_name,
            "numerical_column": numerical_col,
            "categorical_column": categorical_col,
            "statistic": round(statistic, 4),
            "p_value": round(p_value, 6),
            "significant": significant,
            "number_of_groups": len(groups),
            "interpretation": self._interpret_group_differences(significant, p_value, test_name, numerical_col, categorical_col),
            "group_means": [round(g.mean(), 4) for g in groups],
            "group_sizes": [len(g) for g in groups]
        }
        
        self.results.append(result)
        return result
    
    def _interpret_group_differences(self, significant: bool, p_value: float, test_name: str, 
                                     num_col: str, cat_col: str) -> str:
        """Interpret group differences test"""
        if significant:
            interpretation = f"There are statistically significant differences in {num_col} between groups of {cat_col} "
            interpretation += f"({test_name}: F = {self.results[-1]['statistic']}, p = {p_value:.6f}, α = {self.alpha}). "
            interpretation += "This suggests that {cat_col} has a meaningful effect on {num_col}."
        else:
            interpretation = f"There are no statistically significant differences in {num_col} between groups of {cat_col} "
            interpretation += f"({test_name}: F = {self.results[-1]['statistic']}, p = {p_value:.6f}, α = {self.alpha}). "
            interpretation += "This suggests that {cat_col} may not have a meaningful effect on {num_col}, or the sample size is insufficient to detect differences."
        
        return interpretation
    
    def test_categorical_association(self, col1: str, col2: str) -> Dict[str, Any]:
        """
        Test association between two categorical variables using Chi-square test
        
        Args:
            col1: First categorical column
            col2: Second categorical column
        
        Returns:
            Dictionary with test results
        """
        if col1 not in self.categorical_cols or col2 not in self.categorical_cols:
            return {"error": "Both columns must be categorical"}
        
        # Create contingency table
        contingency_table = pd.crosstab(self.df[col1], self.df[col2])
        
        # Check assumptions
        if contingency_table.sum().sum() < 25:
            return {"error": "Insufficient data for chi-square test (<25 observations)"}
        
        # Chi-square test
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        except ValueError as e:
            return {"error": f"Chi-square test failed: {str(e)}"}
        
        significant = p_value < self.alpha
        
        # Calculate Cramér's V for effect size
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        result = {
            "test": "categorical_association",
            "method": "Chi-square test of independence",
            "column1": col1,
            "column2": col2,
            "chi2_statistic": round(chi2, 4),
            "p_value": round(p_value, 6),
            "degrees_of_freedom": dof,
            "significant": significant,
            "cramers_v": round(cramers_v, 4),
            "interpretation": self._interpret_categorical_association(significant, p_value, cramers_v),
            "contingency_table": contingency_table.to_dict()
        }
        
        self.results.append(result)
        return result
    
    def _interpret_categorical_association(self, significant: bool, p_value: float, cramers_v: float) -> str:
        """Interpret categorical association"""
        effect_size = "negligible" if cramers_v < 0.1 else "small" if cramers_v < 0.3 else "medium" if cramers_v < 0.5 else "large"
        
        if significant:
            interpretation = f"There is a statistically significant association between the variables (χ² = {self.results[-1]['chi2_statistic']}, "
            interpretation += f"p = {p_value:.6f}, α = {self.alpha}). "
            interpretation += f"The effect size is {effect_size} (Cramér's V = {cramers_v:.4f}). "
            interpretation += "This suggests the variables are related and not independent."
        else:
            interpretation = f"There is no statistically significant association between the variables (χ² = {self.results[-1]['chi2_statistic']}, "
            interpretation += f"p = {p_value:.6f}, α = {self.alpha}). "
            interpretation += "This suggests the variables are independent."
        
        return interpretation
    
    def run_comprehensive_tests(self) -> List[Dict[str, Any]]:
        """
        Run a comprehensive suite of statistical tests
        
        Returns:
            List of all test results
        """
        self.results = []
        
        # Correlation tests for numeric pairs
        for i in range(len(self.numeric_cols)):
            for j in range(i+1, len(self.numeric_cols)):
                col1, col2 = self.numeric_cols[i], self.numeric_cols[j]
                self.test_correlation(col1, col2, method='pearson')
        
        # Normality tests for numeric columns
        for col in self.numeric_cols:
            self.test_normality(col)
        
        # Outlier tests for numeric columns
        for col in self.numeric_cols:
            self.test_outliers(col, method='iqr')
        
        # Group differences (if categorical columns exist)
        if self.categorical_cols and self.numeric_cols:
            for cat_col in self.categorical_cols:
                for num_col in self.numeric_cols:
                    try:
                        self.test_group_differences(num_col, cat_col, method='auto')
                    except Exception:
                        pass
        
        # Categorical associations
        for i in range(len(self.categorical_cols)):
            for j in range(i+1, len(self.categorical_cols)):
                try:
                    self.test_categorical_association(self.categorical_cols[i], self.categorical_cols[j])
                except Exception:
                    pass
        
        return self.results
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics
        
        Returns:
            Dictionary with summary statistics
        """
        numeric_summary = self.df[self.numeric_cols].describe().to_dict() if self.numeric_cols else {}
        categorical_summary = {}
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            categorical_summary[col] = {
                "unique_count": len(value_counts),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "most_common_percentage": round((value_counts.iloc[0] / len(self.df[col])) * 100, 2)
            }
        
        return {
            "dataset_shape": self.df.shape,
            "numeric_columns": self.numeric_cols,
            "categorical_columns": self.categorical_cols,
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
            "missing_values": self.df.isnull().sum().to_dict(),
            "missing_percentage": (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict()
        }
    
    def format_results_for_report(self) -> str:
        """
        Format test results for inclusion in a report
        
        Returns:
            Formatted Markdown string
        """
        if not self.results:
            return "No statistical tests were performed."
        
        report = "# Statistical Analysis Results\n\n"
        
        # Group by test type
        by_type = {}
        for r in self.results:
            test_type = r['test']
            if test_type not in by_type:
                by_type[test_type] = []
            by_type[test_type].append(r)
        
        type_titles = {
            "correlation": "## Correlation Tests",
            "normality": "## Normality Tests",
            "outliers": "## Outlier Analysis",
            "group_differences": "## Group Differences",
            "categorical_association": "## Categorical Association Tests"
        }
        
        for test_type, results in by_type.items():
            report += f"{type_titles.get(test_type, f'## {test_type.title()}')}\n\n"
            
            for r in results:
                report += f"### {r.get('test_method', test_type.title())}\n\n"
                report += f"**Finding:** {r.get('interpretation', 'No interpretation available.')}\n\n"
                
                # Add key statistics
                if test_type == "correlation":
                    report += f"- **Columns:** {r['column1']} vs {r['column2']}\n"
                    report += f"- **Correlation:** {r['correlation_coefficient']}\n"
                    report += f"- **P-value:** {r['p_value']}\n"
                    report += f"- **Significant:** {r['significant']}\n\n"
                elif test_type == "normality":
                    report += f"- **Column:** {r['column']}\n"
                    report += f"- **Statistic:** {r['statistic']}\n"
                    report += f"- **P-value:** {r['p_value']}\n"
                    report += f"- **Normal:** {r['normal']}\n\n"
                elif test_type == "outliers":
                    report += f"- **Column:** {r['column']}\n"
                    report += f"- **Outliers:** {r['outlier_count']} ({r['outlier_percentage']}%)\n"
                    report += f"- **Range:** [{r['outlier_min']}, {r['outlier_max']}]\n\n"
                elif test_type == "group_differences":
                    report += f"- **Numerical Column:** {r['numerical_column']}\n"
                    report += f"- **Categorical Column:** {r['categorical_column']}\n"
                    report += f"- **Number of Groups:** {r['number_of_groups']}\n"
                    report += f"- **P-value:** {r['p_value']}\n"
                    report += f"- **Significant:** {r['significant']}\n\n"
                elif test_type == "categorical_association":
                    report += f"- **Columns:** {r['column1']} vs {r['column2']}\n"
                    report += f"- **Chi-square:** {r['chi2_statistic']}\n"
                    report += f"- **P-value:** {r['p_value']}\n"
                    report += f"- **Cramér's V:** {r['cramers_v']}\n"
                    report += f"- **Significant:** {r['significant']}\n\n"
                
                report += "---\n\n"
        
        return report
    
    def save_results(self, filepath: str):
        """
        Save test results to a JSON file
        
        Args:
            filepath: Path to save the results
        """
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

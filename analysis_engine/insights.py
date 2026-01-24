"""
Insight Extraction and Formatting
Extracts meaningful insights from analysis results with "why" and "how" explanations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

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

class InsightExtractor:
    """Extracts and formats insights from analysis results"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize insight extractor
        
        Args:
            df: Original DataFrame being analyzed
        """
        self.df = df
        self.insights = []
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def extract_correlation_insights(self, correlation_results: List[Dict]) -> List[Dict]:
        """
        Extract insights from correlation analysis
        
        Args:
            correlation_results: List of correlation test results
        
        Returns:
            List of correlation insights
        """
        insights = []
        
        if not correlation_results:
            return insights
        
        for result in correlation_results:
            if result.get('significant', False):
                col1 = result['column1']
                col2 = result['column2']
                corr = result['correlation_coefficient']
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) >= 0.7 else "moderate" if abs(corr) >= 0.5 else "weak"
                
                insight = {
                    "id": f"corr_insight_{len(self.insights) + 1}",
                    "type": "correlation",
                    "title": f"{strength.capitalize()} {direction.capitalize()} Correlation Between {col1} and {col2}",
                    "what": f"There is a {strength} {direction} correlation of {corr:.3f} between {col1} and {col2}.",
                    "why": self._explain_correlation_why(col1, col2, corr, direction),
                    "how": self._explain_correlation_how(col1, col2, direction),
                    "recommendation": self._recommend_correlation_action(col1, col2, corr),
                    "data_support": {
                        "correlation": corr,
                        "p_value": result.get('p_value'),
                        "sample_size": result.get('sample_size')
                    }
                }
                insights.append(insight)
                self.insights.append(insight)
        
        return insights
    
    def _explain_correlation_why(self, col1: str, col2: str, corr: float, direction: str) -> str:
        """Explain why correlation exists"""
        abs_corr = abs(corr)
        
        if abs_corr >= 0.8:
            explanation = f"The strong {direction} correlation (r={corr:.3f}) indicates that {col1} and {col2} move together very closely. "
            if direction == "positive":
                explanation += f"When {col1} increases, {col2} tends to increase as well, and vice versa. This suggests a strong direct relationship."
            else:
                explanation += f"When {col1} increases, {col2} tends to decrease, and vice versa. This suggests a strong inverse relationship."
        elif abs_corr >= 0.5:
            explanation = f"The moderate {direction} correlation (r={corr:.3f}) shows that {col1} and {col2} are related, but the relationship is not perfect. "
            explanation += f"There may be other factors influencing this relationship, or it may vary in different contexts."
        else:
            explanation = f"The weak {direction} correlation (r={corr:.3f}) suggests a slight tendency for {col1} and {col2} to move in the {direction} direction. "
            explanation += "However, the relationship is weak and may not be practically significant."
        
        return explanation
    
    def _explain_correlation_how(self, col1: str, col2: str, direction: str) -> str:
        """Explain how to use this insight"""
        explanation = f"This correlation can be used in several ways:\n\n"
        
        if direction == "positive":
            explanation += f"1. **Prediction**: If you know the value of {col1}, you can estimate {col2} with moderate confidence.\n"
            explanation += f"2. **Monitoring**: Track {col1} as an indicator for {col2} - if {col1} changes, expect {col2} to follow.\n"
            explanation += f"3. **Causation Investigation**: Consider whether {col1} causes {col2}, {col2} causes {col1}, or if a third factor influences both.\n"
        else:
            explanation += f"1. **Trade-off Analysis**: This inverse relationship suggests a trade-off between {col1} and {col2}.\n"
            explanation += f"2. **Optimization**: Use this relationship to find optimal balance points between competing metrics.\n"
            explanation += f"3. **Resource Allocation**: Increasing {col1} may help reduce {col2}, potentially optimizing overall outcomes.\n"
        
        explanation += f"4. **Model Building**: Include both variables together in predictive models to capture their shared variance."
        
        return explanation
    
    def _recommend_correlation_action(self, col1: str, col2: str, corr: float) -> str:
        """Recommend actions based on correlation"""
        abs_corr = abs(corr)
        
        if abs_corr >= 0.7:
            return f"**Strong Recommendation**: This strong relationship is critical. Consider creating a joint KPI for {col1} and {col2}, investigate potential causal links, and use this relationship for forecasting and resource planning."
        elif abs_corr >= 0.5:
            return f"**Moderate Recommendation**: This relationship is useful for monitoring and decision-making. Use it as a leading indicator and consider including both variables in your core dashboard."
        else:
            return f"**Low Priority**: While statistically significant, this weak relationship may not have practical impact. Use as supplementary information rather than a primary decision driver."
    
    def extract_distribution_insights(self, distribution_results: List[Dict]) -> List[Dict]:
        """
        Extract insights from distribution analysis
        
        Args:
            distribution_results: List of distribution test results
        
        Returns:
            List of distribution insights
        """
        insights = []
        
        if not distribution_results:
            return insights
        
        for result in distribution_results:
            if result.get('test') == 'normality':
                col = result['column']
                normal = result.get('normal', True)
                statistic = result.get('statistic', 0)
                
                if not normal:
                    insight = {
                        "id": f"dist_insight_{len(self.insights) + 1}",
                        "type": "distribution",
                        "title": f"Non-Normal Distribution in {col}",
                        "what": f"The {col} variable does not follow a normal distribution (Shapiro-Wilk statistic: {statistic:.4f}).",
                        "why": self._explain_distribution_why(col, statistic),
                        "how": self._explain_distribution_how(col),
                        "recommendation": self._recommend_distribution_action(col),
                        "data_support": {
                            "statistic": statistic,
                            "p_value": result.get('p_value'),
                            "sample_size": result.get('sample_size')
                        }
                    }
                    insights.append(insight)
                    self.insights.append(insight)
        
        return insights
    
    def _explain_distribution_why(self, col: str, statistic: float) -> str:
        """Explain why distribution is not normal"""
        data = self.df[col].dropna()
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        explanation = f"The Shapiro-Wilk test indicates deviation from normality (statistic={statistic:.4f}). "
        
        if abs(skewness) > 1:
            skew_type = "right-skewed" if skewness > 0 else "left-skewed"
            explanation += f"The data is {skew_type} (skewness={skewness:.3f}), meaning the tail extends more to the {'right' if skewness > 0 else 'left'}. "
            explanation += f"This suggests the presence of {'many high outliers' if skewness > 0 else 'many low outliers'} or a natural constraint at the {'lower' if skewness > 0 else 'upper'} end."
        elif abs(kurtosis) > 3:
            kurt_type = "heavy-tailed" if kurtosis > 0 else "light-tailed"
            explanation += f"The distribution is {kurt_type} (kurtosis={kurtosis:.3f}), meaning it has {'more' if kurtosis > 0 else 'fewer'} extreme values than a normal distribution."
        else:
            explanation += "While the normality test failed, the distribution may still be approximately normal for practical purposes."
        
        return explanation
    
    def _explain_distribution_how(self, col: str) -> str:
        """Explain how to handle non-normal distribution"""
        return f"""How to handle non-normal {col}:

1. **Transformations**: Apply log, square root, or Box-Cox transformations to normalize the data if needed for parametric tests.

2. **Non-Parametric Tests**: Use statistical tests that don't assume normality (Mann-Whitney U, Kruskal-Wallis, Spearman correlation).

3. **Robust Methods**: Use robust statistical methods that are less sensitive to non-normality (median-based measures, trimmed means).

4. **Data Segmentation**: Consider if the data represents multiple subgroups with different distributions. Analyze segments separately.

5. **Monitoring**: Track changes in distribution over time to understand if patterns are stable or evolving.
"""
    
    def _recommend_distribution_action(self, col: str) -> str:
        """Recommend actions for non-normal distribution"""
        return f"**Action Required**: Before applying parametric statistical methods or machine learning algorithms that assume normality, either apply appropriate transformations or use non-parametric alternatives. Document the distribution characteristics in your data quality report."
    
    def extract_outlier_insights(self, outlier_results: List[Dict]) -> List[Dict]:
        """
        Extract insights from outlier analysis
        
        Args:
            outlier_results: List of outlier test results
        
        Returns:
            List of outlier insights
        """
        insights = []
        
        if not outlier_results:
            return insights
        
        for result in outlier_results:
            if result['outlier_count'] > 0:
                col = result['column']
                count = result['outlier_count']
                pct = result['outlier_percentage']
                severity = result.get('severity', 'moderate')
                
                insight = {
                    "id": f"out_insight_{len(self.insights) + 1}",
                    "type": "outlier",
                    "title": f"{severity.capitalize()} Outliers Detected in {col}",
                    "what": f"Found {count} outliers ({pct:.2f}% of data) in {col}, ranging from {result['outlier_min']:.2f} to {result['outlier_max']:.2f}.",
                    "why": self._explain_outliers_why(col, count, pct, severity),
                    "how": self._explain_outliers_how(col, severity),
                    "recommendation": self._recommend_outliers_action(col, pct, severity),
                    "data_support": {
                        "outlier_count": count,
                        "percentage": pct,
                        "range": [result['outlier_min'], result['outlier_max']],
                        "severity": severity
                    }
                }
                insights.append(insight)
                self.insights.append(insight)
        
        return insights
    
    def _explain_outliers_why(self, col: str, count: int, pct: float, severity: str) -> str:
        """Explain why outliers exist"""
        explanation = f"The {severity} outlier count ({count}, {pct:.2f}%) in {col} indicates unusual observations that deviate significantly from typical values. "
        
        if pct > 5:
            explanation += "The high percentage suggests this may not be random outliers but could represent: 1) A distinct subgroup or segment, 2) Data quality issues, 3) A multimodal distribution with different patterns, or 4) Special events or anomalies affecting many records."
        elif pct > 1:
            explanation += "This moderate outlier count is typical in real-world data and may represent: 1) Natural variation, 2) Edge cases, 3) Special conditions or rare events, or 4) Minor data entry errors."
        else:
            explanation += "These few outliers may represent: 1) Genuine extreme cases, 2) Data entry errors, 3) Measurement anomalies, or 4) Rare but valid occurrences."
        
        return explanation
    
    def _explain_outliers_how(self, col: str, severity: str) -> str:
        """Explain how to handle outliers"""
        return f"""How to handle outliers in {col}:

1. **Investigation**: Examine individual outlier records to understand their context. Are they errors or valid extremes?

2. **Context Analysis**: Check if outliers occur in specific time periods, geographic regions, or segments. This may reveal important patterns.

3. **Root Cause**: Identify the source - measurement error, data entry issues, system glitches, or genuine extreme events.

4. **Handling Strategy**:
   - **Remove** if clearly errors
   - **Winsorize** (cap at threshold) if many outliers
   - **Separate Analysis** if outliers form distinct segments
   - **Keep** if valid and informative

5. **Monitor**: Track outlier trends over time to detect data quality issues or changing patterns.

6. **Model Robustness**: Use models robust to outliers (tree-based, robust regression) if outliers are valid.
"""
    
    def _recommend_outliers_action(self, col: str, pct: float, severity: str) -> str:
        """Recommend actions for outliers"""
        if pct > 5:
            return f"**Urgent Action**: High outlier percentage requires immediate investigation. Examine individual outlier records, validate data quality, and consider if outliers represent a meaningful segment. Do not remove without investigation."
        elif pct > 1:
            return f"**Action Recommended**: Review outliers individually to understand their nature. Document findings and decide on handling strategy (remove, cap, or keep) based on investigation results."
        else:
            return f"**Monitor**: Few outliers are normal. Document them for reference and consider their impact on analyses. If using sensitive models, consider robust methods."
    
    def extract_statistical_test_insights(self, test_results: List[Dict]) -> List[Dict]:
        """
        Extract insights from statistical tests
        
        Args:
            test_results: List of statistical test results
        
        Returns:
            List of statistical test insights
        """
        insights = []
        
        if not test_results:
            return insights
        
        for result in test_results:
            if not result.get('significant', False):
                continue
            
            test_type = result.get('test')
            
            if test_type == 'group_differences':
                num_col = result['numerical_column']
                cat_col = result['categorical_column']
                
                insight = {
                    "id": f"group_insight_{len(self.insights) + 1}",
                    "type": "group_differences",
                    "title": f"Significant Differences in {num_col} Across {cat_col} Groups",
                    "what": f"There are statistically significant differences in {num_col} between different groups of {cat_col} (p={result['p_value']:.6f}).",
                    "why": f"The statistical test indicates that {cat_col} has a meaningful effect on {num_col}. Different categories within {cat_col} have systematically different values of {num_col}, suggesting that {cat_col} is an important factor influencing {num_col}.",
                    "how": f"""How to use this insight:

1. **Segment Analysis**: Analyze {num_col} separately for each {cat_col} group to understand specific patterns.

2. **Targeted Strategies**: Develop different strategies or interventions for each {cat_col} segment based on their {num_col} characteristics.

3. **Predictive Modeling**: Include {cat_col} as a key feature when predicting {num_col}.

4. **Prioritization**: Focus efforts on {cat_col} groups with the highest or lowest {num_col} values depending on your objectives.

5. **Root Cause Investigation**: Investigate why {cat_col} groups differ in {num_col} to understand underlying drivers.
""",
                    "recommendation": f"**Strategic Action**: Use {cat_col} for segmentation and targeting. Consider group-specific strategies rather than one-size-fits-all approaches. Include {cat_col} in all analyses involving {num_col}.",
                    "data_support": {
                        "p_value": result['p_value'],
                        "statistic": result['statistic'],
                        "group_means": result.get('group_means', []),
                        "group_sizes": result.get('group_sizes', [])
                    }
                }
                insights.append(insight)
                self.insights.append(insight)
            
            elif test_type == 'categorical_association':
                col1 = result['column1']
                col2 = result['column2']
                cramers_v = result.get('cramers_v', 0)
                
                insight = {
                    "id": f"assoc_insight_{len(self.insights) + 1}",
                    "type": "categorical_association",
                    "title": f"Association Between {col1} and {col2}",
                    "what": f"There is a statistically significant association between {col1} and {col2} (p={result['p_value']:.6f}, Cramér's V={cramers_v:.4f}).",
                    "why": f"These variables are not independent - knowledge of one provides information about the other. This suggests they may be influenced by common factors or causally related.",
                    "how": f"""How to use this association:

1. **Joint Analysis**: Always analyze {col1} and {col2} together rather than in isolation.

2. **Cross-Segmentation**: Use combinations of {col1} and {col2} for detailed market segmentation or user profiling.

3. **Decision Making**: Consider both variables simultaneously when making decisions or designing strategies.

4. **Prediction**: Use one variable to help predict or impute the other if data is missing.

5. **Investigation**: Examine specific combinations to understand which relationships are strongest.
""",
                    "recommendation": f"**Strategic Action**: Treat {col1} and {col2} as related variables. Design dashboards and reports showing their intersection. Avoid analyzing them independently as this may miss important patterns.",
                    "data_support": {
                        "chi2": result['chi2_statistic'],
                        "p_value": result['p_value'],
                        "cramers_v": cramers_v
                    }
                }
                insights.append(insight)
                self.insights.append(insight)
        
        return insights
    
    def extract_modeling_insights(self, model_results: Dict) -> List[Dict]:
        """
        Extract insights from modeling results
        
        Args:
            model_results: Dictionary containing model results
        
        Returns:
            List of modeling insights
        """
        insights = []
        
        # Extract from best model
        if not model_results or 'best_model' not in model_results:
            return insights
        
        best = model_results.get('best_model')
        if not best:
            return insights
        
        # Feature importance insights
        if 'feature_importance' in best:
            top_features = best['feature_importance'][:5]
            
            for i, feat in enumerate(top_features, 1):
                importance = feat.get('importance', 0)
                feature = feat.get('feature', 'unknown')
                
                if importance > 0.1:
                    insight = {
                        "id": f"feat_insight_{len(self.insights) + 1}",
                        "type": "feature_importance",
                        "title": f"Key Predictor: {feature}",
                        "what": f"{feature} is a top predictor in the model with importance of {importance:.4f}.",
                        "why": f"The model assigns high importance to {feature}, meaning it contains critical information for predicting the target. Changes in {feature} have the largest impact on predictions.",
                        "how": f"""How to use this insight:

1. **Monitor**: Track {feature} closely as it's a key driver of outcomes.

2. **Intervention**: If you want to influence outcomes, focus on changing {feature}.

3. **Data Collection**: Ensure {feature} is measured accurately and consistently.

4. **Communication**: Explain to stakeholders that {feature} is a primary factor affecting results.

5. **Further Analysis**: Investigate why {feature} is so important - what's the mechanism?
""",
                        "recommendation": f"**Priority Action**: Make {feature} a primary focus of your strategy. Allocate resources to monitor, improve, and optimize {feature}. Ensure high data quality for this variable.",
                        "data_support": {
                            "importance": importance,
                            "rank": i
                        }
                    }
                    insights.append(insight)
                    self.insights.append(insight)
        
        # Performance insights
        if 'test_accuracy' in best:
            accuracy = best['test_accuracy']
            
            insight = {
                "id": f"perf_insight_{len(self.insights) + 1}",
                "type": "model_performance",
                "title": f"Model Performance: {accuracy:.1%} Accuracy",
                "what": f"The best model achieves {accuracy:.1%} accuracy on test data.",
                "why": f"This accuracy level indicates {'excellent' if accuracy > 0.9 else 'good' if accuracy > 0.8 else 'moderate' if accuracy > 0.7 else 'limited'} predictive capability.",
                "how": f"Use this model for predictions with confidence appropriate to its accuracy level. {accuracy:.1%} of predictions are expected to be correct.",
                "recommendation": f"**Deployment**: {('Ready for production use' if accuracy > 0.8 else 'Consider improvement before deployment - try feature engineering, different algorithms, or more data' if accuracy > 0.7 else 'Requires significant improvement before use')}.",
                "data_support": {"accuracy": accuracy}
            }
            insights.append(insight)
            self.insights.append(insight)
        
        return insights
    
    def generate_all_insights(self, analysis_results: Dict) -> List[Dict]:
        """
        Generate insights from all analysis results
        
        Args:
            analysis_results: Dictionary containing all analysis results
        
        Returns:
            List of all insights
        """
        self.insights = []
        
        # Extract from each analysis type with null checks
        if analysis_results.get('correlations'):
            self.extract_correlation_insights(analysis_results['correlations'])
        
        if analysis_results.get('distributions'):
            self.extract_distribution_insights(analysis_results['distributions'])
        
        if analysis_results.get('outliers'):
            self.extract_outlier_insights(analysis_results['outliers'])
        
        if analysis_results.get('statistical_tests'):
            self.extract_statistical_test_insights(analysis_results['statistical_tests'])
        
        if analysis_results.get('modeling'):
            self.extract_modeling_insights(analysis_results['modeling'])
        
        # Ensure minimum number of insights
        if len(self.insights) < 50:
            self._generate_additional_insights()
        
        return self.insights
    
    def _generate_additional_insights(self):
        """Generate additional insights to meet minimum requirement"""
        # Generate data quality insights
        for col in self.df.columns:
            missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
            
            if missing_pct > 0:
                insight = {
                    "id": f"dq_insight_{len(self.insights) + 1}",
                    "type": "data_quality",
                    "title": f"Missing Data in {col}",
                    "what": f"{col} has {missing_pct:.2f}% missing values.",
                    "why": f"Missing data can bias analysis and reduce statistical power. The {missing_pct:.2f}% missing rate {'is substantial and requires attention' if missing_pct > 10 else 'is manageable but should be addressed' if missing_pct > 5 else 'is minimal'}.",
                    "how": f"Handle missing values by imputation (mean/median for numeric, mode for categorical), removal of incomplete records, or using algorithms that handle missing values natively.",
                    "recommendation": f"{'High Priority' if missing_pct > 10 else 'Medium Priority' if missing_pct > 5 else 'Low Priority'}: Implement data quality checks to prevent missing values and impute existing ones.",
                    "data_support": {"missing_percentage": missing_pct}
                }
                self.insights.append(insight)
        
        # Generate data type insights
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                unique_count = self.df[col].nunique()
                
                if unique_count > len(self.df) * 0.5:
                    insight = {
                        "id": f"card_insight_{len(self.insights) + 1}",
                        "type": "cardinality",
                        "title": f"High Cardinality in {col}",
                        "what": f"{col} has {unique_count} unique values ({unique_count/len(self.df)*100:.1f}% unique), indicating high cardinality.",
                        "why": "High cardinality means many distinct values, which can be challenging for analysis and modeling. This suggests {col} may be an ID, free-text field, or highly granular categorical variable.",
                        "how": "Consider grouping rare categories, using embedding techniques, or treating as a feature requiring special handling. If it's an ID field, exclude from modeling.",
                        "recommendation": "Review if high cardinality is expected. Consider category reduction, hashing, or removing if it's an identifier.",
                        "data_support": {"unique_values": unique_count, "uniqueness_percentage": round(unique_count/len(self.df)*100, 2)}
                    }
                    self.insights.append(insight)
    
    def format_insights_for_report(self) -> str:
        """
        Format all insights for a report
        
        Returns:
            Formatted Markdown string
        """
        if not self.insights:
            return "No insights generated."
        
        report = f"# Data Insights\n\n"
        report += f"**Total Insights Generated:** {len(self.insights)}\n\n"
        
        # Group by type
        by_type = {}
        for insight in self.insights:
            itype = insight['type']
            if itype not in by_type:
                by_type[itype] = []
            by_type[itype].append(insight)
        
        type_order = ['correlation', 'distribution', 'outlier', 'group_differences', 
                      'categorical_association', 'feature_importance', 'model_performance', 
                      'data_quality', 'cardinality']
        
        for itype in type_order:
            if itype not in by_type:
                continue
            
            type_title = itype.replace('_', ' ').title()
            report += f"## {type_title}\n\n"
            
            for insight in by_type[itype]:
                report += f"### {insight['title']}\n\n"
                report += f"**What:** {insight['what']}\n\n"
                report += f"**Why:** {insight['why']}\n\n"
                report += f"**How:** {insight['how']}\n\n"
                report += f"**Recommendation:** {insight['recommendation']}\n\n"
                report += "---\n\n"
        
        return report
    
    def save_insights(self, filepath: str):
        """
        Save insights to JSON file
        
        Args:
            filepath: Path to save insights
        """
        with open(filepath, 'w') as f:
            json.dump(self.insights, f, indent=2, cls=NumpyEncoder)

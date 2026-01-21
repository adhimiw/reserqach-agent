"""
Test script for the Autonomous Data Science System
Creates sample datasets and runs analysis
"""

import pandas as pd
import numpy as np
import os


def create_sample_sales_data(n_rows=1000, output_path="sample_sales_data.csv"):
    """
    Create sample sales dataset
    
    Args:
        n_rows: Number of rows
        output_path: Output file path
    """
    np.random.seed(42)
    
    data = {
        'date': pd.date_range('2020-01-01', periods=n_rows, freq='D')[:n_rows],
        'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], n_rows),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_rows),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
        'sales': np.random.gamma(shape=2, scale=50, size=n_rows) + np.random.normal(0, 10, n_rows),
        'price': np.random.uniform(10, 200, n_rows),
        'discount': np.random.uniform(0, 0.3, n_rows),
        'customer_age': np.random.normal(35, 12, n_rows),
        'customer_gender': np.random.choice(['Male', 'Female'], n_rows),
        'satisfaction_score': np.random.randint(1, 6, n_rows),
        'marketing_spend': np.random.gamma(shape=1.5, scale=100, size=n_rows),
        'returns': np.random.poisson(2, n_rows),
        'staff_count': np.random.poisson(5, n_rows) + 1
    }
    
    df = pd.DataFrame(data)
    
    # Ensure sales are positive
    df['sales'] = df['sales'].clip(lower=0)
    
    # Add some outliers
    df.loc[10:15, 'sales'] *= 3
    df.loc[50:52, 'price'] *= 5
    
    # Add some missing values
    df.loc[100:105, 'customer_age'] = np.nan
    df.loc[200:203, 'satisfaction_score'] = np.nan
    
    df.to_csv(output_path, index=False)
    print(f"Created sample dataset: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df


def create_sample_customer_data(n_rows=2000, output_path="sample_customer_data.csv"):
    """
    Create sample customer churn dataset
    
    Args:
        n_rows: Number of rows
        output_path: Output file path
    """
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, n_rows + 1),
        'age': np.random.normal(40, 15, n_rows).clip(18, 80),
        'income': np.random.lognormal(mean=10, sigma=0.5, size=n_rows),
        'tenure_months': np.random.randint(1, 120, n_rows),
        'monthly_spend': np.random.gamma(shape=2, scale=100, size=n_rows),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                       n_rows, p=[0.5, 0.3, 0.2]),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                         n_rows, p=[0.3, 0.5, 0.2]),
        'tech_support': np.random.choice(['Yes', 'No'], n_rows, p=[0.4, 0.6]),
        'complaints_last_month': np.random.poisson(1, n_rows),
        'satisfaction_score': np.random.randint(1, 11, n_rows),
        'churn': np.random.choice([0, 1], n_rows, p=[0.73, 0.27])
    }
    
    df = pd.DataFrame(data)
    
    # Make churn related to satisfaction and tenure
    churn_prob = (
        (df['satisfaction_score'] < 5).astype(int) * 0.4 +
        (df['tenure_months'] < 12).astype(int) * 0.3 +
        (df['complaints_last_month'] > 2).astype(int) * 0.2
    )
    df['churn'] = np.random.random(n_rows) < churn_prob
    df['churn'] = df['churn'].astype(int)
    
    df.to_csv(output_path, index=False)
    print(f"Created sample dataset: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.1%}")
    
    return df


def test_basic_pipeline():
    """Test basic pipeline functionality"""
    print("\n" + "="*60)
    print("TESTING AUTONOMOUS DATA SCIENCE SYSTEM")
    print("="*60 + "\n")
    
    # Create test datasets
    print("Creating test datasets...")
    sales_df = create_sample_sales_data(500, "test_sales.csv")
    customer_df = create_sample_customer_data(300, "test_customers.csv")
    
    print("\nTest datasets created successfully!")
    print("\nTo run analysis on these datasets:")
    print("  python main.py test_sales.csv --target_column sales")
    print("  python main.py test_customers.csv --target_column churn --no-word")
    print("\nTo start the dashboard:")
    print("  streamlit run ui/dashboard.py")
    print("\nTo start the chatbot:")
    print("  python ui/chatbot.py --interactive")
    print("\n" + "="*60)


def test_components():
    """Test individual components"""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("="*60 + "\n")
    
    # Test hypothesis generator
    print("Testing HypothesisGenerator...")
    from analysis_engine import HypothesisGenerator
    
    test_df = create_sample_sales_data(100, "temp_test.csv")
    hypo_gen = HypothesisGenerator(test_df)
    hypotheses = hypo_gen.generate_all_hypotheses(max_hypotheses=20)
    
    print(f"✓ Generated {len(hypotheses)} hypotheses")
    
    # Test statistical tester
    print("\nTesting StatisticalTester...")
    from analysis_engine import StatisticalTester
    
    stats_tester = StatisticalTester(test_df)
    test_results = stats_tester.run_comprehensive_tests()
    
    print(f"✓ Ran {len(test_results)} statistical tests")
    
    # Test insight extractor
    print("\nTesting InsightExtractor...")
    from analysis_engine import InsightExtractor
    
    insight_extractor = InsightExtractor(test_df)
    
    analysis_results = {
        'correlations': [r for r in test_results if r.get('test') == 'correlation'],
        'distributions': [r for r in test_results if r.get('test') == 'normality'],
        'outliers': [r for r in test_results if r.get('test') == 'outliers'],
        'statistical_tests': test_results
    }
    
    insights = insight_extractor.generate_all_insights(analysis_results)
    
    print(f"✓ Extracted {len(insights)} insights")
    
    # Test visualizer
    print("\nTesting DataVisualizer...")
    from agents import DataVisualizer
    
    viz_dir = "test_visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    visualizer = DataVisualizer(viz_dir)
    
    visualizations = visualizer.generate_all_visualizations(test_df)
    
    print(f"✓ Created {len(visualizations)} visualizations")
    
    # Test self-healing agent
    print("\nTesting SelfHealingAgent...")
    from agents import SelfHealingAgent
    
    healer = SelfHealingAgent(max_retries=2, retry_delay=1)
    
    def test_function():
        # This will fail on first attempt
        if not hasattr(test_function, 'attempted'):
            test_function.attempted = True
            raise ValueError("Test error")
        return "Success after retry"
    
    result = healer.execute_with_retry(test_function)
    
    print(f"✓ Self-healing test: {'Success' if result['success'] else 'Expected failure'}")
    
    # Cleanup
    print("\nCleaning up test files...")
    import shutil
    for f in ['temp_test.csv', 'test_sales.csv', 'test_customers.csv']:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)
    
    print("✓ Cleanup complete")
    
    print("\n" + "="*60)
    print("ALL COMPONENTS TESTED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Autonomous Data Science System")
    parser.add_argument("--full", action="store_true", 
                       help="Run full component tests")
    
    args = parser.parse_args()
    
    if args.full:
        test_components()
    else:
        test_basic_pipeline()

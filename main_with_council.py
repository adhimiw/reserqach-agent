"""
Autonomous Data Science System with LLM Council Integration - Main Entry Point
"""

import argparse
import sys
import os
import asyncio
from datetime import datetime

from config import Config
from workflow import AnalysisPipeline
from analysis_engine import EnhancedAnalysisPipeline, LLMCouncilAdapter


def setup_environment():
    """Initialize environment and configuration"""
    Config.ensure_output_dirs()
    print("="*60)
    print("AUTONOMOUS DATA SCIENCE SYSTEM WITH LLM COUNCIL")
    print("="*60)
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Log level: {Config.LOG_LEVEL}")
    print()


async def run_analysis_with_council(dataset_path: str, target_column: str = None,
                                 generate_word: bool = True,
                                 use_council: bool = True,
                                 council_backend: str = None,
                                 verbose: bool = False):
    """
    Run autonomous data analysis with LLM Council consensus
    
    Args:
        dataset_path: Path to dataset file
        target_column: Target variable for modeling (auto-detected if None)
        generate_word: Whether to generate Word document
        use_council: Whether to use LLM Council for consensus decisions
        council_backend: Path to llm-council backend
        verbose: Whether to show verbose output
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING DATASET: {os.path.basename(dataset_path)}")
    print(f"LLM Council: {'ENABLED' if use_council else 'DISABLED'}")
    print(f"{'='*60}\n")
    
    try:
        # Create enhanced pipeline with council integration
        pipeline = EnhancedAnalysisPipeline(
            dataset_path,
            use_council=use_council,
            council_backend_path=council_backend
        )
        
        # Run full analysis with council
        results = await pipeline.run_full_pipeline_with_council(
            target_column=target_column,
            generate_word=generate_word
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Dataset: {pipeline.base_pipeline.dataset_name}")
        print(f"Output directory: {pipeline.base_pipeline.output_dir}")
        print()
        print("Results Summary:")
        print(f"  - Hypotheses generated: {len(results.get('hypotheses', []))}")
        print(f"  - Statistical tests run: {len(results.get('statistical_tests', []))}")
        print(f"  - Models built: {len(results.get('models', {}).get('models', {}))}")
        print(f"  - Insights extracted: {len(results.get('insights', []))}")
        print(f"  - Visualizations created: {len(results.get('visualizations', {}))}")
        print()
        
        # Council-specific summary
        if use_council:
            print("LLM Council Usage:")
            print(f"  - Hypotheses via Council: {results.get('used_council_for_hypotheses', False)}")
            print(f"  - Insights via Council: {results.get('used_council_for_insights', False)}")
            print(f"  - Model Ranking via Council: {'Yes' if results.get('model_ranking') else 'No'}")
            print()
        
        # List generated files
        print("Generated Files:")
        print(f"  - Markdown Report: {os.path.join(pipeline.base_pipeline.output_dir, f'{pipeline.base_pipeline.dataset_name}_report.md')}")
        if generate_word:
            print(f"  - Word Document: {os.path.join(pipeline.base_pipeline.output_dir, f'{pipeline.base_pipeline.dataset_name}_report.docx')}")
        print(f"  - Visualizations: {os.path.join(pipeline.base_pipeline.output_dir, 'visualizations/')}")
        print(f"  - Analysis Logs: {os.path.join(pipeline.base_pipeline.output_dir, 'logs/')}")
        print()
        
        # Model ranking if available
        if results.get('model_ranking'):
            ranking = results['model_ranking']
            print("Model Ranking (Council Consensus):")
            if ranking.get('recommendation'):
                rec = ranking['recommendation']
                print(f"  - Recommended: {rec.get('model', 'unknown')}")
                print(f"    Justification: {rec.get('justification', 'N/A')}")
            if ranking.get('ranking'):
                for i, model_name in enumerate(ranking['ranking'][:3], 1):
                    print(f"  {i}. {model_name}")
            print()
        
        print(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return results
        
    except FileNotFoundError:
        print(f"\nERROR: Dataset file not found: {dataset_path}")
        print("Please check the file path and try again.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nERROR: Analysis failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def test_council_integration():
    """
    Test LLM Council integration with sample data
    """
    print("\n" + "="*60)
    print("TESTING LLM COUNCIL INTEGRATION")
    print("="*60 + "\n")
    
    # Create sample data
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 20, 50),
        'feature_b': np.random.normal(50, 15, 50),
        'feature_c': np.random.choice(['X', 'Y', 'Z'], 50),
        'target': np.random.normal(200, 40, 50)
    }
    df = pd.DataFrame(data)
    
    # Save temporary dataset
    test_path = "output/test_council_data.csv"
    os.makedirs("output", exist_ok=True)
    df.to_csv(test_path, index=False)
    
    print(f"Created test dataset: {test_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Test council adapter
    print("Testing LLMCouncilAdapter...")
    council_adapter = LLMCouncilAdapter()
    
    # Test hypothesis generation
    print("\n1. Testing hypothesis generation with council...")
    dataset_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict()
    }
    
    hypotheses = await council_adapter.generate_hypotheses_with_council(
        dataset_info,
        max_hypotheses=5
    )
    
    print(f"✓ Generated {len(hypotheses)} hypotheses")
    for h in hypotheses:
        print(f"  - {h.get('hypothesis', h.get('title', 'N/A'))}")
    
    # Test insight generation
    print("\n2. Testing insight generation with council...")
    
    analysis_results = {
        "statistical_tests": [
            {"test": "correlation", "interpretation": "Strong positive correlation found"},
            {"test": "normality", "interpretation": "Data is normally distributed"}
        ],
        "models": {
            "models": {
                "linear_regression": {
                    "metrics": {"model_type": "Linear Regression", "test_r2": 0.85}
                },
                "random_forest": {
                    "metrics": {"model_type": "Random Forest", "test_r2": 0.90}
                }
            }
        }
    }
    
    insights = await council_adapter.generate_insights_with_council(
        analysis_results,
        min_insights=3
    )
    
    print(f"✓ Generated {len(insights)} insights")
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight.get('title', insight.get('what', 'N/A')[:50])}")
    
    # Test model ranking
    print("\n3. Testing model ranking with council...")
    
    model_results = analysis_results["models"]["models"]
    ranking = await council_adapter.rank_models_with_council(model_results)
    
    print(f"✓ Ranked {len(model_results)} models")
    if ranking.get('recommendation'):
        rec = ranking['recommendation']
        print(f"  - Recommended: {rec.get('model', 'unknown')}")
        print(f"    Justification: {rec.get('justification', 'N/A')}")
    
    print("\n" + "="*60)
    print("COUNCIL INTEGRATION TEST COMPLETE")
    print("="*60)
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
        print(f"\nCleaned up test file: {test_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Autonomous Data Science System with LLM Council Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis (with LLM Council)
  python main.py data/sales_data.csv
  
  # With specific target (with LLM Council)
  python main.py data/customer_data.csv --target_column churn
  
  # Without LLM Council (use single LLM)
  python main.py data/metrics.json --no-council
  
  # Custom LLM Council backend
  python main.py data/financials.csv --council-backend /path/to/llm-council
  
  # Test council integration
  python main.py --test-council

  # Test council with verbose output
  python main.py --test-council --verbose
        """
    )
    
    parser.add_argument(
        "dataset",
        nargs='?',
        help="Path to dataset file (CSV, Excel, JSON, or Parquet). Use --test-council to skip."
    )
    
    parser.add_argument(
        "--target", "-t",
        dest="target_column",
        help="Target variable for predictive modeling (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--output", "-o",
        dest="output_dir",
        help="Output directory for analysis results (default: output/analyses/{dataset_name}/)"
    )
    
    parser.add_argument(
        "--no-word",
        action="store_true",
        help="Skip Word document generation (only generate Markdown)"
    )
    
    parser.add_argument(
        "--no-council",
        action="store_true",
        help="Disable LLM Council and use single LLM for decisions"
    )
    
    parser.add_argument(
        "--council-backend",
        dest="council_backend",
        help="Path to llm-council backend directory (default: /home/engine/project/llm-council/backend)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed error messages"
    )
    
    parser.add_argument(
        "--test-council",
        action="store_true",
        help="Run LLM Council integration tests only"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_environment()
    
    # Update config if output directory specified
    if args.output_dir:
        Config.ANALYSES_DIR = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run test or analysis
    if args.test_council:
        asyncio.run(test_council_integration())
    else:
        if not args.dataset:
            print("ERROR: Dataset path is required (unless using --test-council)")
            parser.print_help()
            sys.exit(1)
        
        asyncio.run(run_analysis_with_council(
            dataset_path=args.dataset,
            target_column=args.target_column,
            generate_word=not args.no_word,
            use_council=not args.no_council,
            council_backend=args.council_backend,
            verbose=args.verbose
        ))


if __name__ == "__main__":
    main()

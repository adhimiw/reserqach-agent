"""
Autonomous Data Science System - Main Entry Point
"""

import argparse
import sys
import os
from datetime import datetime

from config import Config
from workflow import AnalysisPipeline


def setup_environment():
    """Initialize environment and configuration"""
    Config.ensure_output_dirs()
    print("="*60)
    print("AUTONOMOUS DATA SCIENCE SYSTEM")
    print("="*60)
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Log level: {Config.LOG_LEVEL}")
    print()


def run_analysis(dataset_path: str, target_column: str = None, 
                generate_word: bool = True, verbose: bool = False,
                use_council: bool = False):
    """
    Run autonomous data analysis on a dataset
    
    Args:
        dataset_path: Path to dataset file
        target_column: Target variable for modeling (auto-detected if None)
        generate_word: Whether to generate Word document
        verbose: Whether to show verbose output
        use_council: Whether to use LLM Council for consensus decisions
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING DATASET: {os.path.basename(dataset_path)}")
    print(f"LLM Council: {'ENABLED' if use_council else 'DISABLED'}")
    print(f"{'='*60}\n")
    
    try:
        # Create pipeline (always use EnhancedAnalysisPipeline to benefit from agents)
        from analysis_engine import EnhancedAnalysisPipeline
        pipeline = EnhancedAnalysisPipeline(dataset_path, use_council=use_council)
        
        if use_council:
            # Import asyncio if using council
            import asyncio
            results = asyncio.run(pipeline.run_full_pipeline_with_council(
                target_column=target_column,
                generate_word=generate_word
            ))
        else:
            # Run full analysis (it will use agents if council is False)
            import asyncio
            results = asyncio.run(pipeline.run_full_pipeline_with_council(
                target_column=target_column,
                generate_word=generate_word
            ))
        
        # Print summary
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Dataset: {pipeline.dataset_name}")
        print(f"Output directory: {pipeline.output_dir}")
        print()
        print("Results Summary:")
        print(f"  - Hypotheses generated: {len(results.get('hypotheses', []))}")
        print(f"  - Statistical tests run: {len(results.get('statistical_tests', []))}")
        print(f"  - Models built: {len(results.get('models', {}).get('models', {}))}")
        print(f"  - Insights extracted: {len(results.get('insights', []))}")
        print(f"  - Visualizations created: {len(results.get('visualizations', {}))}")
        print()
        
        # Council usage summary
        if use_council:
            print("LLM Council Summary:")
            print(f"  - Hypotheses via Council: {results.get('used_council_for_hypotheses', False)}")
            print(f"  - Insights via Council: {results.get('used_council_for_insights', False)}")
            print(f"  - Model Ranking via Council: {'Yes' if results.get('model_ranking') else 'No'}")
            print()
        
        # List generated files
        print("Generated Files:")
        print(f"  - Markdown Report: {os.path.join(pipeline.output_dir, f'{pipeline.dataset_name}_report.md')}")
        if generate_word:
            print(f"  - Word Document: {os.path.join(pipeline.output_dir, f'{pipeline.dataset_name}_report.docx')}")
        print(f"  - Visualizations: {os.path.join(pipeline.output_dir, 'visualizations/')}")
        print(f"  - Analysis Logs: {os.path.join(pipeline.output_dir, 'logs/')}")
        print()
        
        print(f"Analysis completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Autonomous Data Science System - Automatically analyze datasets and generate insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data/sales_data.csv
  python main.py data/customer_data.xlsx --target_column purchase_amount
  python main.py data/metrics.json --no-word --verbose
  python main.py data/financials.csv --output ./my_analysis
        """
    )
    
    parser.add_argument(
        "dataset",
        help="Path to dataset file (CSV, Excel, JSON, or Parquet)"
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
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed error messages"
    )
    
    parser.add_argument(
        "--use-council",
        action="store_true",
        help="Enable LLM Council for multi-agent consensus decisions"
    )
    
    parser.add_argument(
        "--council-backend",
        dest="council_backend",
        help="Path to llm-council backend directory (default: /home/engine/project/llm-council/backend)"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_environment()
    
    # Update config if output directory specified
    if args.output_dir:
        Config.ANALYSES_DIR = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis
    run_analysis(
        dataset_path=args.dataset,
        target_column=args.target_column,
        generate_word=not args.no_word,
        verbose=args.verbose,
        use_council=args.use_council
    )


if __name__ == "__main__":
    main()

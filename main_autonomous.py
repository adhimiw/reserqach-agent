"""
Autonomous Data Science System - Main Entry Point
Uses autonomous code generation and execution
"""

import argparse
import sys
import os
from datetime import datetime

from config import Config
from workflow.autonomous_pipeline import AutonomousAnalysisPipeline


def setup_environment():
    """Initialize environment and configuration"""
    Config.ensure_output_dirs()
    print("="*60)
    print("AUTONOMOUS DATA SCIENCE SYSTEM")
    print("Agent writes and executes its own code")
    print("="*60)
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Log level: {Config.LOG_LEVEL}")
    print()


def run_autonomous_analysis(dataset_path: str,
                           target_column: str = None,
                           task_type: str = "auto",
                           create_notebook: bool = True,
                           verbose: bool = False):
    """
    Run autonomous data analysis with code generation
    
    Args:
        dataset_path: Path to dataset file
        target_column: Target variable for modeling (optional)
        task_type: Type of ML task (classification, regression, auto)
        create_notebook: Whether to generate Jupyter notebook
        verbose: Whether to show verbose output
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING DATASET: {os.path.basename(dataset_path)}")
    print(f"Mode: AUTONOMOUS CODE GENERATION AND EXECUTION")
    print(f"{'='*60}\n")
    
    if target_column:
        print(f"Target Column: {target_column}")
        print(f"Task Type: {task_type}")
    print()
    
    try:
        # Create autonomous pipeline
        pipeline = AutonomousAnalysisPipeline(dataset_path)
        
        # Run full autonomous analysis
        results = pipeline.run_full_autonomous_pipeline(
            target_column=target_column,
            task_type=task_type,
            create_notebook=create_notebook
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("AUTONOMOUS ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Dataset: {pipeline.dataset_name}")
        print(f"Output directory: {pipeline.output_dir}")
        print(f"Run ID: {pipeline.run_id}")
        print()
        print("Results Summary:")
        print(f"  - Dataset shape: {results['dataset_info']['shape']}")
        print(f"  - Columns analyzed: {len(results['dataset_info']['columns'])}")
        print(f"  - Autonomous tasks executed: {len(results['autonomous_analysis'])}")
        print(f"  - Code files generated: {len(pipeline.generated_code_paths)}")
        print(f"  - Execution time: {results.get('duration_seconds', 0):.1f} seconds")
        print()
        
        # List generated code files
        print("Generated Code Files:")
        for i, code_path in enumerate(pipeline.generated_code_paths, 1):
            filename = os.path.basename(code_path)
            print(f"  {i}. {filename}")
        print()
        
        # List generated notebook
        if 'notebook_path' in results:
            print(f"Jupyter Notebook: {results['notebook_path']}")
            print()
        
        # List analysis tasks
        print("Autonomous Tasks Executed:")
        for task in results['autonomous_analysis']:
            task_name = task.get('task', 'unknown')
            timestamp = task.get('timestamp', '')
            status = "SUCCESS" if task.get('result', {}).get('success') else "FAILED"
            print(f"  - {task_name}: {status}")
        print()
        
        # Output directory structure
        print("Output Directory Structure:")
        print(f"  - Generated code: {os.path.join(pipeline.output_dir, 'generated_code')}")
        print(f"  - Notebooks: {os.path.join(pipeline.output_dir, 'notebooks')}")
        print(f"  - Visualizations: {os.path.join(pipeline.output_dir, 'visualizations')}")
        print(f"  - Logs: {os.path.join(pipeline.output_dir, 'logs')}")
        print()
        
        print("Generated Reports:")
        print(f"  - Results JSON: {os.path.join(pipeline.output_dir, 'autonomous_analysis_results.json')}")
        print(f"  - Code Manifest: {os.path.join(pipeline.output_dir, 'generated_code', 'code_manifest.json')}")
        print(f"  - Execution Log: {os.path.join(pipeline.output_dir, 'logs', 'autonomous_execution_log.json')}")
        print()
        
        print(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("="*60)
        print("THE AGENT GENERATED AND EXECUTED ITS OWN CODE!")
        print(f"All code saved as .py files in: {os.path.join(pipeline.output_dir, 'generated_code')}")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR DURING ANALYSIS")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Autonomous Data Science System - Code Generation and Execution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic exploratory analysis
  python main_autonomous.py data.csv

  # With target column for modeling
  python main_autonomous.py data.csv --target target_column

  # Specify task type
  python main_autonomous.py data.csv --target target_column --task-type classification

  # Without notebook generation
  python main_autonomous.py data.csv --no-notebook
        """
    )
    
    parser.add_argument('dataset', help='Path to dataset file (CSV, Excel, JSON, Parquet)')
    parser.add_argument('--target', help='Target column for modeling (optional)')
    parser.add_argument('--task-type', choices=['classification', 'regression', 'auto'],
                       default='auto', help='Type of ML task (default: auto)')
    parser.add_argument('--no-notebook', action='store_true',
                       help='Do not generate Jupyter notebook')
    parser.add_argument('--verbose', action='store_true',
                       help='Show verbose output')
    
    args = parser.parse_args()
    
    # Check if dataset file exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Setup and run analysis
    setup_environment()
    
    run_autonomous_analysis(
        dataset_path=args.dataset,
        target_column=args.target,
        task_type=args.task_type,
        create_notebook=not args.no_notebook,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()

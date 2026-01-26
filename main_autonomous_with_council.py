"""
Autonomous Data Science System with LLM Council AND Autonomous Code Execution
Combines LLM Council consensus with agent-generated code
"""

import argparse
import sys
import os
import asyncio
from datetime import datetime

from config import Config
from workflow.autonomous_pipeline import AutonomousAnalysisPipeline
from analysis_engine import LLMCouncilAdapter


def setup_environment():
    """Initialize environment and configuration"""
    Config.ensure_output_dirs()
    print("="*60)
    print("AUTONOMOUS DATA SCIENCE SYSTEM")
    print("LLM Council + Autonomous Code Generation")
    print("="*60)
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Log level: {Config.LOG_LEVEL}")
    print()


async def run_autonomous_with_council(dataset_path: str,
                                       target_column: str = None,
                                       use_council: bool = True,
                                       task_type: str = "auto",
                                       create_notebook: bool = True):
    """
    Run autonomous analysis with LLM Council guidance
    
    Args:
        dataset_path: Path to dataset file
        target_column: Target variable for modeling
        use_council: Whether to use LLM Council for guidance
        task_type: Type of ML task
        create_notebook: Whether to generate Jupyter notebook
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING DATASET: {os.path.basename(dataset_path)}")
    print(f"Mode: AUTONOMOUS CODE GENERATION")
    print(f"LLM Council: {'ENABLED' if use_council else 'DISABLED'}")
    print(f"{'='*60}\n")
    
    try:
        # Create autonomous pipeline
        pipeline = AutonomousAnalysisPipeline(dataset_path)
        
        # Use Council to guide the analysis if enabled
        council_guidance = {}
        if use_council:
            print("🤖 Getting LLM Council guidance for analysis strategy...\n")
            council_adapter = LLMCouncilAdapter()
            
            # Get dataset info
            import pandas as pd
            df = pd.read_csv(dataset_path)
            dataset_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict()
            }
            
            # Get analysis strategy from Council
            strategy_prompt = f"""Dataset: {dataset_path}
Shape: {dataset_info['shape']}
Columns: {dataset_info['columns']}

Provide an analysis strategy for this dataset. Include:
1. Key features to explore
2. Potential target columns for modeling
3. Recommended analysis tasks
4. Suggested visualizations
5. Potential hypotheses to test
"""
            
            strategy = await council_adapter.run_council_consensus(
                strategy_prompt,
                task="analysis_strategy"
            )
            
            council_guidance['strategy'] = strategy
            print(f"📋 Council Strategy Received:")
            print(f"   {strategy.get('recommendation', 'Using default strategy')[:100]}...\n")
        
        # Setup environment
        print("🔧 Setting up isolated Python environment...\n")
        pipeline.setup_environment()
        
        # Step 1: Exploratory Analysis
        print("🔍 Step 1: Autonomous Exploratory Analysis")
        print("   Agent generating and executing EDA code...\n")
        pipeline.load_and_analyze_dataset()
        print("   ✓ Exploratory analysis complete\n")
        
        # Step 2: Feature Engineering
        print("⚙️  Step 2: Autonomous Feature Engineering")
        print("   Agent generating and executing feature engineering code...\n")
        pipeline.autonomous_feature_engineering(target_column=target_column)
        print("   ✓ Feature engineering complete\n")
        
        # Step 3: Model Building (if target provided)
        if target_column:
            print(f"🤖 Step 3: Autonomous Model Building")
            print(f"   Agent generating and executing model code for target: {target_column}...\n")
            
            # Get Council guidance for model selection if enabled
            if use_council:
                model_prompt = f"""Target column: {target_column}
Task type: {task_type}

Recommend appropriate machine learning models for this target.
Consider:
1. Model types (classification/regression)
2. Complexity trade-offs
3. Interpretability requirements
"""
                model_guidance = await council_adapter.run_council_consensus(
                    model_prompt,
                    task="model_selection"
                )
                council_guidance['model_guidance'] = model_guidance
                print(f"   📋 Council Model Recommendation: {model_guidance.get('recommendation', 'Using default models')[:80]}...\n")
            
            pipeline.autonomous_model_building(target_column, task_type=task_type)
            print("   ✓ Model building complete\n")
        
        # Step 4: Generate Notebook
        if create_notebook:
            print("📓 Step 4: Generating Jupyter Notebook")
            print("   Compiling all generated code into notebook...\n")
            notebook_path = pipeline.generate_analysis_notebook()
            print(f"   ✓ Notebook created: {notebook_path}\n")
        
        # Save manifests
        pipeline.autonomous_coder.save_coding_manifest()
        pipeline.code_executor.save_code_manifest()
        
        # Print summary
        print(f"\n{'='*60}")
        print("AUTONOMOUS ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Dataset: {pipeline.dataset_name}")
        print(f"Output directory: {pipeline.output_dir}")
        print(f"Run ID: {pipeline.run_id}")
        print()
        print("Results Summary:")
        print(f"  - Dataset shape: {pipeline.results['dataset_info']['shape']}")
        print(f"  - Columns analyzed: {len(pipeline.results['dataset_info']['columns'])}")
        print(f"  - Autonomous tasks executed: {len(pipeline.results['autonomous_analysis'])}")
        print(f"  - Code files generated: {len(pipeline.generated_code_paths)}")
        print()
        
        if use_council:
            print("LLM Council Usage:")
            print("  ✓ Provided analysis strategy guidance")
            if target_column:
                print("  ✓ Provided model selection recommendations")
            print()
        
        print("Generated Code Files:")
        for i, code_path in enumerate(pipeline.generated_code_paths, 1):
            filename = os.path.basename(code_path)
            print(f"  {i}. {filename}")
        print()
        
        if 'notebook_path' in pipeline.results:
            print(f"Jupyter Notebook:")
            print(f"  📓 {pipeline.results['notebook_path']}")
            print()
        
        print("Output Directory Structure:")
        print(f"  📂 Generated Code: {os.path.join(pipeline.output_dir, 'generated_code')}")
        print(f"  📂 Notebooks: {os.path.join(pipeline.output_dir, 'notebooks')}")
        print(f"  📂 Visualizations: {os.path.join(pipeline.output_dir, 'visualizations')}")
        print(f"  📂 Logs: {os.path.join(pipeline.output_dir, 'logs')}")
        print()
        
        print("Key Reports:")
        print(f"  📊 Results: {os.path.join(pipeline.output_dir, 'autonomous_analysis_results.json')}")
        print(f"  📋 Code Manifest: {os.path.join(pipeline.output_dir, 'generated_code', 'code_manifest.json')}")
        print(f"  📝 Execution Log: {os.path.join(pipeline.output_dir, 'logs', 'autonomous_execution_log.json')}")
        print()
        
        print("="*60)
        print("🎉 THE AGENT GENERATED AND EXECUTED ITS OWN CODE!")
        print(f"💾 All code saved in: {os.path.join(pipeline.output_dir, 'generated_code')}")
        print(f"🐍 Python environment: {pipeline.results.get('environment_path', 'N/A')}")
        print("="*60)
        print()
        
        print(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Instructions for viewing
        print("="*60)
        print("NEXT STEPS")
        print("="*60)
        print(f"1. View generated Python code:")
        print(f"   ls {os.path.join(pipeline.output_dir, 'generated_code')}")
        print()
        print(f"2. Run any generated code file:")
        print(f"   python {pipeline.generated_code_paths[0]}")
        print()
        if 'notebook_path' in pipeline.results:
            print(f"3. Open Jupyter notebook:")
            print(f"   jupyter notebook {pipeline.results['notebook_path']}")
            print()
        print(f"4. View results:")
        print(f"   cat {os.path.join(pipeline.output_dir, 'autonomous_analysis_results.json')}")
        print("="*60)
        
        return pipeline.results
        
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
        description='Autonomous Data Science System - LLM Council + Code Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic autonomous analysis with Council
  python main_autonomous_with_council.py data.csv

  # With target column for modeling
  python main_autonomous_with_council.py data.csv --target target_column

  # Without Council guidance
  python main_autonomous_with_council.py data.csv --target column --no-council

  # Specify task type
  python main_autonomous_with_council.py data.csv --target column --task-type classification
        """
    )
    
    parser.add_argument('dataset', help='Path to dataset file (CSV, Excel, JSON, Parquet)')
    parser.add_argument('--target', help='Target column for modeling (optional)')
    parser.add_argument('--task-type', choices=['classification', 'regression', 'auto'],
                       default='auto', help='Type of ML task (default: auto)')
    parser.add_argument('--no-council', action='store_true',
                       help='Disable LLM Council guidance')
    parser.add_argument('--no-notebook', action='store_true',
                       help='Do not generate Jupyter notebook')
    
    args = parser.parse_args()
    
    # Check if dataset file exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Setup and run analysis
    setup_environment()
    
    asyncio.run(run_autonomous_with_council(
        dataset_path=args.dataset,
        target_column=args.target,
        use_council=not args.no_council,
        task_type=args.task_type,
        create_notebook=not args.no_notebook
    ))


if __name__ == '__main__':
    main()

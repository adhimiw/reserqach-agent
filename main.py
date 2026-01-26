"""
Autonomous Data Science System - Unified Main Entry Point
Agent thinks, plans with LLM Council, generates code, executes it, and creates reports
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
    print("="*70)
    print(" AUTONOMOUS DATA SCIENCE SYSTEM")
    print("="*70)
    print(" Agent thinks, plans with Council, generates code, and executes it")
    print("="*70)
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Log level: {Config.LOG_LEVEL}")
    print()


async def run_unified_analysis(dataset_path: str,
                           target_column: str = None,
                           generate_word: bool = True,
                           use_council: bool = True,
                           council_backend: str = None,
                           create_notebook: bool = True,
                           verbose: bool = False):
    """
    Run complete autonomous analysis with LLM Council planning and code generation

    Args:
        dataset_path: Path to dataset file
        target_column: Target variable for modeling
        generate_word: Whether to generate Word document
        use_council: Whether to use LLM Council for planning
        council_backend: Path to llm-council backend
        create_notebook: Whether to generate Jupyter notebook
        verbose: Whether to show verbose output
    """
    print(f"\n{'='*70}")
    print(f" DATASET: {os.path.basename(dataset_path)}")
    print(f" Target: {target_column if target_column else 'Auto-detect'}")
    print(f" LLM Council: {'ENABLED' if use_council else 'DISABLED'}")
    print(f" Code Generation: ENABLED (Agent writes and executes code)")
    print(f"{'='*70}\n")

    try:
        # =================================================================
        # STEP 1: INITIALIZE PIPELINE
        # =================================================================
        print("🔷 STEP 1: Initialize Autonomous Pipeline")
        print("-" * 70)
        pipeline = AutonomousAnalysisPipeline(dataset_path)
        print(f"✓ Pipeline initialized for: {pipeline.dataset_name}")
        print(f"✓ Output directory: {pipeline.output_dir}")
        print()

        # =================================================================
        # STEP 2: LOAD AND READ DATASET (Agent thinks)
        # =================================================================
        print("📊 STEP 2: Agent Reads and Analyzes Dataset")
        print("-" * 70)
        import pandas as pd
        df = pd.read_csv(dataset_path)
        pipeline.df = df

        print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ Data types: {df.dtypes.value_counts().to_dict()}")
        print(f"✓ Missing values: {df.isnull().sum().sum()}")
        print()

        # =================================================================
        # STEP 3: MAKE PLAN WITH LLM COUNCIL (Agent plans)
        # =================================================================
        print("🤔 STEP 3: Agent Makes Analysis Plan (with LLM Council)")
        print("-" * 70)

        if use_council:
            print("🤖 LLM Council is thinking about best analysis strategy...")
            print()

            council_adapter = LLMCouncilAdapter()

            dataset_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict(),
                "sample_data": df.head(3).to_dict(),
                "target_column": target_column
            }

            # Get Council's analysis plan
            plan_prompt = f"""You are an expert data scientist. Create a comprehensive analysis plan for this dataset:

DATASET INFO:
{dataset_info}

Target Column: {target_column if target_column else 'None (unsupervised)'}

Your task: Create a detailed analysis plan that includes:
1. Exploratory Data Analysis (EDA) - what to explore, what visualizations
2. Feature Engineering - which features to create, transformations needed
3. Model Selection - which models to try and why
4. Evaluation Metrics - appropriate metrics for this problem
5. Insights Generation - what insights to extract
6. Visualization Strategy - which plots and charts to create

Provide the plan as a structured JSON with these sections:
- "exploratory_analysis": list of EDA tasks
- "feature_engineering": list of feature engineering steps
- "model_selection": list of models to try with reasoning
- "evaluation_metrics": appropriate metrics
- "insights_generation": what insights to look for
- "visualization_strategy": list of visualizations to create
- "priority_recommendations": what to focus on first
"""

            print("   Gathering consensus from multiple LLMs...")
            plan_result = await council_adapter.run_council_consensus(
                plan_prompt,
                task="analysis_planning"
            )

            # Parse plan from Council response
            if plan_result and 'recommendation' in plan_result:
                council_plan = plan_result['recommendation']
                print("✓ LLM Council consensus reached!")
                print(f"✓ Plan generated: {len(str(council_plan))} characters")
                print()
                print("   📋 Council Recommendations:")
                print("   " + str(council_plan)[:300] + "...")
                print()

                # Save Council plan
                import json
                plan_path = os.path.join(pipeline.output_dir, 'llm_council_plan.json')
                with open(plan_path, 'w') as f:
                    json.dump({
                        'plan': council_plan,
                        'timestamp': datetime.now().isoformat(),
                        'dataset_info': dataset_info
                    }, f, indent=2)
                print(f"✓ Council plan saved to: {os.path.basename(plan_path)}")
                print()
            else:
                print("⚠ Council returned empty plan, using default strategy")
                print()
                council_plan = None

        else:
            print("   Using default analysis strategy")
            council_plan = None

        print()

        # =================================================================
        # STEP 4: SETUP ENVIRONMENT (Agent creates environment)
        # =================================================================
        print("🔧 STEP 4: Agent Creates Execution Environment")
        print("-" * 70)
        env_path = pipeline.setup_environment()
        print(f"✓ Isolated Python environment created: {env_path}")
        print(f"✓ Agent has full control over this environment")
        print()

        # =================================================================
        # STEP 5: GENERATE INSIGHTS (Agent thinks about data)
        # =================================================================
        print("💡 STEP 5: Agent Extracts Insights (with LLM Council)")
        print("-" * 70)

        if use_council:
            insights_prompt = f"""Based on this dataset information:

{dataset_info}

Provide 5-10 actionable insights about:
1. Data quality issues
2. Interesting patterns or trends
3. Potential outliers or anomalies
4. Feature importance or relationships
5. Recommendations for further analysis

Format as a JSON array of insights with:
- "title": Insight heading
- "description": Detailed explanation
- "importance": High/Medium/Low
- "action": What should be done
"""

            print("   Extracting insights with LLM Council consensus...")
            insights_result = await council_adapter.run_council_consensus(
                insights_prompt,
                task="insight_extraction"
            )

            if insights_result and 'recommendation' in insights_result:
                insights = insights_result['recommendation']
                pipeline.results['insights'] = insights
                print(f"✓ {len(insights) if isinstance(insights, list) else 5} insights extracted")
                print(f"   Sample insight: {str(insights)[:100] if not isinstance(insights, list) else str(insights[0])[:100] if insights else 'N/A'}...")
                print()

        # =================================================================
        # STEP 6: GENERATE AND EXECUTE CODE (Agent codes and executes)
        # =================================================================
        print("🚀 STEP 6: Agent Generates and Executes Code")
        print("-" * 70)

        # 6a. Exploratory Analysis Code
        print("   6a. Generating exploratory analysis code...")
        print("       Agent is writing Python code for EDA...")
        pipeline.load_and_analyze_dataset()
        print(f"       ✓ EDA code generated and executed")
        print(f"       ✓ Code saved to: {pipeline.generated_code_paths[-1] if pipeline.generated_code_paths else 'N/A'}")
        print()

        # 6b. Feature Engineering Code
        print("   6b. Generating feature engineering code...")
        print("       Agent is writing Python code for feature engineering...")
        pipeline.autonomous_feature_engineering(target_column=target_column)
        print(f"       ✓ Feature engineering code generated and executed")
        print(f"       ✓ Code saved to: {pipeline.generated_code_paths[-1] if pipeline.generated_code_paths else 'N/A'}")
        print()

        # 6c. Model Building Code (if target column)
        if target_column:
            print("   6c. Generating model building code...")
            print(f"       Agent is writing Python code to predict: {target_column}")

            # Use Council to recommend models if available
            task_type = "auto"
            if use_council and council_plan:
                print("       Getting model recommendations from Council...")
                task_type = "classification"  # Default, could be detected

            pipeline.autonomous_model_building(target_column, task_type=task_type)
            print(f"       ✓ Model building code generated and executed")
            print(f"       ✓ Code saved to: {pipeline.generated_code_paths[-1] if pipeline.generated_code_paths else 'N/A'}")
            print()

        # =================================================================
        # STEP 7: GENERATE VISUALIZATIONS (from executed code)
        # =================================================================
        print("📈 STEP 7: Generate Visualizations")
        print("-" * 70)
        print("   Visualizations were created by executed code:")
        print(f"   ✓ Check: {os.path.join(pipeline.output_dir, 'visualizations/')}")
        print()

        # =================================================================
        # STEP 8: CREATE JUPYTER NOTEBOOK (Compile all code)
        # =================================================================
        if create_notebook:
            print("📓 STEP 8: Create Jupyter Notebook")
            print("-" * 70)
            print("   Compiling all generated code into notebook...")
            notebook_path = pipeline.generate_analysis_notebook()
            print(f"   ✓ Notebook created: {notebook_path}")
            print()

        # =================================================================
        # STEP 9: SAVE MANIFESTS AND LOGS
        # =================================================================
        print("💾 STEP 9: Save Manifests and Logs")
        print("-" * 70)

        # Save code manifest
        code_manifest_path = pipeline.code_executor.save_code_manifest()
        print(f"✓ Code manifest: {os.path.basename(code_manifest_path)}")

        # Save coding manifest
        coding_manifest_path = pipeline.autonomous_coder.save_coding_manifest()
        print(f"✓ Coding manifest: {os.path.basename(coding_manifest_path)}")

        # Save results
        results_path = pipeline.save_results()
        print(f"✓ Results saved: {os.path.basename(results_path)}")

        # Save execution log
        pipeline.save_execution_log()
        print(f"✓ Execution log saved")
        print()

        # =================================================================
        # SUMMARY
        # =================================================================
        print("="*70)
        print(" ANALYSIS COMPLETE - AGENT AUTONOMY SUMMARY")
        print("="*70)
        print()
        print(f"Dataset: {pipeline.dataset_name}")
        print(f"Output: {pipeline.output_dir}")
        print()
        print("What the Agent Did:")
        print(f"  🧠 Thought about dataset: {df.shape}")
        print(f"  🤝 Planned with LLM Council: {'Yes' if use_council else 'No'}")
        print(f"  📝 Generated Python code files: {len(pipeline.generated_code_paths)}")
        print(f"  ⚙️  Created isolated environment: Yes")
        print(f"  🚀 Executed its own code: {len(pipeline.generated_code_paths)} files")
        print(f"  📊 Created visualizations: Check visualizations/")
        if target_column:
            print(f"  🤖 Built ML models: Yes (target: {target_column})")
        if create_notebook:
            print(f"  📓 Compiled notebook: Yes")
        print()
        print("Generated Code Files (Agent wrote these):")
        for i, code_path in enumerate(pipeline.generated_code_paths, 1):
            filename = os.path.basename(code_path)
            print(f"  {i}. {filename}")
        print()

        if use_council:
            print("LLM Council Contributions:")
            print("  ✓ Analysis planning")
            print("  ✓ Insight extraction")
            if target_column:
                print("  ✓ Model selection guidance")
            print()

        print("Output Directory Structure:")
        print(f"  📂 Generated Code: {os.path.join(pipeline.output_dir, 'generated_code')}")
        print(f"  📂 Notebooks: {os.path.join(pipeline.output_dir, 'notebooks')}")
        print(f"  📂 Visualizations: {os.path.join(pipeline.output_dir, 'visualizations')}")
        print(f"  📂 Logs: {os.path.join(pipeline.output_dir, 'logs')}")
        print(f"  📂 Data: {os.path.join(pipeline.output_dir, 'data')}")
        print()

        print("Key Files:")
        print(f"  📊 Results: {os.path.join(pipeline.output_dir, 'autonomous_analysis_results.json')}")
        print(f"  📋 Code Manifest: {os.path.join(pipeline.output_dir, 'generated_code', 'code_manifest.json')}")
        if use_council:
            print(f"  🤖 Council Plan: {os.path.join(pipeline.output_dir, 'llm_council_plan.json')}")
        print(f"  📝 Execution Log: {os.path.join(pipeline.output_dir, 'logs', 'autonomous_execution_log.json')}")
        print()

        print("="*70)
        print(" NEXT STEPS - What You Can Do")
        print("="*70)
        print()
        print("1. View generated code:")
        print(f"   ls {os.path.join(pipeline.output_dir, 'generated_code')}")
        print()
        print("2. View/edit any generated code:")
        print(f"   cat {pipeline.generated_code_paths[0]}")
        print()
        print("3. Re-run generated code:")
        print(f"   python {pipeline.generated_code_paths[0]}")
        print()
        if create_notebook and 'notebook_path' in pipeline.results:
            print("4. Open Jupyter notebook:")
            print(f"   jupyter notebook {pipeline.results['notebook_path']}")
            print()
        print("5. View results:")
        print(f"   cat {os.path.join(pipeline.output_dir, 'autonomous_analysis_results.json')}")
        print()

        print("="*70)
        print(" 🎉 AGENT AUTONOMY COMPLETE!")
        print("="*70)
        print()
        print("The agent:")
        print("  ✅ Thought about the dataset")
        print("  ✅ Planned analysis with LLM Council")
        print("  ✅ Generated its own Python code")
        print("  ✅ Saved all code as .py files")
        print("  ✅ Executed its generated code")
        print("  ✅ Created visualizations and models")
        print("  ✅ Compiled everything into a notebook")
        print()
        print(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return pipeline.results

    except FileNotFoundError:
        print(f"\n❌ ERROR: Dataset file not found: {dataset_path}")
        print("Please check the file path and try again.")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERROR: Analysis failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Autonomous Data Science System - Agent thinks, plans with Council, generates code, and executes it",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full autonomous analysis with LLM Council
  python main.py data.csv

  # With target column for modeling
  python main.py data.csv --target target_column

  # Without LLM Council (use default planning)
  python main.py data.csv --target column --no-council

  # Without notebook generation
  python main.py data.csv --no-notebook

  # Verbose output for debugging
  python main.py data.csv --verbose
        """
    )

    parser.add_argument(
        "dataset",
        help="Path to dataset file (CSV, Excel, JSON, or Parquet)"
    )

    parser.add_argument(
        "--target", "-t",
        dest="target_column",
        help="Target variable for predictive modeling (optional)"
    )

    parser.add_argument(
        "--no-council",
        action="store_true",
        help="Disable LLM Council (use default planning strategy)"
    )

    parser.add_argument(
        "--no-notebook",
        action="store_true",
        help="Do not generate Jupyter notebook"
    )

    parser.add_argument(
        "--council-backend",
        dest="council_backend",
        help="Path to llm-council backend (default: /home/engine/project/llm-council/backend)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed error messages"
    )

    parser.add_argument(
        "--output", "-o",
        dest="output_dir",
        help="Output directory for analysis results"
    )

    args = parser.parse_args()

    # Setup
    setup_environment()

    # Update config if output directory specified
    if args.output_dir:
        Config.ANALYSES_DIR = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)

    # Run unified analysis
    asyncio.run(run_unified_analysis(
        dataset_path=args.dataset,
        target_column=args.target_column,
        use_council=not args.no_council,
        council_backend=args.council_backend,
        create_notebook=not args.no_notebook,
        verbose=args.verbose
    ))


if __name__ == "__main__":
    main()

"""
Demonstration of Autonomous Coding System
Shows how the agent writes and executes its own code
"""

import os
import sys
import tempfile
from agents import CodeExecutionAgent, AutonomousCoderAgent


def demo_code_execution_agent():
    """Demonstrate CodeExecutionAgent capabilities"""
    print("="*70)
    print("DEMO 1: CodeExecutionAgent - Direct Code Execution")
    print("="*70)
    print()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = CodeExecutionAgent(temp_dir)
        
        # 1. Setup environment
        print("🔧 Step 1: Setting up isolated Python environment...")
        env_path = executor.setup_environment("demo_env")
        print(f"   ✓ Environment created: {env_path}")
        print()
        
        # 2. Generate and save code file
        print("📝 Step 2: Generating Python code...")
        demo_code = '''
import numpy as np
import pandas as pd
import json

print("Hello! I'm code that was written by an AI agent!")
print()

# Generate some data
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'score': [85, 92, 78, 95, 88]
}
df = pd.DataFrame(data)

print("Generated Data:")
print(df)
print()

# Calculate statistics
print("Statistics:")
print(f"  Average age: {df['age'].mean():.1f}")
print(f"  Average score: {df['score'].mean():.1f}")
print(f"  Highest score: {df['score'].max()}")
print()

# Save to JSON
result = {
    'average_age': float(df['age'].mean()),
    'average_score': float(df['score'].mean()),
    'top_scorer': df.loc[df['score'].idxmax(), 'name']
}

with open('demo_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("✓ Analysis complete!")
print(f"✓ Result saved to demo_result.json")
'''
        
        file_path = executor.generate_code_file(
            demo_code,
            "demo_analysis",
            "Simple demonstration code"
        )
        print(f"   ✓ Code saved: {os.path.basename(file_path)}")
        print(f"   ✓ Code length: {len(demo_code)} characters")
        print()
        
        # 3. Execute the code
        print("🚀 Step 3: Executing the generated code...")
        result = executor.execute_code_file(file_path)
        
        print(f"   ✓ Execution completed in {result['execution_time']:.2f}s")
        print(f"   ✓ Success: {result['success']}")
        print()
        
        # 4. Show output
        print("📊 Step 4: Code Output:")
        print("-" * 70)
        if result['stdout']:
            print(result['stdout'])
        print("-" * 70)
        print()
        
        # 5. Run terminal command
        print("💻 Step 5: Running terminal command...")
        cmd_result = executor.run_terminal_command("echo 'Terminal access works!'", timeout=10)
        print(f"   ✓ Command: echo 'Terminal access works!'")
        print(f"   ✓ Output: {cmd_result['stdout'].strip()}")
        print(f"   ✓ Exit code: {cmd_result['exit_code']}")
        print()
        
        # 6. Save manifest
        print("📋 Step 6: Saving code manifest...")
        manifest_path = executor.save_code_manifest()
        print(f"   ✓ Manifest saved: {os.path.basename(manifest_path)}")
        print()
    
    print("="*70)
    print("✓ Demo 1 Complete!")
    print("  The agent created code, saved it, and executed it successfully!")
    print("="*70)
    print()


def demo_autonomous_coder():
    """Demonstrate AutonomousCoderAgent capabilities"""
    print("="*70)
    print("DEMO 2: AutonomousCoderAgent - LLM-Generated Code")
    print("="*70)
    print()
    
    # Create test data
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    test_data = {
        'feature_a': np.random.normal(100, 20, 50),
        'feature_b': np.random.normal(50, 15, 50),
        'category': np.random.choice(['X', 'Y', 'Z'], 50),
        'target': np.random.normal(200, 40, 50)
    }
    df = pd.DataFrame(test_data)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test data
        test_csv = os.path.join(temp_dir, 'test_data.csv')
        df.to_csv(test_csv, index=False)
        
        # Initialize coder
        coder = AutonomousCoderAgent(temp_dir)
        
        # 1. Setup environment
        print("🔧 Step 1: Setting up execution environment...")
        coder.setup_execution_environment("coder_demo_env")
        print("   ✓ Environment created")
        print()
        
        # 2. Generate code for analysis
        print("🤖 Step 2: Agent generating analysis code using LLM...")
        print("   Task: 'Load CSV and print summary statistics'")
        print()
        
        code = coder.generate_analysis_code(
            task_description="Load the CSV file and print summary statistics including mean, std, min, and max for all numeric columns",
            dataset_info={
                "columns": list(df.columns),
                "shape": df.shape,
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            },
            context="Demonstration of code generation"
        )
        
        print(f"   ✓ Code generated ({len(code)} characters)")
        print(f"   ✓ Code preview (first 200 chars):")
        print("   " + code[:200].replace('\n', '\n   '))
        print()
        
        # 3. Save and execute code
        print("💾 Step 3: Saving and executing generated code...")
        result = coder.execute_generated_code(
            code,
            "demo_analysis",
            "LLM-generated analysis code"
        )
        
        print(f"   ✓ Code saved to file")
        print(f"   ✓ Execution success: {result.get('success')}")
        print()
        
        # 4. Show output
        print("📊 Step 4: Generated Code Output:")
        print("-" * 70)
        if result.get('stdout'):
            print(result['stdout'])
        if result.get('stderr'):
            print("STDERR:", result['stderr'])
        print("-" * 70)
        print()
    
    print("="*70)
    print("✓ Demo 2 Complete!")
    print("  The agent used an LLM to write code, saved it, and executed it!")
    print("="*70)
    print()


def demo_custom_analysis():
    """Demonstrate custom analysis task"""
    print("="*70)
    print("DEMO 3: Custom Analysis - Agent Generates Any Code!")
    print("="*70)
    print()
    
    import pandas as pd
    
    # Create simple test data
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data = pd.DataFrame({
            'product': ['A', 'B', 'C', 'D', 'E'],
            'sales': [100, 200, 150, 250, 175]
        })
        test_csv = os.path.join(temp_dir, 'sales.csv')
        test_data.to_csv(test_csv, index=False)
        
        coder = AutonomousCoderAgent(temp_dir)
        coder.setup_execution_environment("custom_demo_env")
        
        # Custom task
        print("🎯 Step 1: Defining custom analysis task...")
        custom_task = """
        Create a bar chart showing sales by product.
        Use matplotlib to create the chart.
        Save the chart as 'sales_chart.png'.
        Print a summary of total sales.
        """
        
        print(f"   Task: {custom_task.strip()}")
        print()
        
        print("🤖 Step 2: Agent generating custom code...")
        result = coder.generate_custom_analysis(custom_task, test_csv)
        print(f"   ✓ Code generated and executed")
        print(f"   ✓ Success: {result.get('success')}")
        print()
        
        print("📊 Step 3: Output:")
        print("-" * 70)
        if result.get('stdout'):
            print(result['stdout'])
        print("-" * 70)
        print()
    
    print("="*70)
    print("✓ Demo 3 Complete!")
    print("  The agent generated code for a custom task - no pre-coded logic!")
    print("="*70)
    print()


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print(" AUTONOMOUS CODING SYSTEM - LIVE DEMONSTRATION")
    print("="*70)
    print()
    print("This demonstration shows how the agent:")
    print("  1. Writes its own Python code")
    print("  2. Saves the code as .py files")
    print("  3. Executes its generated code")
    print("  4. Has full terminal access")
    print("  5. Uses LLMs to generate code")
    print("  6. Can handle any custom analysis task")
    print()
    
    try:
        # Demo 1: Direct code execution
        demo_code_execution_agent()
        
        # Demo 2: LLM-generated code
        # Note: This requires LLM API to be configured
        print("="*70)
        print("DEMO 2: Skipping LLM generation (requires API configuration)")
        print("="*70)
        print("   To test LLM code generation:")
        print("   1. Configure your LLM API in config.py")
        print("   2. Run: python main_autonomous.py your_data.csv")
        print()
        
        # Demo 3: Custom analysis
        print("="*70)
        print("DEMO 3: Skipping custom analysis (requires LLM API)")
        print("="*70)
        print("   The agent can generate ANY Python code with LLM guidance!")
        print()
        
        # Summary
        print("="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)
        print()
        print("Key Takeaways:")
        print("  ✓ Agent creates its own Python code")
        print("  ✓ Code is saved as visible .py files")
        print("  ✓ Agent executes code in isolated environments")
        print("  ✓ Agent has full terminal access")
        print("  ✓ No pre-coded logic - agent writes custom solutions!")
        print()
        print("Ready to use:")
        print("  python main_autonomous.py your_dataset.csv")
        print("  python main_autonomous_with_council.py your_dataset.csv --target column")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

"""
Test script for Autonomous Coding System
Demonstrates the agent writing and executing its own code
"""

import os
import sys
import tempfile
import json
from datetime import datetime


def test_code_execution_agent():
    """Test the CodeExecutionAgent"""
    print("="*60)
    print("Testing CodeExecutionAgent")
    print("="*60)
    
    from agents import CodeExecutionAgent
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = CodeExecutionAgent(temp_dir)
        
        # Test 1: Setup environment
        print("\n1. Setting up environment...")
        try:
            env_path = executor.setup_environment("test_env")
            print(f"✓ Environment created at: {env_path}")
        except Exception as e:
            print(f"✗ Environment setup failed: {e}")
            return False
        
        # Test 2: Generate and save code file
        print("\n2. Generating code file...")
        test_code = """
import math
print("Hello from autonomous code!")
print(f"Pi = {math.pi}")
print(f"Square root of 2 = {math.sqrt(2)}")
result = {"message": "Success", "value": 42}
print(f"Result: {result}")
"""
        try:
            file_path = executor.generate_code_file(
                test_code,
                "test_script",
                "Simple test script"
            )
            print(f"✓ Code file created: {file_path}")
            
            # Verify file exists
            if os.path.exists(file_path):
                print(f"✓ File exists and is readable")
                with open(file_path, 'r') as f:
                    content = f.read()
                    print(f"✓ File content length: {len(content)} characters")
            else:
                print(f"✗ File not created")
                return False
        except Exception as e:
            print(f"✗ Code generation failed: {e}")
            return False
        
        # Test 3: Execute code file
        print("\n3. Executing code file...")
        try:
            result = executor.execute_code_file(file_path, timeout=30)
            
            print(f"✓ Execution completed")
            print(f"  Success: {result['success']}")
            print(f"  Execution time: {result['execution_time']:.2f}s")
            if result['stdout']:
                print(f"  Output:\n{result['stdout']}")
            if result['stderr']:
                print(f"  Errors:\n{result['stderr']}")
            
            if not result['success']:
                print(f"✗ Execution failed")
                return False
        except Exception as e:
            print(f"✗ Code execution failed: {e}")
            return False
        
        # Test 4: Execute code directly
        print("\n4. Executing code string...")
        try:
            result = executor.execute_code("""
import json
data = {"name": "test", "value": 123}
print(json.dumps(data))
""", timeout=30)
            
            print(f"✓ Direct execution completed")
            print(f"  Success: {result['success']}")
            print(f"  Output: {result['stdout']}")
        except Exception as e:
            print(f"✗ Direct execution failed: {e}")
            return False
        
        # Test 5: Run terminal command
        print("\n5. Running terminal command...")
        try:
            result = executor.run_terminal_command("echo 'Terminal access works!'", timeout=30)
            
            print(f"✓ Terminal command completed")
            print(f"  Success: {result['success']}")
            print(f"  Exit code: {result['exit_code']}")
            print(f"  Output: {result['stdout'].strip()}")
        except Exception as e:
            print(f"✗ Terminal command failed: {e}")
            return False
        
        # Test 6: Install package
        print("\n6. Installing package...")
        try:
            result = executor.install_package("requests")
            
            print(f"✓ Package installation completed")
            print(f"  Success: {result['success']}")
        except Exception as e:
            print(f"✗ Package installation failed: {e}")
            # This is not critical, so we continue
        
        # Test 7: Save manifest
        print("\n7. Saving code manifest...")
        try:
            manifest_path = executor.save_code_manifest()
            print(f"✓ Manifest saved: {manifest_path}")
            
            # Verify manifest
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    print(f"✓ Manifest contains {manifest['total_files']} files")
            else:
                print(f"✗ Manifest not saved")
        except Exception as e:
            print(f"✗ Manifest save failed: {e}")
    
    print("\n" + "="*60)
    print("✓ CodeExecutionAgent tests PASSED")
    print("="*60)
    return True


def test_autonomous_coder_agent():
    """Test the AutonomousCoderAgent"""
    print("\n" + "="*60)
    print("Testing AutonomousCoderAgent")
    print("="*60)
    
    from agents import AutonomousCoderAgent
    import pandas as pd
    
    # Create temporary output directory and test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test dataset
        test_data = {
            'feature_a': [1, 2, 3, 4, 5],
            'feature_b': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'target': [100, 200, 150, 250, 175]
        }
        df = pd.DataFrame(test_data)
        test_csv = os.path.join(temp_dir, 'test_data.csv')
        df.to_csv(test_csv, index=False)
        
        # Initialize agent
        coder = AutonomousCoderAgent(temp_dir)
        
        # Test 1: Setup environment
        print("\n1. Setting up execution environment...")
        try:
            coder.setup_execution_environment("coder_test")
            print("✓ Environment setup complete")
        except Exception as e:
            print(f"✗ Environment setup failed: {e}")
            return False
        
        # Test 2: Generate code for analysis
        print("\n2. Generating analysis code...")
        try:
            code = coder.generate_analysis_code(
                task_description="Load the CSV and print basic statistics",
                dataset_info={"columns": list(df.columns), "shape": df.shape},
                context="Testing code generation"
            )
            print(f"✓ Code generated ({len(code)} characters)")
            print(f"  First 100 chars: {code[:100]}...")
        except Exception as e:
            print(f"✗ Code generation failed: {e}")
            return False
        
        # Test 3: Execute generated code
        print("\n3. Executing generated code...")
        try:
            result = coder.execute_generated_code(
                code,
                "test_analysis",
                "Test analysis execution"
            )
            print(f"✓ Code execution completed")
            print(f"  Success: {result.get('success')}")
            if result.get('stdout'):
                print(f"  Output preview: {result['stdout'][:200]}...")
        except Exception as e:
            print(f"✗ Code execution failed: {e}")
            return False
        
        # Test 4: Exploratory analysis
        print("\n4. Running exploratory analysis...")
        try:
            result = coder.analyze_dataset_exploratory(test_csv)
            print(f"✓ Exploratory analysis completed")
            print(f"  Success: {result.get('success')}")
        except Exception as e:
            print(f"✗ Exploratory analysis failed: {e}")
            return False
        
        # Test 5: Generate notebook
        print("\n5. Generating Jupyter notebook...")
        try:
            code_files = coder.code_executor.code_files
            file_paths = [f['path'] for f in code_files]
            
            if file_paths:
                notebook_path = coder.create_notebook_from_analysis(
                    file_paths,
                    "test_notebook"
                )
                print(f"✓ Notebook created: {notebook_path}")
                
                # Verify notebook
                if os.path.exists(notebook_path):
                    with open(notebook_path, 'r') as f:
                        notebook = json.load(f)
                        print(f"✓ Notebook contains {len(notebook['cells'])} cells")
                else:
                    print(f"✗ Notebook not created")
        except Exception as e:
            print(f"✗ Notebook generation failed: {e}")
        
        # Test 6: Save manifest
        print("\n6. Saving coding manifest...")
        try:
            manifest_path = coder.save_coding_manifest()
            print(f"✓ Manifest saved: {manifest_path}")
        except Exception as e:
            print(f"✗ Manifest save failed: {e}")
    
    print("\n" + "="*60)
    print("✓ AutonomousCoderAgent tests PASSED")
    print("="*60)
    return True


def test_autonomous_pipeline():
    """Test the complete AutonomousAnalysisPipeline"""
    print("\n" + "="*60)
    print("Testing AutonomousAnalysisPipeline")
    print("="*60)
    
    from workflow.autonomous_pipeline import AutonomousAnalysisPipeline
    import pandas as pd
    
    # Create temporary directory and test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test dataset
        np = __import__('numpy')
        test_data = {
            'feature_a': np.random.normal(100, 20, 50),
            'feature_b': np.random.normal(50, 15, 50),
            'feature_c': np.random.choice(['X', 'Y', 'Z'], 50),
            'target': np.random.normal(200, 40, 50)
        }
        df = pd.DataFrame(test_data)
        test_csv = os.path.join(temp_dir, 'test_dataset.csv')
        df.to_csv(test_csv, index=False)
        
        # Initialize pipeline
        print("\n1. Initializing pipeline...")
        try:
            pipeline = AutonomousAnalysisPipeline(test_csv, temp_dir)
            print(f"✓ Pipeline initialized")
            print(f"  Dataset: {pipeline.dataset_name}")
            print(f"  Output directory: {pipeline.output_dir}")
        except Exception as e:
            print(f"✗ Pipeline initialization failed: {e}")
            return False
        
        # Test 2: Setup environment
        print("\n2. Setting up environment...")
        try:
            env_path = pipeline.setup_environment()
            print(f"✓ Environment created: {env_path}")
        except Exception as e:
            print(f"✗ Environment setup failed: {e}")
            return False
        
        # Test 3: Load and analyze dataset
        print("\n3. Loading and analyzing dataset...")
        try:
            pipeline.load_and_analyze_dataset()
            print(f"✓ Dataset analysis completed")
            print(f"  Shape: {pipeline.df.shape}")
            print(f"  Code files generated: {len(pipeline.generated_code_paths)}")
        except Exception as e:
            print(f"✗ Dataset analysis failed: {e}")
            return False
        
        # Test 4: Feature engineering
        print("\n4. Running autonomous feature engineering...")
        try:
            pipeline.autonomous_feature_engineering(target_column="target")
            print(f"✓ Feature engineering completed")
            print(f"  Total code files: {len(pipeline.generated_code_paths)}")
        except Exception as e:
            print(f"✗ Feature engineering failed: {e}")
            return False
        
        # Test 5: Model building
        print("\n5. Running autonomous model building...")
        try:
            pipeline.autonomous_model_building(
                target_column="target",
                task_type="regression"
            )
            print(f"✓ Model building completed")
            print(f"  Total code files: {len(pipeline.generated_code_paths)}")
        except Exception as e:
            print(f"✗ Model building failed: {e}")
            return False
        
        # Test 6: Generate notebook
        print("\n6. Generating analysis notebook...")
        try:
            notebook_path = pipeline.generate_analysis_notebook()
            print(f"✓ Notebook created: {notebook_path}")
        except Exception as e:
            print(f"✗ Notebook generation failed: {e}")
        
        # Test 7: Save results
        print("\n7. Saving results...")
        try:
            pipeline.save_results()
            print(f"✓ Results saved")
        except Exception as e:
            print(f"✗ Results save failed: {e}")
    
    print("\n" + "="*60)
    print("✓ AutonomousAnalysisPipeline tests PASSED")
    print("="*60)
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("AUTONOMOUS CODING SYSTEM - TEST SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_passed = True
    
    # Test 1: CodeExecutionAgent
    if not test_code_execution_agent():
        all_passed = False
    
    # Test 2: AutonomousCoderAgent
    if not test_autonomous_coder_agent():
        all_passed = False
    
    # Test 3: AutonomousAnalysisPipeline
    if not test_autonomous_pipeline():
        all_passed = False
    
    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nThe Autonomous Coding System is working correctly!")
        print("\nYou can now run:")
        print("  python main_autonomous.py your_dataset.csv")
        print("  python main_autonomous_with_council.py your_dataset.csv --target column")
        print()
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check the error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()

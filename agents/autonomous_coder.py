"""
Autonomous Coding Agent
Generates and executes its own Python code for data analysis tasks
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .code_executor import CodeExecutionAgent
from llm_api_integration import LLMAPIClient

logger = logging.getLogger("AutonomousCoderAgent")


class AutonomousCoderAgent:
    """Agent that autonomously writes and executes Python code for data analysis"""
    
    def __init__(self, output_dir: str, code_executor: CodeExecutionAgent = None,
                 llm_client: LLMAPIClient = None):
        """
        Initialize the autonomous coding agent
        
        Args:
            output_dir: Directory for outputs
            code_executor: CodeExecutorAgent instance (creates one if None)
            llm_client: LLM client for code generation (creates one if None)
        """
        self.output_dir = output_dir
        self.code_executor = code_executor or CodeExecutionAgent(output_dir)
        self.llm_client = llm_client or LLMAPIClient()
        
        self.code_history = []
        self.analysis_tasks = []
        
        logger.info("AutonomousCoderAgent initialized")
    
    def setup_execution_environment(self, env_name: str = "analysis_env") -> str:
        """Set up the execution environment"""
        return self.code_executor.setup_environment(env_name)
    
    def generate_analysis_code(self, task_description: str,
                              dataset_info: Dict[str, Any] = None,
                              context: str = "") -> str:
        """
        Generate Python code for a specific analysis task using LLM
        
        Args:
            task_description: Description of the analysis task
            dataset_info: Information about the dataset (columns, types, etc.)
            context: Additional context for code generation
            
        Returns:
            Generated Python code
        """
        prompt = f"""You are an expert data scientist. Generate Python code to accomplish the following task:

TASK: {task_description}

{'DATASET INFO:' + json.dumps(dataset_info, indent=2) if dataset_info else ''}

{'CONTEXT: ' + context if context else ''}

Requirements:
1. Write complete, executable Python code
2. Include necessary imports (pandas, numpy, matplotlib, seaborn, scikit-learn)
3. Add comments explaining key steps
4. Handle errors gracefully with try-except blocks
5. Print clear output and results
6. Save any visualizations to the current directory
7. Return data in a structured format when possible
8. Use best practices and proper code organization

Generate ONLY the Python code, no explanations or markdown.
"""
        
        try:
            logger.info(f"Generating code for task: {task_description[:50]}...")
            
            # Use LLM to generate code
            response = self.llm_client.generate_response(prompt, model="gpt-4")
            
            # Extract code from response
            code = self._extract_code_from_response(response)
            
            # Record in history
            self.code_history.append({
                "task": task_description,
                "code": code,
                "generated_at": datetime.now().isoformat()
            })
            
            logger.info(f"Code generated successfully ({len(code)} characters)")
            return code
            
        except Exception as e:
            logger.error(f"Failed to generate code: {e}")
            raise
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Look for code blocks
        if "```python" in response:
            code_start = response.find("```python") + 9
            code_end = response.find("```", code_start)
            if code_end != -1:
                return response[code_start:code_end].strip()
        elif "```" in response:
            code_start = response.find("```") + 3
            code_end = response.find("```", code_start)
            if code_end != -1:
                return response[code_start:code_end].strip()
        
        # If no code blocks, return the whole response
        return response.strip()
    
    def execute_generated_code(self, code: str, task_name: str,
                              description: str = "") -> Dict[str, Any]:
        """
        Generate a code file and execute it
        
        Args:
            code: Python code to execute
            task_name: Name for the task (used for filename)
            description: Description of what the code does
            
        Returns:
            Execution result
        """
        # Save code to file
        file_path = self.code_executor.generate_code_file(
            code, task_name, description
        )
        
        # Execute the file
        execution_result = self.code_executor.execute_code_file(file_path)
        
        # Record task
        self.analysis_tasks.append({
            "task_name": task_name,
            "file_path": file_path,
            "description": description,
            "execution_result": execution_result,
            "executed_at": datetime.now().isoformat()
        })
        
        return execution_result
    
    def analyze_dataset_exploratory(self, dataset_path: str) -> Dict[str, Any]:
        """
        Perform exploratory data analysis using autonomous code generation
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            Analysis results
        """
        task = """Perform comprehensive exploratory data analysis:
1. Load the dataset from the given path
2. Display basic info (shape, columns, data types)
3. Calculate summary statistics
4. Check for missing values
5. Identify numeric and categorical columns
6. Create visualizations:
   - Histogram distributions for numeric columns
   - Correlation heatmap
   - Count plots for categorical columns
7. Save a summary report as JSON

Dataset path: {dataset_path}
"""
        
        code = self.generate_analysis_code(
            task.format(dataset_path=dataset_path),
            context="Exploratory Data Analysis"
        )
        
        result = self.execute_generated_code(
            code, "exploratory_analysis",
            "Exploratory data analysis with visualizations"
        )
        
        return result
    
    def perform_feature_engineering(self, dataset_path: str,
                                   target_column: str = None) -> Dict[str, Any]:
        """
        Generate and execute feature engineering code
        
        Args:
            dataset_path: Path to the dataset
            target_column: Target variable for supervised feature engineering
            
        Returns:
            Execution results
        """
        task = f"""Perform advanced feature engineering:
1. Load the dataset from {dataset_path}
2. Identify feature types (numeric, categorical, datetime)
3. Handle missing values appropriately
4. Create new features:
   - Interaction features between numeric columns
   - Polynomial features for important numeric columns
   - Aggregation features if applicable
   - Encoding for categorical variables (one-hot, label encoding)
5. Scale/normalize numeric features
6. Handle outliers
7. Save the engineered dataset to a new CSV file
8. Print a summary of created features

{'Target column: ' + target_column if target_column else 'Unsupervised feature engineering'}
"""
        
        code = self.generate_analysis_code(
            task,
            context="Feature Engineering for Machine Learning"
        )
        
        result = self.execute_generated_code(
            code, "feature_engineering",
            "Advanced feature engineering pipeline"
        )
        
        return result
    
    def build_models_autonomously(self, dataset_path: str,
                                  target_column: str,
                                  task_type: str = "auto") -> Dict[str, Any]:
        """
        Generate and execute model building code
        
        Args:
            dataset_path: Path to the dataset
            target_column: Target variable
            task_type: Type of task (classification, regression, auto)
            
        Returns:
            Model building results
        """
        task = f"""Build and evaluate machine learning models:
1. Load dataset from {dataset_path}
2. Target column: {target_column}
3. Determine task type (classification or regression)
4. Split data into train/test sets
5. Train multiple models:
   - For classification: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
   - For regression: Linear Regression, Random Forest, Gradient Boosting, XGBoost
6. Evaluate using appropriate metrics (accuracy, precision, recall, F1, R², MSE, MAE)
7. Perform cross-validation
8. Feature importance analysis
9. Save the best model using joblib
10. Create visualizations:
    - Feature importance plot
    - Confusion matrix (for classification)
    - ROC curve (for classification)
    - Actual vs Predicted (for regression)
11. Save evaluation metrics to JSON

Print detailed results for each model.
"""
        
        code = self.generate_analysis_code(
            task,
            context=f"Model Building - {task_type}"
        )
        
        result = self.execute_generated_code(
            code, "model_building",
            f"Model training and evaluation for {target_column}"
        )
        
        return result
    
    def generate_custom_analysis(self, custom_task: str,
                                dataset_path: str = None) -> Dict[str, Any]:
        """
        Generate code for custom analysis task
        
        Args:
            custom_task: Custom task description
            dataset_path: Optional dataset path
            
        Returns:
            Execution results
        """
        task = custom_task
        if dataset_path:
            task += f"\nDataset path: {dataset_path}"
        
        code = self.generate_analysis_code(
            task,
            context="Custom Analysis Task"
        )
        
        result = self.execute_generated_code(
            code, "custom_analysis",
            f"Custom analysis: {custom_task[:50]}..."
        )
        
        return result
    
    def create_notebook_from_analysis(self, code_files: List[str],
                                     notebook_name: str = "analysis") -> str:
        """
        Create a Jupyter notebook from generated code files
        
        Args:
            code_files: List of code file paths to include
            notebook_name: Name for the notebook
            
        Returns:
            Path to generated notebook
        """
        code_cells = []
        
        for code_file in code_files:
            try:
                with open(code_file, 'r') as f:
                    content = f.read()
                    # Skip header comments
                    lines = content.split('\n')
                    start_idx = 0
                    for i, line in enumerate(lines):
                        if line.strip() and not line.strip().startswith('"""'):
                            start_idx = i
                            break
                    
                    code = '\n'.join(lines[start_idx:])
                    code_cells.append(code)
            except Exception as e:
                logger.warning(f"Failed to read {code_file}: {e}")
        
        notebook_path = self.code_executor.generate_notebook(
            code_cells, filename=notebook_name
        )
        
        return notebook_path
    
    def save_coding_manifest(self) -> str:
        """Save manifest of all coding activities"""
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "total_code_files": len(self.code_history),
            "total_tasks_executed": len(self.analysis_tasks),
            "code_history": self.code_history,
            "analysis_tasks": self.analysis_tasks,
            "code_executor_manifest_path": self.code_executor.save_code_manifest()
        }
        
        manifest_path = os.path.join(self.output_dir, 'autonomous_coding_manifest.json')
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Autonomous coding manifest saved to {manifest_path}")
        return manifest_path

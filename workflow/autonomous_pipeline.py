"""
Autonomous Data Science Pipeline
Uses autonomous code generation and execution for all analysis tasks
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from config import Config
from agents import AutonomousCoderAgent, CodeExecutionAgent

logger = logging.getLogger("AutonomousPipeline")


class AutonomousAnalysisPipeline:
    """Pipeline that autonomously generates and executes code for data analysis"""
    
    def __init__(self, dataset_path: str, output_dir: str = None):
        """
        Initialize autonomous analysis pipeline
        
        Args:
            dataset_path: Path to dataset file
            output_dir: Output directory for results
        """
        self.dataset_path = dataset_path
        self.dataset_name = os.path.basename(dataset_path).split('.')[0]
        self.output_dir = output_dir or Config.get_analysis_output_dir(self.dataset_name)
        self.run_id = os.path.basename(self.output_dir)
        
        # Create output directories
        for subdir in ['data', 'generated_code', 'visualizations', 'insights', 'logs', 'notebooks']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
        # Initialize autonomous coder
        self.code_executor = CodeExecutionAgent(self.output_dir)
        self.autonomous_coder = AutonomousCoderAgent(self.output_dir, self.code_executor)
        
        # Results storage
        self.results = {
            "dataset_info": {},
            "autonomous_analysis": [],
            "generated_code_files": [],
            "execution_results": [],
            "models": {},
            "insights": [],
            "visualizations": {},
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "autonomous_mode": True
        }
        
        self.execution_log = []
        self.generated_code_paths = []
        
        logger.info(f"AutonomousPipeline initialized for dataset: {self.dataset_name}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup_environment(self, env_name: str = "analysis_env") -> str:
        """
        Set up isolated execution environment
        
        Args:
            env_name: Name for the environment
            
        Returns:
            Path to the environment
        """
        logger.info(f"Setting up execution environment: {env_name}")
        env_path = self.autonomous_coder.setup_execution_environment(env_name)
        self.results["environment_path"] = env_path
        return env_path
    
    def load_and_analyze_dataset(self) -> Dict[str, Any]:
        """
        Load dataset and perform autonomous exploratory analysis
        
        Returns:
            Analysis results
        """
        self._log_step("Load and analyze dataset", "start")
        
        try:
            # Load dataset info first
            self.df = pd.read_csv(self.dataset_path)
            
            # Store dataset info
            self.results['dataset_info'] = {
                "shape": self.df.shape,
                "columns": list(self.df.columns),
                "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
                "missing_values": self.df.isnull().sum().to_dict(),
                "memory_usage_mb": self.df.memory_usage(deep=True).sum() / (1024**2),
                "file_path": self.dataset_path
            }
            
            # Save original dataset
            original_path = os.path.join(self.output_dir, 'data', 'original.csv')
            self.df.to_csv(original_path, index=False)
            
            # Perform autonomous exploratory analysis
            analysis_result = self.autonomous_coder.analyze_dataset_exploratory(
                self.dataset_path
            )
            
            self.generated_code_paths.append(analysis_result.get('file_path'))
            self.results['autonomous_analysis'].append({
                "task": "exploratory_analysis",
                "result": analysis_result,
                "timestamp": datetime.now().isoformat()
            })
            
            self._log_step("Load and analyze dataset", "complete", 
                          f"Analyzed {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            return analysis_result
            
        except Exception as e:
            self._log_step("Load and analyze dataset", "error", str(e))
            raise
    
    def autonomous_feature_engineering(self, target_column: str = None) -> Dict[str, Any]:
        """
        Perform autonomous feature engineering
        
        Args:
            target_column: Target variable for supervised feature engineering
            
        Returns:
            Feature engineering results
        """
        self._log_step("Autonomous feature engineering", "start")
        
        try:
            result = self.autonomous_coder.perform_feature_engineering(
                self.dataset_path,
                target_column=target_column
            )
            
            self.generated_code_paths.append(result.get('file_path'))
            self.results['autonomous_analysis'].append({
                "task": "feature_engineering",
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            self._log_step("Autonomous feature engineering", "complete",
                          "Feature engineering completed")
            
            return result
            
        except Exception as e:
            self._log_step("Autonomous feature engineering", "error", str(e))
            raise
    
    def autonomous_model_building(self, target_column: str,
                                  task_type: str = "auto") -> Dict[str, Any]:
        """
        Build models autonomously
        
        Args:
            target_column: Target variable
            task_type: Type of task (classification, regression, auto)
            
        Returns:
            Model building results
        """
        self._log_step("Autonomous model building", "start")
        
        try:
            result = self.autonomous_coder.build_models_autonomously(
                self.dataset_path,
                target_column,
                task_type=task_type
            )
            
            self.generated_code_paths.append(result.get('file_path'))
            self.results['autonomous_analysis'].append({
                "task": "model_building",
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            self._log_step("Autonomous model building", "complete",
                          f"Models built for target: {target_column}")
            
            return result
            
        except Exception as e:
            self._log_step("Autonomous model building", "error", str(e))
            raise
    
    def run_custom_analysis(self, custom_task: str) -> Dict[str, Any]:
        """
        Run custom analysis task
        
        Args:
            custom_task: Description of custom analysis
            
        Returns:
            Analysis results
        """
        self._log_step("Custom analysis", "start", custom_task[:50])
        
        try:
            result = self.autonomous_coder.generate_custom_analysis(
                custom_task,
                dataset_path=self.dataset_path
            )
            
            self.generated_code_paths.append(result.get('file_path'))
            self.results['autonomous_analysis'].append({
                "task": "custom_analysis",
                "description": custom_task,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            self._log_step("Custom analysis", "complete",
                          "Custom analysis completed")
            
            return result
            
        except Exception as e:
            self._log_step("Custom analysis", "error", str(e))
            raise
    
    def generate_analysis_notebook(self, notebook_name: str = "analysis") -> str:
        """
        Generate a Jupyter notebook from all generated code
        
        Args:
            notebook_name: Name for the notebook
            
        Returns:
            Path to generated notebook
        """
        self._log_step("Generate analysis notebook", "start")
        
        try:
            notebook_path = self.autonomous_coder.create_notebook_from_analysis(
                self.generated_code_paths,
                notebook_name
            )
            
            self.results['notebook_path'] = notebook_path
            
            self._log_step("Generate analysis notebook", "complete",
                          f"Notebook created: {notebook_path}")
            
            return notebook_path
            
        except Exception as e:
            self._log_step("Generate analysis notebook", "error", str(e))
            raise
    
    def run_full_autonomous_pipeline(self, target_column: str = None,
                                    task_type: str = "auto",
                                    create_notebook: bool = True) -> Dict[str, Any]:
        """
        Run complete autonomous analysis pipeline
        
        Args:
            target_column: Target variable for modeling
            task_type: Type of ML task
            create_notebook: Whether to generate a notebook
            
        Returns:
            Complete results dictionary
        """
        start_time = datetime.now()
        
        self._log_step("Full autonomous pipeline", "start")
        
        try:
            # Setup environment
            self.setup_environment()
            
            # Step 1: Exploratory analysis
            self.load_and_analyze_dataset()
            
            # Step 2: Feature engineering
            self.autonomous_feature_engineering(target_column=target_column)
            
            # Step 3: Model building (if target column provided)
            if target_column:
                self.autonomous_model_building(target_column, task_type=task_type)
            
            # Step 4: Generate notebook
            if create_notebook:
                self.generate_analysis_notebook()
            
            # Save manifests
            self.autonomous_coder.save_coding_manifest()
            self.code_executor.save_code_manifest()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.results['duration_seconds'] = duration
            self.results['completed_at'] = end_time.isoformat()
            
            self._log_step("Full autonomous pipeline", "complete",
                          f"Completed in {duration:.1f} seconds")
            
            # Save final results
            self.save_results()
            
            return self.results
            
        except Exception as e:
            self._log_step("Full autonomous pipeline", "error", str(e))
            raise
    
    def save_results(self):
        """Save results to JSON file"""
        results_path = os.path.join(self.output_dir, 'autonomous_analysis_results.json')
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        return results_path
    
    def _log_step(self, step: str, status: str, message: str = ""):
        """Log pipeline step"""
        log_entry = {
            "step": step,
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.execution_log.append(log_entry)
        
        if status == "start":
            logger.info(f"[START] {step}")
        elif status == "complete":
            logger.info(f"[COMPLETE] {step}: {message}")
        elif status == "error":
            logger.error(f"[ERROR] {step}: {message}")
    
    def save_execution_log(self):
        """Save execution log to file"""
        log_path = os.path.join(self.output_dir, 'logs', 'autonomous_execution_log.json')
        
        with open(log_path, 'w') as f:
            json.dump({
                "run_id": self.run_id,
                "dataset_name": self.dataset_name,
                "output_dir": self.output_dir,
                "execution_log": self.execution_log,
                "total_generated_files": len(self.generated_code_paths),
                "results_summary": {
                    "autonomous_tasks": len(self.results.get('autonomous_analysis', [])),
                    "code_files": len(self.generated_code_paths)
                }
            }, f, indent=2, default=str)
        
        logger.info(f"Execution log saved to {log_path}")

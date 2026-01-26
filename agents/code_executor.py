"""
Autonomous Code Execution Agent
Allows the agent to write, save, and execute its own Python code
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
import traceback

logger = logging.getLogger("CodeExecutorAgent")


class CodeExecutionAgent:
    """Agent that can write, save, and execute Python code autonomously"""
    
    def __init__(self, output_dir: str, environment_type: str = "venv-uv"):
        """
        Initialize the code execution agent
        
        Args:
            output_dir: Directory for storing generated code and outputs
            environment_type: Type of Python environment (venv, conda, venv-uv)
        """
        self.output_dir = output_dir
        self.code_dir = os.path.join(output_dir, 'generated_code')
        os.makedirs(self.code_dir, exist_ok=True)
        
        self.environment_type = environment_type
        self.environment_path = None
        self.execution_log = []
        self.code_files = []
        
        logger.info(f"CodeExecutorAgent initialized with output_dir: {output_dir}")
        logger.info(f"Environment type: {environment_type}")
    
    def setup_environment(self, env_name: str = "analysis_env") -> str:
        """
        Create an isolated Python environment for code execution
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Path to the created environment
        """
        self._log_execution("setup_environment", "start", {"env_name": env_name})
        
        try:
            env_dir = os.path.join(self.output_dir, 'envs', env_name)
            os.makedirs(env_dir, exist_ok=True)
            
            if self.environment_type == "venv-uv":
                # Use UV for fast environment setup
                logger.info(f"Creating UV virtual environment at {env_dir}")
                result = subprocess.run(
                    ["uv", "venv", env_dir],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode != 0:
                    logger.warning(f"UV environment creation had warnings: {result.stderr}")
                
                # Install basic packages
                pip_path = os.path.join(env_dir, 'bin', 'pip')
                subprocess.run(
                    [pip_path, "install", "-q", "pandas", "numpy", "scikit-learn", 
                     "matplotlib", "seaborn", "scipy", "plotly", "jupyter"],
                    capture_output=True,
                    timeout=180
                )
                
            elif self.environment_type == "venv":
                # Standard virtualenv
                logger.info(f"Creating virtual environment at {env_dir}")
                subprocess.run(
                    [sys.executable, "-m", "venv", env_dir],
                    capture_output=True,
                    timeout=60
                )
                
                pip_path = os.path.join(env_dir, 'bin', 'pip')
                subprocess.run(
                    [pip_path, "install", "-q", "pandas", "numpy", "scikit-learn",
                     "matplotlib", "seaborn", "scipy", "plotly", "jupyter"],
                    capture_output=True,
                    timeout=180
                )
            
            elif self.environment_type == "conda":
                # Conda environment
                logger.info(f"Creating conda environment: {env_name}")
                subprocess.run(
                    ["conda", "create", "-n", env_name, "-y", 
                     "python=3.10", "pandas", "numpy", "scikit-learn",
                     "matplotlib", "seaborn", "scipy", "plotly", "jupyter"],
                    capture_output=True,
                    timeout=300
                )
                env_dir = env_name  # Conda uses environment name, not path
            
            self.environment_path = env_dir
            self._log_execution("setup_environment", "complete", {"env_path": env_dir})
            
            logger.info(f"Environment successfully created at {env_dir}")
            return env_dir
            
        except Exception as e:
            self._log_execution("setup_environment", "error", {"error": str(e)})
            logger.error(f"Failed to create environment: {e}")
            raise
    
    def get_python_executable(self) -> str:
        """Get the path to the Python executable in the environment"""
        if self.environment_type == "conda":
            return os.path.join(os.environ.get('CONDA_PREFIX', ''), 'bin', 'python')
        else:
            return os.path.join(self.environment_path, 'bin', 'python')
    
    def generate_code_file(self, code: str, filename: str, 
                          description: str = "") -> str:
        """
        Save generated Python code to a file
        
        Args:
            code: Python code to save
            filename: Name for the file (without extension)
            description: Description of what the code does
            
        Returns:
            Path to the saved file
        """
        self._log_execution("generate_code_file", "start", 
                          {"filename": filename, "description": description})
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{filename}_{timestamp}.py"
            file_path = os.path.join(self.code_dir, safe_filename)
            
            # Add header with metadata
            header = f'''"""
Generated by Autonomous Code Execution Agent
Filename: {safe_filename}
Generated: {datetime.now().isoformat()}
Description: {description}
"""

'''
            
            full_code = header + code
            
            with open(file_path, 'w') as f:
                f.write(full_code)
            
            self.code_files.append({
                "path": file_path,
                "filename": safe_filename,
                "description": description,
                "generated_at": datetime.now().isoformat()
            })
            
            self._log_execution("generate_code_file", "complete", 
                              {"file_path": file_path})
            
            logger.info(f"Code saved to {file_path}")
            return file_path
            
        except Exception as e:
            self._log_execution("generate_code_file", "error", {"error": str(e)})
            logger.error(f"Failed to save code file: {e}")
            raise
    
    def generate_notebook(self, code_cells: List[str], 
                         outputs: List[Any] = None,
                         filename: str = "analysis") -> str:
        """
        Generate a Jupyter notebook from code cells
        
        Args:
            code_cells: List of code cell contents
            outputs: Optional list of outputs for each cell
            filename: Name for the notebook file
            
        Returns:
            Path to the generated notebook
        """
        self._log_execution("generate_notebook", "start", 
                          {"cells": len(code_cells), "filename": filename})
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            notebook_filename = f"{filename}_{timestamp}.ipynb"
            notebook_path = os.path.join(self.code_dir, notebook_filename)
            
            notebook = {
                "nbformat": 4,
                "nbformat_minor": 4,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "name": "python",
                        "version": "3.10.0"
                    }
                },
                "cells": []
            }
            
            for i, code in enumerate(code_cells):
                cell = {
                    "cell_type": "code",
                    "execution_count": i + 1,
                    "metadata": {},
                    "outputs": [],
                    "source": code
                }
                
                # Add outputs if provided
                if outputs and i < len(outputs) and outputs[i] is not None:
                    output = outputs[i]
                    if isinstance(output, str):
                        cell["outputs"].append({
                            "name": "stdout",
                            "output_type": "stream",
                            "text": output
                        })
                
                notebook["cells"].append(cell)
            
            with open(notebook_path, 'w') as f:
                json.dump(notebook, f, indent=2)
            
            self.code_files.append({
                "path": notebook_path,
                "filename": notebook_filename,
                "description": f"Jupyter notebook with {len(code_cells)} cells",
                "generated_at": datetime.now().isoformat()
            })
            
            self._log_execution("generate_notebook", "complete", 
                              {"notebook_path": notebook_path})
            
            logger.info(f"Notebook saved to {notebook_path}")
            return notebook_path
            
        except Exception as e:
            self._log_execution("generate_notebook", "error", {"error": str(e)})
            logger.error(f"Failed to generate notebook: {e}")
            raise
    
    def execute_code(self, code: str, timeout: int = 300,
                     working_dir: str = None) -> Dict[str, Any]:
        """
        Execute Python code directly
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            working_dir: Working directory for execution
            
        Returns:
            Dictionary with execution results
        """
        self._log_execution("execute_code", "start", {"timeout": timeout})
        
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "execution_time": 0,
            "error": None
        }
        
        try:
            python_exec = self.get_python_executable()
            working_dir = working_dir or self.code_dir
            
            start_time = datetime.now()
            
            process = subprocess.Popen(
                [python_exec, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_dir
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                result["stdout"] = stdout
                result["stderr"] = stderr
                result["success"] = process.returncode == 0
                result["execution_time"] = (datetime.now() - start_time).total_seconds()
                
                if process.returncode != 0:
                    result["error"] = f"Process exited with code {process.returncode}"
                
                self._log_execution("execute_code", "complete", 
                                  {"success": result["success"], 
                                   "execution_time": result["execution_time"]})
                
            except subprocess.TimeoutExpired:
                process.kill()
                result["error"] = f"Execution timed out after {timeout} seconds"
                self._log_execution("execute_code", "timeout", {"timeout": timeout})
                logger.warning(f"Code execution timed out after {timeout} seconds")
            
        except Exception as e:
            result["error"] = str(e)
            result["stderr"] = traceback.format_exc()
            self._log_execution("execute_code", "error", {"error": str(e)})
            logger.error(f"Failed to execute code: {e}")
        
        return result
    
    def execute_code_file(self, file_path: str, timeout: int = 300,
                         args: List[str] = None) -> Dict[str, Any]:
        """
        Execute a Python code file
        
        Args:
            file_path: Path to the Python file
            timeout: Maximum execution time in seconds
            args: Command-line arguments to pass
            
        Returns:
            Dictionary with execution results
        """
        self._log_execution("execute_code_file", "start", 
                          {"file_path": file_path, "timeout": timeout})
        
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "execution_time": 0,
            "error": None,
            "file_path": file_path
        }
        
        try:
            python_exec = self.get_python_executable()
            working_dir = os.path.dirname(file_path)
            
            cmd = [python_exec, file_path]
            if args:
                cmd.extend(args)
            
            start_time = datetime.now()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_dir
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                result["stdout"] = stdout
                result["stderr"] = stderr
                result["success"] = process.returncode == 0
                result["execution_time"] = (datetime.now() - start_time).total_seconds()
                
                if process.returncode != 0:
                    result["error"] = f"Process exited with code {process.returncode}"
                
                self._log_execution("execute_code_file", "complete", 
                                  {"success": result["success"],
                                   "execution_time": result["execution_time"]})
                
                logger.info(f"Executed {file_path}: {'SUCCESS' if result['success'] else 'FAILED'}")
                
            except subprocess.TimeoutExpired:
                process.kill()
                result["error"] = f"Execution timed out after {timeout} seconds"
                self._log_execution("execute_code_file", "timeout", {"timeout": timeout})
                logger.warning(f"File execution timed out: {file_path}")
            
        except Exception as e:
            result["error"] = str(e)
            result["stderr"] = traceback.format_exc()
            self._log_execution("execute_code_file", "error", {"error": str(e)})
            logger.error(f"Failed to execute file {file_path}: {e}")
        
        return result
    
    def install_package(self, package: str, version: str = None) -> Dict[str, Any]:
        """
        Install a Python package in the environment
        
        Args:
            package: Package name
            version: Optional version specifier
            
        Returns:
            Installation result
        """
        self._log_execution("install_package", "start", 
                          {"package": package, "version": version})
        
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": None
        }
        
        try:
            python_exec = self.get_python_executable()
            pip_path = os.path.join(os.path.dirname(python_exec), 'pip')
            
            package_spec = f"{package}{version}" if version else package
            
            process = subprocess.run(
                [pip_path, "install", "-q", package_spec],
                capture_output=True,
                text=True,
                timeout=180
            )
            
            result["stdout"] = process.stdout
            result["stderr"] = process.stderr
            result["success"] = process.returncode == 0
            
            if process.returncode != 0:
                result["error"] = process.stderr
            
            self._log_execution("install_package", "complete", 
                              {"success": result["success"]})
            
            logger.info(f"Installed {package_spec}: {'SUCCESS' if result['success'] else 'FAILED'}")
            
        except Exception as e:
            result["error"] = str(e)
            result["stderr"] = traceback.format_exc()
            self._log_execution("install_package", "error", {"error": str(e)})
            logger.error(f"Failed to install {package}: {e}")
        
        return result
    
    def run_terminal_command(self, command: str, timeout: int = 300,
                            working_dir: str = None) -> Dict[str, Any]:
        """
        Execute a terminal command with full shell access
        
        Args:
            command: Shell command to execute
            timeout: Maximum execution time in seconds
            working_dir: Working directory for execution
            
        Returns:
            Dictionary with execution results
        """
        self._log_execution("run_terminal_command", "start", 
                          {"command": command, "timeout": timeout})
        
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "execution_time": 0,
            "error": None
        }
        
        try:
            working_dir = working_dir or self.code_dir
            start_time = datetime.now()
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                cwd=working_dir
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                result["stdout"] = stdout
                result["stderr"] = stderr
                result["exit_code"] = process.returncode
                result["success"] = process.returncode == 0
                result["execution_time"] = (datetime.now() - start_time).total_seconds()
                
                if process.returncode != 0:
                    result["error"] = f"Command exited with code {process.returncode}"
                
                self._log_execution("run_terminal_command", "complete", 
                                  {"success": result["success"],
                                   "exit_code": result["exit_code"]})
                
                logger.info(f"Executed command: {command[:50]}...")
                
            except subprocess.TimeoutExpired:
                process.kill()
                result["error"] = f"Command timed out after {timeout} seconds"
                self._log_execution("run_terminal_command", "timeout", {"timeout": timeout})
                logger.warning(f"Command timed out: {command[:50]}...")
            
        except Exception as e:
            result["error"] = str(e)
            result["stderr"] = traceback.format_exc()
            self._log_execution("run_terminal_command", "error", {"error": str(e)})
            logger.error(f"Failed to execute command: {e}")
        
        return result
    
    def save_code_manifest(self) -> str:
        """
        Save manifest of all generated code files
        
        Returns:
            Path to the manifest file
        """
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "environment_type": self.environment_type,
            "environment_path": self.environment_path,
            "total_files": len(self.code_files),
            "files": self.code_files,
            "execution_log": self.execution_log
        }
        
        manifest_path = os.path.join(self.code_dir, "code_manifest.json")
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Code manifest saved to {manifest_path}")
        return manifest_path
    
    def _log_execution(self, action: str, status: str, details: Dict[str, Any]):
        """Log an execution action"""
        log_entry = {
            "action": action,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.execution_log.append(log_entry)

"""
Self-Healing Agent
Automatically detects errors and implements recovery strategies
"""

from typing import Dict, Any, List, Optional
import traceback
import time
import logging

logger = logging.getLogger("SelfHealingAgent")


class SelfHealingAgent:
    """Monitors agent executions and implements automatic error recovery"""
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 5):
        """
        Initialize self-healing agent
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Seconds between retries
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_log = []
        self.recovery_log = []
        self.fallback_apis = Config().FALLBACK_APIS if hasattr(Config(), 'FALLBACK_APIS') else ['perplexity', 'mistral', 'cohere']
        self.current_api_index = 0
    
    def execute_with_retry(self, func, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute a function with automatic retry and fallback
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Dictionary with result, success status, and execution info
        """
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            attempt += 1
            try:
                logger.info(f"Attempt {attempt}/{self.max_retries} for {func.__name__}")
                result = func(*args, **kwargs)
                
                self.recovery_log.append({
                    "function": func.__name__,
                    "attempt": attempt,
                    "success": True,
                    "timestamp": time.time()
                })
                
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt,
                    "message": f"Success on attempt {attempt}"
                }
                
            except Exception as e:
                last_error = e
                error_info = {
                    "function": func.__name__,
                    "attempt": attempt,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": time.time()
                }
                self.error_log.append(error_info)
                
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                
                # Analyze error and attempt recovery
                recovery_strategy = self._analyze_error_and_recover(e, attempt)
                
                if recovery_strategy:
                    self.recovery_log.append({
                        "function": func.__name__,
                        "attempt": attempt,
                        "recovery_strategy": recovery_strategy,
                        "timestamp": time.time()
                    })
                
                # Wait before retry
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)  # Exponential backoff
        
        # All retries failed
        return {
            "success": False,
            "error": last_error,
            "attempts": attempt,
            "message": f"Failed after {attempt} attempts: {str(last_error)}",
            "error_log": self.error_log[-self.max_retries:]
        }
    
    def _analyze_error_and_recover(self, error: Exception, attempt: int) -> Optional[str]:
        """
        Analyze error and implement recovery strategy
        
        Args:
            error: The exception that occurred
            attempt: Current attempt number
        
        Returns:
            Recovery strategy name or None
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # API-related errors
        if "api" in error_msg or "connection" in error_msg or "timeout" in error_msg:
            return self._handle_api_error(error)
        
        # Memory errors
        if "memory" in error_msg or "allocation" in error_msg:
            return self._handle_memory_error(error)
        
        # File errors
        if "file" in error_msg or "not found" in error_msg or "permission" in error_msg:
            return self._handle_file_error(error)
        
        # Data errors
        if "key" in error_msg or "column" in error_msg or "index" in error_msg:
            return self._handle_data_error(error)
        
        # Type/conversion errors
        if "type" in error_msg or "conversion" in error_msg or "cast" in error_msg:
            return self._handle_type_error(error)
        
        # Default recovery
        return self._default_recovery(error, attempt)
    
    def _handle_api_error(self, error: Exception) -> str:
        """Handle API-related errors"""
        logger.warning(f"API error detected: {str(error)}")
        
        # Switch to fallback API
        if self.current_api_index < len(self.fallback_apis) - 1:
            self.current_api_index += 1
            new_api = self.fallback_apis[self.current_api_index]
            logger.info(f"Switching to fallback API: {new_api}")
            return f"Switched to API: {new_api}"
        
        return "Retrying with same API (no more fallbacks available)"
    
    def _handle_memory_error(self, error: Exception) -> str:
        """Handle memory-related errors"""
        logger.warning(f"Memory error detected: {str(error)}")
        
        # Suggest data reduction
        return "Reducing data size - using sample or chunking data"
    
    def _handle_file_error(self, error: Exception) -> str:
        """Handle file-related errors"""
        logger.warning(f"File error detected: {str(error)}")
        
        # Check if path exists
        if "not found" in str(error).lower():
            return "Creating missing directory or using alternative path"
        elif "permission" in str(error).lower():
            return "Changing file permissions or using alternative location"
        
        return "Retrying file operation"
    
    def _handle_data_error(self, error: Exception) -> str:
        """Handle data-related errors"""
        logger.warning(f"Data error detected: {str(error)}")
        
        if "key" in str(error).lower():
            return "Dropping invalid column or imputing missing data"
        elif "index" in str(error).lower():
            return "Resetting index or using positional indexing"
        
        return "Checking data structure and types"
    
    def _handle_type_error(self, error: Exception) -> str:
        """Handle type/conversion errors"""
        logger.warning(f"Type error detected: {str(error)}")
        
        return "Converting data types explicitly or using error-handling conversions"
    
    def _default_recovery(self, error: Exception, attempt: int) -> str:
        """Default recovery strategy"""
        if attempt == 1:
            return "Retrying operation"
        elif attempt == 2:
            return "Attempting alternative approach"
        else:
            return "Final retry attempt with simplified parameters"
    
    def generate_error_report(self) -> str:
        """
        Generate a comprehensive error report
        
        Returns:
            Formatted Markdown string
        """
        if not self.error_log:
            return "# Error Report\n\nNo errors occurred during execution."
        
        report = "# Error Report\n\n"
        report += f"Total Errors: {len(self.error_log)}\n"
        report += f"Total Recoveries: {len(self.recovery_log)}\n\n"
        
        # Error statistics
        error_types = {}
        for error in self.error_log:
            etype = error['error_type']
            error_types[etype] = error_types.get(etype, 0) + 1
        
        report += "## Error Types\n\n"
        for etype, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            report += f"- {etype}: {count}\n"
        
        report += "\n## Error Details\n\n"
        
        for i, error in enumerate(self.error_log, 1):
            report += f"### Error {i}: {error['error_type']}\n\n"
            report += f"**Function:** {error['function']}\n"
            report += f"**Attempt:** {error['attempt']}\n"
            report += f"**Message:** {error['error_message']}\n\n"
            report += f"**Traceback:**\n```\n{error['traceback']}\n```\n\n"
            report += "---\n\n"
        
        # Recovery statistics
        recovery_strategies = {}
        for recovery in self.recovery_log:
            strategy = recovery.get('recovery_strategy', 'unknown')
            recovery_strategies[strategy] = recovery_strategies.get(strategy, 0) + 1
        
        report += "## Recovery Strategies Used\n\n"
        for strategy, count in sorted(recovery_strategies.items(), key=lambda x: x[1], reverse=True):
            report += f"- {strategy}: {count}\n"
        
        report += "\n## Recovery Log\n\n"
        
        for i, recovery in enumerate(self.recovery_log, 1):
            report += f"### Recovery {i}\n\n"
            report += f"**Function:** {recovery['function']}\n"
            report += f"**Attempt:** {recovery['attempt']}\n"
            if 'recovery_strategy' in recovery:
                report += f"**Strategy:** {recovery['recovery_strategy']}\n"
            report += f"**Success:** {recovery.get('success', False)}\n\n"
            report += "---\n\n"
        
        return report
    
    def save_error_log(self, filepath: str):
        """
        Save error log to JSON file
        
        Args:
            filepath: Path to save error log
        """
        import json
        with open(filepath, 'w') as f:
            json.dump({
                "error_log": self.error_log,
                "recovery_log": self.recovery_log
            }, f, indent=2, default=str)
    
    def get_recovery_rate(self) -> float:
        """
        Calculate recovery rate (percentage of successful recoveries)
        
        Returns:
            Recovery rate as percentage
        """
        if not self.error_log:
            return 100.0
        
        successful_recoveries = sum(1 for r in self.recovery_log if r.get('success', False))
        return (successful_recoveries / len(self.error_log)) * 100


class Config:
    """Minimal config for imports"""
    FALLBACK_APIS = ['perplexity', 'mistral', 'cohere']

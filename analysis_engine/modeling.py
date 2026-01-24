"""
Predictive Modeling Engine
Implements various predictive models for forecasting and classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           mean_squared_error, r2_score, mean_absolute_error, 
                           roc_auc_score, precision_recall_fscore_support)
from sklearn.impute import SimpleImputer
from typing import Dict, Any, List, Optional, Tuple
import json


class ModelBuilder:
    """Builds and evaluates predictive models"""
    
    def __init__(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize model builder
        
        Args:
            df: Input DataFrame
            target_column: Name of target variable column
            test_size: Proportion of data for testing (default: 0.2)
            random_state: Random seed for reproducibility
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        self.models = {}
        self.best_model = None
        self.task_type = None  # 'classification' or 'regression'
        self.preprocessing_applied = False
    
    def prepare_data(self, drop_columns: Optional[List[str]] = None, 
                     encode_categorical: bool = True, 
                     scale_features: bool = True) -> Dict[str, Any]:
        """
        Prepare data for modeling
        
        Args:
            drop_columns: Columns to drop (besides target)
            encode_categorical: Whether to encode categorical variables
            scale_features: Whether to scale numerical features
        
        Returns:
            Dictionary with preparation info
        """
        df_prep = self.df.copy()
        
        # Drop columns if specified
        if drop_columns:
            df_prep = df_prep.drop(columns=[col for col in drop_columns if col in df_prep.columns])
        
        # Separate features and target
        if self.target_column not in df_prep.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        
        y = df_prep[self.target_column]
        X = df_prep.drop(columns=[self.target_column])
        
        # Determine task type
        if y.dtype in ['object', 'category']:
            self.task_type = 'classification'
            # Encode target
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y.astype(str))
        elif y.nunique() <= 20:  # Few unique values for numeric target
            self.task_type = 'classification'
            y = y.astype(int)
        else:
            self.task_type = 'regression'
        
        self.y = y
        
        # Handle categorical features
        if encode_categorical:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.categorical_encoders = {}
            
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.categorical_encoders[col] = le
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Scale features
        if scale_features:
            self.scaler = StandardScaler()
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        self.preprocessing_applied = True
        
        return {
            "task_type": self.task_type,
            "feature_count": len(self.feature_names),
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "features": self.feature_names
        }
    
    def build_linear_regression(self, regularization: Optional[str] = None, alpha: float = 1.0) -> Dict[str, Any]:
        """
        Build linear regression model
        
        Args:
            regularization: None, 'ridge', or 'lasso'
            alpha: Regularization strength
        
        Returns:
            Dictionary with model evaluation
        """
        if not self.preprocessing_applied:
            self.prepare_data()
        
        if self.task_type != 'regression':
            return {"error": "Linear regression is for regression tasks only"}
        
        if regularization == 'ridge':
            model = Ridge(alpha=alpha, random_state=self.random_state)
        elif regularization == 'lasso':
            model = Lasso(alpha=alpha, random_state=self.random_state)
        else:
            model = LinearRegression()
        
        # Train model
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Metrics
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
        
        result = {
            "model_type": f"Linear Regression ({regularization})" if regularization else "Linear Regression",
            "train_mse": round(train_mse, 4),
            "test_mse": round(test_mse, 4),
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "train_mae": round(train_mae, 4),
            "test_mae": round(test_mae, 4),
            "feature_importance": feature_importance.to_dict('records'),
            "interpretation": self._interpret_regression_results(test_r2, test_mae)
        }
        
        model_name = f"linear_regression_{regularization}" if regularization else "linear_regression"
        self.models[model_name] = {"model": model, "metrics": result}
        
        return result
    
    def _interpret_regression_results(self, r2: float, mae: float) -> str:
        """Interpret regression results"""
        if r2 > 0.9:
            model_quality = "excellent"
        elif r2 > 0.7:
            model_quality = "good"
        elif r2 > 0.5:
            model_quality = "moderate"
        else:
            model_quality = "poor"
        
        interpretation = f"The model achieves a {model_quality} fit with R² = {r2:.4f}. "
        interpretation += f"On average, predictions are off by {mae:.4f} units (MAE). "
        
        if r2 > 0.8:
            interpretation += "The model explains most of the variance in the target variable and should be reliable for predictions."
        elif r2 > 0.5:
            interpretation += "The model explains a reasonable portion of the variance but may benefit from additional features or different algorithms."
        else:
            interpretation += "The model explains limited variance and may not be suitable for accurate predictions. Consider feature engineering or alternative approaches."
        
        return interpretation
    
    def build_random_forest(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                           min_samples_split: int = 2, min_samples_leaf: int = 1) -> Dict[str, Any]:
        """
        Build Random Forest model
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf node
        
        Returns:
            Dictionary with model evaluation
        """
        if not self.preprocessing_applied:
            self.prepare_data()
        
        if self.task_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Train model
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        if self.task_type == 'classification':
            # Classification metrics
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            
            # Classification report
            test_report = classification_report(self.y_test, y_test_pred, output_dict=True, zero_division=0)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            result = {
                "model_type": "Random Forest Classifier",
                "train_accuracy": round(train_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
                "classification_report": test_report,
                "feature_importance": feature_importance.to_dict('records'),
                "interpretation": self._interpret_classification_results(test_accuracy, test_report)
            }
        else:
            # Regression metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            result = {
                "model_type": "Random Forest Regressor",
                "train_mse": round(train_mse, 4),
                "test_mse": round(test_mse, 4),
                "train_r2": round(train_r2, 4),
                "test_r2": round(test_r2, 4),
                "train_mae": round(train_mae, 4),
                "test_mae": round(test_mae, 4),
                "feature_importance": feature_importance.to_dict('records'),
                "interpretation": self._interpret_regression_results(test_r2, test_mae)
            }
        
        model_name = "random_forest"
        self.models[model_name] = {"model": model, "metrics": result}
        
        return result
    
    def _interpret_classification_results(self, accuracy: float, report: Dict) -> str:
        """Interpret classification results"""
        if accuracy > 0.9:
            model_quality = "excellent"
        elif accuracy > 0.8:
            model_quality = "very good"
        elif accuracy > 0.7:
            model_quality = "good"
        elif accuracy > 0.6:
            model_quality = "moderate"
        else:
            model_quality = "poor"
        
        interpretation = f"The model achieves {model_quality} accuracy of {accuracy:.4f} on test data. "
        
        # Macro average F1
        if 'macro avg' in report:
            macro_f1 = report['macro avg']['f1-score']
            interpretation += f"The macro F1-score is {macro_f1:.4f}, indicating "
            
            if macro_f1 > 0.8:
                interpretation += "strong overall performance across all classes."
            elif macro_f1 > 0.6:
                interpretation += "good overall performance with some class imbalance."
            else:
                interpretation += "challenges in handling class imbalance or difficult classes."
        
        # Check for overfitting
        if hasattr(self, 'models') and 'random_forest' in self.models:
            train_acc = self.models['random_forest']['metrics'].get('train_accuracy')
            if train_acc and train_acc - accuracy > 0.1:
                interpretation += " There are signs of overfitting (train accuracy significantly higher than test accuracy)."
        
        return interpretation
    
    def build_gradient_boosting(self, n_estimators: int = 100, learning_rate: float = 0.1,
                               max_depth: int = 3) -> Dict[str, Any]:
        """
        Build Gradient Boosting model
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Shrinks contribution of each tree
            max_depth: Maximum depth of individual trees
        
        Returns:
            Dictionary with model evaluation
        """
        if not self.preprocessing_applied:
            self.prepare_data()
        
        if self.task_type == 'classification':
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=self.random_state
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=self.random_state
            )
        
        # Train model
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        if self.task_type == 'classification':
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            test_report = classification_report(self.y_test, y_test_pred, output_dict=True, zero_division=0)
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            result = {
                "model_type": "Gradient Boosting Classifier",
                "train_accuracy": round(train_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
                "classification_report": test_report,
                "feature_importance": feature_importance.to_dict('records'),
                "interpretation": self._interpret_classification_results(test_accuracy, test_report)
            }
        else:
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            result = {
                "model_type": "Gradient Boosting Regressor",
                "train_mse": round(train_mse, 4),
                "test_mse": round(test_mse, 4),
                "train_r2": round(train_r2, 4),
                "test_r2": round(test_r2, 4),
                "train_mae": round(train_mae, 4),
                "test_mae": round(test_mae, 4),
                "feature_importance": feature_importance.to_dict('records'),
                "interpretation": self._interpret_regression_results(test_r2, test_mae)
            }
        
        model_name = "gradient_boosting"
        self.models[model_name] = {"model": model, "metrics": result}
        
        return result
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all built models
        
        Returns:
            DataFrame with model comparison
        """
        if not self.models:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, model_data in self.models.items():
            metrics = model_data['metrics']
            
            row = {
                "model": metrics.get('model_type', model_name),
                "model_name": model_name
            }
            
            if self.task_type == 'classification':
                row['test_accuracy'] = metrics.get('test_accuracy')
                row['train_accuracy'] = metrics.get('train_accuracy')
                row['macro_f1'] = metrics.get('classification_report', {}).get('macro avg', {}).get('f1-score')
                row['weighted_f1'] = metrics.get('classification_report', {}).get('weighted avg', {}).get('f1-score')
            else:
                row['test_r2'] = metrics.get('test_r2')
                row['train_r2'] = metrics.get('train_r2')
                row['test_mse'] = metrics.get('test_mse')
                row['test_mae'] = metrics.get('test_mae')
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by best metric
        if self.task_type == 'classification':
            df = df.sort_values('test_accuracy', ascending=False)
        else:
            df = df.sort_values('test_r2', ascending=False)
        
        return df
    
    def get_best_model(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Get the best performing model
        
        Returns:
            Tuple of (model, metrics)
        """
        if not self.models:
            return None, {}
        
        comparison = self.compare_models()
        best_model_name = comparison.iloc[0]['model_name']
        
        self.best_model = self.models[best_model_name]
        return self.best_model['model'], self.best_model['metrics']
    
    def feature_importance_summary(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance summary across all models
        
        Args:
            top_n: Number of top features to show
        
        Returns:
            DataFrame with feature importance summary
        """
        importance_data = []
        
        for model_name, model_data in self.models.items():
            metrics = model_data['metrics']
            if 'feature_importance' in metrics:
                for feat_imp in metrics['feature_importance'][:top_n]:
                    importance_data.append({
                        'model': metrics.get('model_type', model_name),
                        'feature': feat_imp['feature'],
                        'importance': feat_imp['importance']
                    })
        
        if not importance_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(importance_data)
        
        # Calculate average importance across models
        avg_importance = df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        return pd.DataFrame({
            'feature': avg_importance.index,
            'average_importance': avg_importance.values
        })
    
    def predict(self, X_new: pd.DataFrame = None) -> np.ndarray:
        """
        Make predictions using the best model
        
        Args:
            X_new: New data to predict (default: use test set)
        
        Returns:
            Predictions array
        """
        if self.best_model is None:
            self.get_best_model()
        
        model = self.best_model['model']
        
        if X_new is None:
            return model.predict(self.X_test)
        
        # Apply same preprocessing
        for col, encoder in self.categorical_encoders.items():
            if col in X_new.columns:
                X_new[col] = encoder.transform(X_new[col].astype(str))
        
        # Handle missing values and scale
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_new = pd.DataFrame(imputer.fit_transform(X_new), columns=X_new.columns)
        X_new = pd.DataFrame(self.scaler.transform(X_new), columns=X_new.columns)
        
        return model.predict(X_new)
    
    def format_results_for_report(self) -> str:
        """
        Format modeling results for report
        
        Returns:
            Formatted Markdown string
        """
        if not self.models:
            return "No models have been built."
        
        report = "# Predictive Modeling Results\n\n"
        report += f"Task Type: {self.task_type.capitalize()}\n"
        report += f"Target Variable: {self.target_column}\n"
        report += f"Features: {len(self.feature_names)}\n\n"
        
        # Model comparison
        comparison = self.compare_models()
        report += "## Model Comparison\n\n"
        report += comparison.to_markdown(index=False)
        report += "\n\n"
        
        # Best model details
        best_model, best_metrics = self.get_best_model()
        report += f"## Best Model: {best_metrics.get('model_type')}\n\n"
        report += f"**Performance:** {best_metrics.get('interpretation')}\n\n"
        
        # Feature importance
        feature_importance = self.feature_importance_summary()
        if not feature_importance.empty:
            report += "## Feature Importance\n\n"
            report += feature_importance.head(15).to_markdown(index=False)
            report += "\n\n"
        
        return report
    
    def save_results(self, filepath: str):
        """
        Save modeling results to JSON
        
        Args:
            filepath: Path to save results
        """
        results = {
            "task_type": self.task_type,
            "target_column": self.target_column,
            "features": self.feature_names,
            "models": {}
        }
        
        for model_name, model_data in self.models.items():
            # Convert non-serializable objects
            metrics_copy = model_data['metrics'].copy()
            if 'feature_importance' in metrics_copy:
                # Convert DataFrames to lists
                metrics_copy['feature_importance'] = metrics_copy['feature_importance']
            results['models'][model_name] = {
                'metrics': metrics_copy
            }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

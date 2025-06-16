from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sp500_analysis.application.evaluation.evaluator import Evaluator


class TemporalEvaluator:
    """Temporal evaluator implementing step_7_0_train_models.py evaluation logic."""

    def __init__(self) -> None:
        self.base_evaluator = Evaluator()

    def evaluate_holdout(
        self, 
        model: Any, 
        X_holdout: pd.DataFrame, 
        y_holdout: pd.Series, 
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model on holdout data (Zone C).
        
        Args:
            model: Trained model
            X_holdout: Holdout features
            y_holdout: Holdout targets
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Handle scaling if model has scaler
            if hasattr(model, '_scaler') and model._scaler is not None:
                X_holdout_scaled = pd.DataFrame(
                    model._scaler.transform(X_holdout),
                    columns=X_holdout.columns,
                    index=X_holdout.index
                )
            else:
                X_holdout_scaled = X_holdout

            # Make predictions
            y_pred = model.predict(X_holdout_scaled)
            
            # Calculate basic metrics
            metrics = self.base_evaluator.evaluate(y_holdout, y_pred)
            
            # Add temporal-specific metrics
            temporal_metrics = self._calculate_temporal_metrics(y_holdout, y_pred)
            metrics.update(temporal_metrics)
            
            # Add directional accuracy
            direction_accuracy = self._calculate_directional_accuracy(y_holdout, y_pred)
            metrics['Directional_Accuracy'] = direction_accuracy
            
            logging.info(f"Holdout evaluation for {model_name}:")
            for metric, value in metrics.items():
                logging.info(f"  {metric}: {value:.6f}")
                
            return metrics
            
        except Exception as e:
            logging.error(f"Holdout evaluation failed for {model_name}: {e}")
            return {'error': str(e)}

    def evaluate_backtest(
        self, 
        model: Any, 
        X_val: pd.DataFrame, 
        y_val: pd.Series, 
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data (Zone B) for backtesting.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation targets
            model_name: Name of the model
            
        Returns:
            Dictionary with backtest metrics
        """
        try:
            # Handle scaling if model has scaler
            if hasattr(model, '_scaler') and model._scaler is not None:
                X_val_scaled = pd.DataFrame(
                    model._scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index
                )
            else:
                X_val_scaled = X_val

            # Make predictions
            y_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            metrics = self.base_evaluator.evaluate(y_val, y_pred)
            
            # Add temporal metrics
            temporal_metrics = self._calculate_temporal_metrics(y_val, y_pred)
            metrics.update(temporal_metrics)
            
            logging.info(f"Backtest evaluation for {model_name}:")
            for metric, value in metrics.items():
                logging.info(f"  {metric}: {value:.6f}")
                
            return metrics
            
        except Exception as e:
            logging.error(f"Backtest evaluation failed for {model_name}: {e}")
            return {'error': str(e)}

    def _calculate_temporal_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate additional temporal-specific metrics."""
        
        metrics = {}
        
        try:
            # Mean Absolute Percentage Error (MAPE)
            non_zero_mask = y_true != 0
            if non_zero_mask.any():
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                metrics['MAPE'] = mape
            
            # Normalized RMSE
            y_range = y_true.max() - y_true.min()
            if y_range > 0:
                rmse = sqrt(mean_squared_error(y_true, y_pred))
                metrics['NRMSE'] = rmse / y_range
            
            # Mean Bias Error
            metrics['MBE'] = np.mean(y_pred - y_true)
            
            # Standard Deviation of Residuals
            residuals = y_pred - y_true
            metrics['Residual_Std'] = np.std(residuals)
            
        except Exception as e:
            logging.warning(f"Could not calculate some temporal metrics: {e}")
            
        return metrics

    def _calculate_directional_accuracy(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray
    ) -> float:
        """Calculate directional accuracy (how often prediction direction matches actual direction)."""
        
        try:
            if len(y_true) < 2:
                return 0.0
                
            # Calculate actual and predicted changes
            actual_changes = np.diff(y_true.values)
            predicted_changes = np.diff(y_pred)
            
            # Calculate directional accuracy
            correct_directions = np.sign(actual_changes) == np.sign(predicted_changes)
            directional_accuracy = np.mean(correct_directions)
            
            return directional_accuracy
            
        except Exception as e:
            logging.warning(f"Could not calculate directional accuracy: {e}")
            return 0.0

    def calculate_forecast_metrics(
        self, 
        forecast_values: np.ndarray, 
        model_name: str
    ) -> Dict[str, Any]:
        """Calculate metrics for forecasted values."""
        
        metrics = {
            'forecast_mean': np.mean(forecast_values),
            'forecast_std': np.std(forecast_values),
            'forecast_min': np.min(forecast_values),
            'forecast_max': np.max(forecast_values),
            'forecast_range': np.max(forecast_values) - np.min(forecast_values)
        }
        
        logging.info(f"Forecast metrics for {model_name}:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value:.6f}")
            
        return metrics 
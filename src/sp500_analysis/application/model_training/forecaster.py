from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from sp500_analysis.config.settings import settings


class ModelForecaster:
    """Model forecaster implementing step_7_0_train_models.py forecasting logic."""

    def __init__(self, forecast_days: int = 30) -> None:
        self.forecast_days = forecast_days

    def generate_complete_forecast(
        self, 
        model: Any, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        model_name: str,
        model_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate complete forecast results including future predictions.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            model_name: Name of the model
            model_params: Model parameters
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Generate future predictions
            future_predictions = self._generate_future_predictions(
                model, X_train, model_name
            )
            
            # Generate forecast dates
            forecast_dates = self._generate_forecast_dates()
            
            # Create forecast summary
            forecast_summary = self._create_forecast_summary(
                future_predictions, forecast_dates, model_name
            )
            
            # Calculate forecast statistics
            forecast_stats = self._calculate_forecast_statistics(future_predictions)
            
            return {
                'predictions': dict(zip(forecast_dates, future_predictions)),
                'summary': forecast_summary,
                'statistics': forecast_stats,
                'model_params': model_params,
                'forecast_generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Forecast generation failed for {model_name}: {e}")
            return {'error': str(e)}

    def _generate_future_predictions(
        self, 
        model: Any, 
        X_train: pd.DataFrame, 
        model_name: str
    ) -> np.ndarray:
        """Generate predictions for future dates."""
        
        try:
            # Use the last observation as the base for prediction
            last_observation = X_train.iloc[-1:].copy()
            
            # Handle scaling if model has scaler
            if hasattr(model, '_scaler') and model._scaler is not None:
                last_observation_scaled = pd.DataFrame(
                    model._scaler.transform(last_observation),
                    columns=last_observation.columns,
                    index=last_observation.index
                )
            else:
                last_observation_scaled = last_observation
            
            # For simplicity, use the last observation to predict multiple future points
            # In practice, you would implement iterative prediction or use model-specific approaches
            predictions = []
            
            for i in range(self.forecast_days):
                # Make prediction
                pred = model.predict(last_observation_scaled)[0]
                predictions.append(pred)
                
                # For iterative prediction (if supported), update features
                # This is a simplified approach
                if hasattr(model, 'supports_iterative_prediction'):
                    # Update features for next prediction
                    # This would be model-specific implementation
                    pass
            
            return np.array(predictions)
            
        except Exception as e:
            logging.error(f"Future prediction generation failed: {e}")
            # Return conservative predictions (flat forecast)
            last_target_value = X_train.iloc[-1, -1] if len(X_train.columns) > 0 else 0.0
            return np.full(self.forecast_days, last_target_value)

    def _generate_forecast_dates(self) -> List[str]:
        """Generate forecast dates starting from tomorrow."""
        
        start_date = datetime.now() + timedelta(days=1)
        forecast_dates = []
        
        for i in range(self.forecast_days):
            forecast_date = start_date + timedelta(days=i)
            forecast_dates.append(forecast_date.strftime('%Y-%m-%d'))
            
        return forecast_dates

    def _create_forecast_summary(
        self, 
        predictions: np.ndarray, 
        dates: List[str], 
        model_name: str
    ) -> Dict[str, Any]:
        """Create a summary of the forecast."""
        
        return {
            'model_name': model_name,
            'forecast_period': f"{dates[0]} to {dates[-1]}",
            'num_predictions': len(predictions),
            'mean_prediction': float(np.mean(predictions)),
            'prediction_trend': self._calculate_trend(predictions),
            'confidence_range': {
                'lower': float(np.percentile(predictions, 25)),
                'upper': float(np.percentile(predictions, 75))
            }
        }

    def _calculate_forecast_statistics(self, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate statistical measures of the forecast."""
        
        return {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'volatility': float(np.std(predictions) / np.mean(predictions)) if np.mean(predictions) != 0 else 0.0,
            'total_change': float(predictions[-1] - predictions[0]) if len(predictions) > 1 else 0.0,
            'cumulative_return': float(np.sum(predictions))
        }

    def _calculate_trend(self, predictions: np.ndarray) -> str:
        """Calculate the overall trend of predictions."""
        
        if len(predictions) < 2:
            return "stable"
            
        start_value = predictions[0]
        end_value = predictions[-1]
        change_pct = (end_value - start_value) / start_value * 100 if start_value != 0 else 0
        
        if change_pct > 2:
            return "bullish"
        elif change_pct < -2:
            return "bearish"
        else:
            return "stable"

    def generate_scenario_forecasts(
        self, 
        model: Any, 
        X_train: pd.DataFrame, 
        model_name: str,
        scenarios: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Generate forecasts under different scenarios."""
        
        if scenarios is None:
            scenarios = {
                'optimistic': 1.1,
                'base': 1.0,
                'pessimistic': 0.9
            }
        
        scenario_results = {}
        
        for scenario_name, multiplier in scenarios.items():
            try:
                # Generate base forecast
                base_forecast = self._generate_future_predictions(model, X_train, model_name)
                
                # Apply scenario multiplier
                scenario_forecast = base_forecast * multiplier
                
                # Store results
                scenario_results[scenario_name] = {
                    'predictions': scenario_forecast.tolist(),
                    'statistics': self._calculate_forecast_statistics(scenario_forecast)
                }
                
            except Exception as e:
                logging.error(f"Scenario {scenario_name} generation failed: {e}")
                scenario_results[scenario_name] = {'error': str(e)}
        
        return scenario_results 
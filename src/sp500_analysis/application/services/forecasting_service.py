"""
Forecasting Service - Handles advanced forecasting features from step_7_0_train_models (1).py
in a modular way.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from sp500_analysis.config.settings import settings


class ForecastingService:
    """Service for advanced forecasting functionality."""

    def __init__(self):
        self.results_dir = settings.results_dir
        self.processed_dir = settings.processed_dir

    def generate_comprehensive_forecast(
        self,
        model: Any,
        X_all: pd.DataFrame,
        y_all: pd.Series,
        model_name: str,
        best_params: Dict[str, Any],
        forecast_horizon: int = 20
    ) -> Dict[str, Any]:
        """
        Generate comprehensive forecast with all features from step_7_0_train_models (1).py.
        """
        
        logging.info(f"[{model_name}] Generating comprehensive forecast")
        
        try:
            # Step 1: Generate predictions for last 20 days (validation)
            last_20_days_results = self._generate_last_20_days_predictions(
                model, X_all, y_all, model_name
            )
            
            # Step 2: Generate future forecast for next 20 business days
            future_forecast_results = self._generate_business_days_forecast(
                model, last_20_days_results, model_name, forecast_horizon
            )
            
            # Step 3: Create complete forecast DataFrame
            complete_forecast_df = self._create_complete_forecast_dataframe(
                model_name, best_params, X_all, last_20_days_results, 
                future_forecast_results
            )
            
            # Step 4: Validate and save forecast
            self._validate_and_save_forecast(complete_forecast_df, model_name)
            
            return {
                'last_20_days': last_20_days_results,
                'future_forecast': future_forecast_results,
                'complete_dataframe': complete_forecast_df,
                'forecast_horizon': forecast_horizon
            }
            
        except Exception as e:
            logging.error(f"[{model_name}] Error in comprehensive forecast: {e}")
            return {}

    def _generate_last_20_days_predictions(
        self,
        model: Any,
        X_all: pd.DataFrame,
        y_all: pd.Series,
        model_name: str,
        lag_days: int = 20
    ) -> Dict[str, Any]:
        """Generate predictions for the last 20 days for validation."""
        
        logging.info(f"[{model_name}] Generating last 20 days predictions")
        
        try:
            # Get last 20 days of data
            if len(X_all) < lag_days:
                logging.warning(f"[{model_name}] Not enough data for {lag_days} days")
                return {}
            
            # Extract last N days
            X_last = X_all.tail(lag_days).copy()
            y_last = y_all.tail(lag_days).copy()
            
            # Generate predictions
            if hasattr(model, 'predict'):
                if model_name in ['LSTM', 'TTS']:
                    # Handle sequence models differently
                    predictions = self._predict_sequence_model(model, X_last, model_name)
                else:
                    predictions = model.predict(X_last)
            else:
                logging.error(f"[{model_name}] Model does not have predict method")
                return {}
            
            # Create results dictionary
            results = {
                'dates': X_last.index.tolist() if hasattr(X_last.index, 'tolist') else list(range(len(X_last))),
                'real_values': y_last.tolist(),
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                'model_name': model_name,
                'n_predictions': len(predictions)
            }
            
            logging.info(f"[{model_name}] Last 20 days predictions: {len(predictions)} values")
            return results
            
        except Exception as e:
            logging.error(f"[{model_name}] Error generating last 20 days predictions: {e}")
            return {}

    def _generate_business_days_forecast(
        self,
        model: Any,
        last_20_days_results: Dict[str, Any],
        model_name: str,
        forecast_days: int = 20
    ) -> Dict[str, Any]:
        """Generate forecast for next N business days."""
        
        logging.info(f"[{model_name}] Generating {forecast_days} business days forecast")
        
        try:
            # Generate target business dates
            today = datetime.now().date()
            target_dates = self._generate_business_dates(today, forecast_days)
            
            # Try to load characteristics file for external features
            characteristics_file = self.processed_dir / "datos_economicos_filtrados.xlsx"
            
            if characteristics_file.exists():
                # Use characteristics file for prediction
                predictions = self._predict_from_characteristics_file(
                    model, characteristics_file, target_dates, model_name
                )
            else:
                # Use fallback method
                predictions = self._generate_fallback_predictions(
                    model, last_20_days_results, target_dates, model_name
                )
            
            results = {
                'target_dates': [d.strftime('%Y-%m-%d') for d in target_dates],
                'predictions': predictions,
                'model_name': model_name,
                'forecast_method': 'characteristics_file' if characteristics_file.exists() else 'fallback',
                'n_predictions': len(predictions)
            }
            
            logging.info(f"[{model_name}] Future forecast: {len(predictions)} predictions")
            return results
            
        except Exception as e:
            logging.error(f"[{model_name}] Error generating business days forecast: {e}")
            return {}

    def _predict_sequence_model(self, model: Any, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """Handle prediction for sequence models (LSTM, TTS)."""
        
        try:
            if model_name == 'LSTM':
                # LSTM expects 3D input: (samples, timesteps, features)
                if hasattr(model, 'sequence_length'):
                    seq_len = model.sequence_length
                else:
                    seq_len = 10  # Default
                
                # Create sequences
                sequences = []
                for i in range(seq_len, len(X) + 1):
                    seq = X.iloc[i-seq_len:i].values
                    sequences.append(seq)
                
                if sequences:
                    X_seq = np.array(sequences)
                    predictions = model.predict(X_seq)
                    if predictions.ndim > 1:
                        predictions = predictions.flatten()
                    return predictions
                else:
                    return np.zeros(len(X))
                    
            elif model_name == 'TTS':
                # TTS wrapper should handle this internally
                return model.predict(X)
            else:
                return model.predict(X)
                
        except Exception as e:
            logging.error(f"[{model_name}] Error in sequence prediction: {e}")
            return np.zeros(len(X))

    def _generate_business_dates(self, start_date: datetime.date, n_days: int) -> List[datetime.date]:
        """Generate N business days starting from start_date."""
        
        business_dates = []
        current_date = start_date
        
        while len(business_dates) < n_days:
            # Monday = 0, Sunday = 6
            if current_date.weekday() < 5:  # Monday to Friday
                business_dates.append(current_date)
            current_date += timedelta(days=1)
        
        return business_dates

    def _predict_from_characteristics_file(
        self,
        model: Any,
        characteristics_file: Path,
        target_dates: List[datetime.date],
        model_name: str
    ) -> List[float]:
        """Generate predictions using characteristics file."""
        
        try:
            # Load characteristics
            df_char = pd.read_excel(characteristics_file)
            df_char['date'] = pd.to_datetime(df_char['date']).dt.date
            
            predictions = []
            
            for target_date in target_dates:
                # Find matching row in characteristics file
                matching_rows = df_char[df_char['date'] == target_date]
                
                if not matching_rows.empty:
                    # Use the characteristics for prediction
                    row = matching_rows.iloc[0]
                    
                    # Prepare features (exclude date and target columns)
                    feature_cols = [col for col in df_char.columns 
                                  if col not in ['date', 'pricing_Target']]
                    features = row[feature_cols].values.reshape(1, -1)
                    
                    # Make prediction
                    if model_name in ['LSTM', 'TTS']:
                        pred = self._predict_sequence_single(model, features, model_name)
                    else:
                        pred = model.predict(features)[0]
                    
                    predictions.append(float(pred))
                else:
                    # No characteristics available, use fallback
                    predictions.append(0.0)
            
            return predictions
            
        except Exception as e:
            logging.error(f"[{model_name}] Error predicting from characteristics: {e}")
            return [0.0] * len(target_dates)

    def _predict_sequence_single(self, model: Any, features: np.ndarray, model_name: str) -> float:
        """Make single prediction for sequence models."""
        
        try:
            if model_name == 'LSTM':
                # For LSTM, we need to create a sequence
                # Use the features repeated to create a minimal sequence
                if hasattr(model, 'sequence_length'):
                    seq_len = model.sequence_length
                else:
                    seq_len = 10
                
                # Create sequence by repeating the features
                sequence = np.tile(features, (seq_len, 1))
                sequence = sequence.reshape(1, seq_len, -1)
                
                pred = model.predict(sequence)
                return float(pred.flatten()[0])
                
            elif model_name == 'TTS':
                # TTS wrapper should handle this
                pred = model.predict(features)
                return float(pred[0])
            else:
                pred = model.predict(features)
                return float(pred[0])
                
        except Exception as e:
            logging.error(f"[{model_name}] Error in single sequence prediction: {e}")
            return 0.0

    def _generate_fallback_predictions(
        self,
        model: Any,
        last_20_days_results: Dict[str, Any],
        target_dates: List[datetime.date],
        model_name: str
    ) -> List[float]:
        """Generate fallback predictions using trend from last 20 days."""
        
        try:
            # Get last predictions
            last_predictions = last_20_days_results.get('predictions', [])
            
            if not last_predictions:
                return [0.0] * len(target_dates)
            
            # Calculate simple trend
            if len(last_predictions) >= 5:
                recent_avg = np.mean(last_predictions[-5:])
                overall_avg = np.mean(last_predictions)
                trend = recent_avg - overall_avg
            else:
                recent_avg = np.mean(last_predictions)
                trend = 0.0
            
            # Generate predictions with trend
            predictions = []
            for i in range(len(target_dates)):
                pred_value = recent_avg + (trend * i * 0.1)  # Apply trend gradually
                predictions.append(float(pred_value))
            
            logging.info(f"[{model_name}] Fallback predictions generated with trend: {trend:.4f}")
            return predictions
            
        except Exception as e:
            logging.error(f"[{model_name}] Error generating fallback predictions: {e}")
            return [0.0] * len(target_dates)

    def _create_complete_forecast_dataframe(
        self,
        model_name: str,
        best_params: Dict[str, Any],
        X_all: pd.DataFrame,
        last_20_days_results: Dict[str, Any],
        future_forecast_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create complete forecast DataFrame matching step_7_0_train_models (1).py format."""
        
        try:
            forecast_data = []
            
            # Add last 20 days data (validation)
            if last_20_days_results:
                dates = last_20_days_results.get('dates', [])
                real_values = last_20_days_results.get('real_values', [])
                predictions = last_20_days_results.get('predictions', [])
                
                for i, (date, real_val, pred_val) in enumerate(zip(dates, real_values, predictions)):
                    forecast_data.append({
                        'date': date,
                        'Model': model_name,
                        'Valor_Real': real_val,
                        'Valor_Predicho': pred_val,
                        'Periodo': 'Validation_Last_20_Days',
                        'Tipo': 'Historical_Validation',
                        'Parametros_Modelo': str(best_params)
                    })
            
            # Add future forecast data
            if future_forecast_results:
                target_dates = future_forecast_results.get('target_dates', [])
                predictions = future_forecast_results.get('predictions', [])
                
                for date, pred_val in zip(target_dates, predictions):
                    forecast_data.append({
                        'date': date,
                        'Model': model_name,
                        'Valor_Real': np.nan,  # Future values are unknown
                        'Valor_Predicho': pred_val,
                        'Periodo': 'Forecast_Future_20_Days',
                        'Tipo': 'Future_Forecast',
                        'Parametros_Modelo': str(best_params)
                    })
            
            # Create DataFrame
            df_forecast = pd.DataFrame(forecast_data)
            
            if not df_forecast.empty:
                df_forecast['date'] = pd.to_datetime(df_forecast['date'])
                df_forecast = df_forecast.sort_values('date').reset_index(drop=True)
            
            logging.info(f"[{model_name}] Complete forecast DataFrame: {len(df_forecast)} rows")
            return df_forecast
            
        except Exception as e:
            logging.error(f"[{model_name}] Error creating complete forecast DataFrame: {e}")
            return pd.DataFrame()

    def _validate_and_save_forecast(self, df_forecast: pd.DataFrame, model_name: str) -> None:
        """Validate and save forecast results."""
        
        try:
            if df_forecast.empty:
                logging.warning(f"[{model_name}] Empty forecast DataFrame")
                return
            
            # Validate forecast
            validation_issues = []
            
            # Check for missing predictions
            missing_preds = df_forecast['Valor_Predicho'].isna().sum()
            if missing_preds > 0:
                validation_issues.append(f"Missing predictions: {missing_preds}")
            
            # Check for extreme values
            pred_values = df_forecast['Valor_Predicho'].dropna()
            if not pred_values.empty:
                if pred_values.abs().max() > 10:  # Extreme return values
                    validation_issues.append("Extreme prediction values detected")
            
            # Log validation results
            if validation_issues:
                for issue in validation_issues:
                    logging.warning(f"[{model_name}] Validation issue: {issue}")
            else:
                logging.info(f"[{model_name}] Forecast validation passed")
            
            # Save forecast
            output_file = self.results_dir / f"{model_name.lower()}_forecast_complete.csv"
            df_forecast.to_csv(output_file, index=False)
            
            logging.info(f"[{model_name}] Forecast saved: {output_file}")
            
        except Exception as e:
            logging.error(f"[{model_name}] Error validating/saving forecast: {e}") 
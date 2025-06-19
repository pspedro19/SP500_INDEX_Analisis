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
            
            # Step 5: Save individual model predictions CSV (for all_models_predictions.csv combination)
            self._save_individual_model_predictions(complete_forecast_df, model_name)
            
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
            
            # Extract last N days for target values
            X_last = X_all.tail(lag_days).copy()
            y_last = y_all.tail(lag_days).copy()
            
            # Generate predictions
            if hasattr(model, 'predict'):
                if model_name in ['LSTM', 'TTS']:
                    # CORRECCIÓN CRÍTICA PARA TTS: Proporcionar más contexto histórico
                    if model_name == 'TTS':
                        # Para TTS, necesitamos proporcionar suficientes datos históricos
                        sequence_length = getattr(model, 'sequence_length', 17)
                        required_context = sequence_length + lag_days
                        
                        if len(X_all) >= required_context:
                            # Usar datos históricos suficientes para crear secuencias
                            X_with_context = X_all.tail(required_context)
                            logging.info(f"[TTS] Usando {len(X_with_context)} muestras con contexto (sequence_length={sequence_length})")
                            
                            # Predecir usando el contexto completo
                            all_predictions = model.predict(X_with_context)
                            
                            # Extraer solo las últimas lag_days predicciones
                            predictions = all_predictions[-lag_days:] if len(all_predictions) >= lag_days else all_predictions
                            
                            logging.info(f"[TTS] Extraídas {len(predictions)} predicciones de los últimos {lag_days} días")
                        else:
                            # No hay suficiente contexto, usar método de respaldo
                            logging.warning(f"[TTS] Contexto insuficiente: {len(X_all)} < {required_context}")
                            predictions = self._predict_sequence_model(model, X_last, model_name)
                    else:
                        # Para LSTM, usar método existente
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
                # CORRECCIÓN CRÍTICA PARA TTS: No usar solo las últimas muestras
                # El TTS wrapper maneja las secuencias internamente, pero necesita suficientes datos
                
                # Verificar si tenemos suficientes datos para crear secuencias
                sequence_length = getattr(model, 'sequence_length', 17)
                
                if len(X) < sequence_length:
                    # Si no tenemos suficientes datos en X, necesitamos acceso a más datos históricos
                    # Por ahora, usar el predict del wrapper que maneja este caso
                    logging.warning(f"[TTS] Solo {len(X)} muestras disponibles, menos que sequence_length={sequence_length}")
                    # El TTSWrapper manejará esto con su lógica de fallback mejorada
                    return model.predict(X)
                else:
                    # Tenemos suficientes datos, usar predict normal
                    return model.predict(X)
            else:
                return model.predict(X)
                
        except Exception as e:
            logging.error(f"[{model_name}] Error in sequence prediction: {e}")
            return np.zeros(len(X))

    def _predict_sequence_single(self, model: Any, features: np.ndarray, model_name: str) -> float:
        """Make a single prediction with sequence models."""
        try:
            if model_name == 'TTS':
                # For TTS, use the model's predict method directly
                # Convert to DataFrame if needed
                if hasattr(features, 'shape') and len(features.shape) == 2:
                    features_df = pd.DataFrame(features)
                    pred = model.predict(features_df)
                    return float(pred[0]) if hasattr(pred, '__getitem__') else float(pred)
                else:
                    return 0.0
            elif model_name == 'LSTM':
                # For LSTM, might need sequence reshaping
                pred = model.predict(features)
                return float(pred[0]) if hasattr(pred, '__getitem__') else float(pred)
            else:
                # Regular prediction
                pred = model.predict(features)
                return float(pred[0]) if hasattr(pred, '__getitem__') else float(pred)
        except Exception as e:
            logging.error(f"[{model_name}] Error in single sequence prediction: {e}")
            return 0.0

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
            
            logging.info(f"[{model_name}] Loaded characteristics file with {len(df_char)} rows")
            logging.info(f"[{model_name}] Characteristics date range: {df_char['date'].min()} to {df_char['date'].max()}")
            logging.info(f"[{model_name}] Target dates: {target_dates[0]} to {target_dates[-1]}")
            
            predictions = []
            
            # Check if we have any matching dates
            target_dates_set = set(target_dates)
            char_dates_set = set(df_char['date'].tolist())
            matching_dates = target_dates_set.intersection(char_dates_set)
            
            if len(matching_dates) == 0:
                logging.warning(f"[{model_name}] No matching dates between target and characteristics file")
                # Use the last available row for all predictions
                if len(df_char) > 0:
                    last_row = df_char.iloc[-1]
                    feature_cols = [col for col in df_char.columns 
                                  if col not in ['date', 'pricing_Target']]
                    features = last_row[feature_cols].values.reshape(1, -1)
                    
                    for target_date in target_dates:
                        # Make prediction using last available characteristics
                        if model_name in ['LSTM', 'TTS']:
                            pred = self._predict_sequence_single(model, features, model_name)
                        else:
                            pred = model.predict(features)[0]
                        
                        predictions.append(float(pred))
                else:
                    # No data available, use fallback
                    predictions = [0.0] * len(target_dates)
            else:
                # We have some matching dates
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
                        # Use nearest available date
                        nearest_row = df_char.iloc[-1]  # Use last available
                        feature_cols = [col for col in df_char.columns 
                                      if col not in ['date', 'pricing_Target']]
                        features = nearest_row[feature_cols].values.reshape(1, -1)
                        
                        if model_name in ['LSTM', 'TTS']:
                            pred = self._predict_sequence_single(model, features, model_name)
                        else:
                            pred = model.predict(features)[0]
                        
                        predictions.append(float(pred))
            
            logging.info(f"[{model_name}] Generated {len(predictions)} predictions from characteristics file")
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
            
            # CRITICAL: Use ALL historical data, not just last 20 days
            # This ensures we have the same number of dates as the input Excel
            logging.info(f"[{model_name}] Creating predictions for ALL {len(X_all)} historical records")
            
            # Get predictions for ALL historical data
            if hasattr(X_all, 'index') and len(X_all) > 0:
                # Make predictions for all historical data
                if model_name in ['LSTM', 'TTS']:
                    # Handle sequence models - may need special handling
                    all_predictions = self._predict_sequence_model_all(X_all, model_name)
                else:
                    # Regular models
                    all_predictions = self.trained_model.predict(X_all) if hasattr(self, 'trained_model') else [0.0] * len(X_all)
                
                # Get corresponding real values and dates
                y_all = getattr(self, 'y_all', pd.Series([np.nan] * len(X_all), index=X_all.index))
                
                # Create records for ALL historical data
                for i in range(len(X_all)):
                    # Get the date - try different ways
                    if hasattr(X_all.index, 'to_list'):
                        date_val = X_all.index.to_list()[i]
                    elif hasattr(X_all, 'iloc'):
                        date_val = X_all.index[i] if len(X_all.index) > i else f"2024-01-{i+1:02d}"
                    else:
                        date_val = f"2024-01-{i+1:02d}"
                    
                    # Get real and predicted values
                    real_val = y_all.iloc[i] if i < len(y_all) else np.nan
                    pred_val = all_predictions[i] if i < len(all_predictions) else 0.0
                    
                    forecast_data.append({
                        'date': date_val,
                        'Model': model_name,
                        'Valor_Real': real_val,
                        'Valor_Predicho': pred_val,
                        'Periodo': 'Historical_Training',
                        'Tipo': 'Historical_Validation',
                        'Parametros_Modelo': str(best_params)
                    })
            
            # Add future forecast data (will be filtered out for individual files)
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
                df_forecast['date'] = pd.to_datetime(df_forecast['date'], errors='coerce')
                df_forecast = df_forecast.dropna(subset=['date'])  # Remove invalid dates
                df_forecast = df_forecast.sort_values('date').reset_index(drop=True)
            
            logging.info(f"[{model_name}] Complete forecast DataFrame: {len(df_forecast)} rows (including {len([x for x in forecast_data if x['Tipo'] == 'Historical_Validation'])} historical)")
            return df_forecast
            
        except Exception as e:
            logging.error(f"[{model_name}] Error creating complete forecast DataFrame: {e}")
            return pd.DataFrame()
            
    def _predict_sequence_model_all(self, X_all: pd.DataFrame, model_name: str) -> List[float]:
        """Generate predictions for all data using sequence models."""
        try:
            if hasattr(self, 'trained_model'):
                predictions = self.trained_model.predict(X_all)
                return predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
            else:
                logging.warning(f"[{model_name}] No trained model available, using zeros")
                return [0.0] * len(X_all)
        except Exception as e:
            logging.error(f"[{model_name}] Error in sequence prediction: {e}")
            return [0.0] * len(X_all)

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

    def _save_individual_model_predictions(self, df_forecast: pd.DataFrame, model_name: str) -> None:
        """Save individual model predictions in the format expected by _create_all_models_predictions_csv."""
        
        try:
            if df_forecast.empty:
                logging.warning(f"[{model_name}] Empty forecast DataFrame, skipping individual save")
                return
            
            # Create a copy for individual model format
            df_individual = df_forecast.copy()
            
            # CRITICAL: Filter out future predictions - only keep historical data
            # This ensures we match exactly the dates from the input Excel
            if 'Tipo' in df_individual.columns:
                # Only keep historical validation, exclude future forecasts
                df_individual = df_individual[df_individual['Tipo'] != 'Future_Forecast'].copy()
                logging.info(f"[{model_name}] Filtered out future forecasts, kept {len(df_individual)} historical records")
            
            # Ensure standard column names
            df_individual = df_individual.rename(columns={
                'Valor_Real': 'Valor_Real',
                'Valor_Predicho': 'Valor_Predicho',
                'Model': 'Model',
                'date': 'date'
            })
            
            # Ensure required columns exist
            required_columns = ['date', 'Model', 'Valor_Real', 'Valor_Predicho']
            for col in required_columns:
                if col not in df_individual.columns:
                    if col == 'Model':
                        df_individual['Model'] = model_name
                    elif col == 'Valor_Real':
                        df_individual['Valor_Real'] = np.nan
                    elif col == 'Valor_Predicho':
                        df_individual['Valor_Predicho'] = np.nan
            
            # Ensure date is in string format
            if 'date' in df_individual.columns:
                df_individual['date'] = pd.to_datetime(df_individual['date']).dt.strftime('%Y-%m-%d')
            
            # Save individual model file
            output_file = self.results_dir / f"s&p500_{model_name.lower()}_three_zones.csv"
            df_individual.to_csv(output_file, index=False)
            
            logging.info(f"[{model_name}] Individual predictions saved: {output_file}")
            logging.info(f"[{model_name}] Records saved: {len(df_individual)} (historical only)")
            
        except Exception as e:
            logging.error(f"[{model_name}] Error saving individual predictions: {e}") 
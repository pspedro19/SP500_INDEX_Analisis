"""
Metrics Service - Handles all metrics generation, fact tables, and inverse transforms
from step_7_0_train_models (1).py in a modular way.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from sp500_analysis.config.settings import settings


class MetricsService:
    """Service for generating metrics, fact tables, and inverse transforms."""

    def __init__(self):
        self.base_dir = settings.project_root
        self.results_dir = settings.results_dir
        self.processed_dir = settings.processed_dir

    def generate_fact_predictions_csv(
        self, 
        input_file: Optional[str] = None, 
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate FactPredictions CSV with EXACT structure from reference file.
        Structure: PrediccionKey,FechaKey,ModeloKey,MercadoKey,ValorReal,ValorPredicho,ErrorAbsoluto,ErrorPorcentual,TipoPeriodo,ZonaEntrenamiento,EsPrediccionFutura
        """
        
        if input_file is None:
            input_file = self.results_dir / 'all_models_predictions.csv'
        
        if output_file is None:
            output_file = self.results_dir / 'hechos_predicciones_fields.csv'
        
        try:
            logging.info(f"📊 Generating FactPredictions with EXACT structure from {input_file}")
            
            # Load predictions
            df = pd.read_csv(input_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Handle NaN values in Periodo by filling with 'Unknown'
            df['Periodo'] = df['Periodo'].fillna('Unknown')
            
            # Create model and market keys mapping
            modelo_mapping = {
                'CATBOOST': 1,
                'LIGHTGBM': 2, 
                'XGBOOST': 3,
                'MLP': 4,
                'SVM': 5,
                'LSTM': 6,
                'TTS': 7  # Adding TTS with key 7
            }
            
            mercado_mapping = {'S&P500': 1}
            
            # CRITICAL FIX: Load actual dates from reference files
            fecha_mapping = self._create_correct_date_mapping()
            
            # Create fact predictions with EXACT structure
            fact_predictions = []
            prediction_key = 1
            
            for _, row in df.iterrows():
                # Map periodo to exact reference format
                zona_entrenamiento = self._map_periodo_to_zona_exact(row.get('Periodo', 'Unknown'))
                tipo_periodo = self._map_periodo_to_tipo_exact(row.get('Periodo', 'Unknown'))
                
                # Determine if it's future prediction
                es_prediccion_futura = 'True' if row.get('Periodo') == 'Forecast' else 'False'
                
                # Get the correct FechaKey using real dates
                fecha_key = fecha_mapping.get(row['date'], len(fecha_mapping) + 1)
                
                # Extract actual values, handle NaN properly
                valor_real = row.get('Valor_Real')
                valor_predicho = row.get('Valor_Predicho')
                
                # Convert NaN to proper float values
                if pd.isna(valor_real):
                    valor_real = 0.0
                else:
                    valor_real = float(valor_real)
                    
                if pd.isna(valor_predicho):
                    valor_predicho = 0.0
                else:
                    valor_predicho = float(valor_predicho)
                
                fact_record = {
                    'PrediccionKey': prediction_key,
                    'FechaKey': fecha_key,
                    'ModeloKey': modelo_mapping.get(row['Model'], 1),
                    'MercadoKey': mercado_mapping.get(row.get('Tipo_Mercado', 'S&P500'), 1),
                    'ValorReal': valor_real,
                    'ValorPredicho': valor_predicho,
                    'ErrorAbsoluto': abs(valor_real - valor_predicho),
                    'ErrorPorcentual': self._calculate_error_porcentual(valor_real, valor_predicho),
                    'TipoPeriodo': tipo_periodo,
                    'ZonaEntrenamiento': zona_entrenamiento,
                    'EsPrediccionFutura': es_prediccion_futura
                }
                fact_predictions.append(fact_record)
                prediction_key += 1
            
            # Create DataFrame with exact column order
            df_fact = pd.DataFrame(fact_predictions)
            
            # Ensure exact column order from reference
            column_order = [
                'PrediccionKey', 'FechaKey', 'ModeloKey', 'MercadoKey', 
                'ValorReal', 'ValorPredicho', 'ErrorAbsoluto', 'ErrorPorcentual',
                'TipoPeriodo', 'ZonaEntrenamiento', 'EsPrediccionFutura'
            ]
            df_fact = df_fact[column_order]
            
            # Sort by ModeloKey, then by FechaKey for organized output
            df_fact = df_fact.sort_values(['ModeloKey', 'FechaKey']).reset_index(drop=True)
            
            # Reset PrediccionKey to be sequential after sorting
            df_fact['PrediccionKey'] = range(1, len(df_fact) + 1)
            
            # Save CSV
            df_fact.to_csv(output_file, index=False)
            logging.info(f"✅ FactPredictions saved with EXACT structure: {output_file}")
            logging.info(f"📊 Generated {len(df_fact)} predictions including TTS")
            
            # Log TTS confirmation
            tts_count = len(df_fact[df_fact['ModeloKey'] == 7])  # TTS has key 7
            logging.info(f"🎯 TTS CONFIRMED: {tts_count} predictions with ModeloKey=7")
            
            # Log date range information
            min_fecha_key = df_fact['FechaKey'].min()
            max_fecha_key = df_fact['FechaKey'].max()
            logging.info(f"📅 Date range: FechaKey {min_fecha_key} to {max_fecha_key}")
            
            return df_fact
            
        except Exception as e:
            logging.error(f"❌ Error generating fact predictions: {e}")
            return pd.DataFrame()

    def _create_correct_date_mapping(self) -> Dict[pd.Timestamp, int]:
        """
        Create the correct date mapping using the actual reference files:
        1. datos_economicos_1month_SP500_TRAINING_FPI.xlsx
        2. datos_economicos_filtrados.xlsx (últimos 20 días)
        """
        logging.info("📅 Creating correct date mapping from reference files")
        
        fecha_mapping = {}
        fecha_key = 1
        
        try:
            # Load main training file (FPI)
            training_fpi_file = self.base_dir / "data" / "3_trainingdata" / "datos_economicos_1month_SP500_TRAINING_FPI.xlsx"
            if training_fpi_file.exists():
                logging.info(f"📂 Loading dates from: {training_fpi_file}")
                df_fpi = pd.read_excel(training_fpi_file)
                if 'date' in df_fpi.columns:
                    df_fpi['date'] = pd.to_datetime(df_fpi['date'])
                    for date in sorted(df_fpi['date'].unique()):
                        fecha_mapping[date] = fecha_key
                        fecha_key += 1
                    logging.info(f"✅ Added {len(df_fpi)} dates from FPI file, range: {df_fpi['date'].min()} to {df_fpi['date'].max()}")
            
            # Load filtered file (últimos 20 días)
            filtered_file = self.base_dir / "data" / "3_trainingdata" / "datos_economicos_filtrados.xlsx"
            if filtered_file.exists():
                logging.info(f"📂 Loading dates from: {filtered_file}")
                df_filtered = pd.read_excel(filtered_file)
                if 'date' in df_filtered.columns:
                    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
                    # Add filtered dates only if they're not already in the mapping
                    new_dates = 0
                    for date in sorted(df_filtered['date'].unique()):
                        if date not in fecha_mapping:
                            fecha_mapping[date] = fecha_key
                            fecha_key += 1
                            new_dates += 1
                    logging.info(f"✅ Added {new_dates} new dates from filtered file, range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")
            
            logging.info(f"📊 Total unique dates mapped: {len(fecha_mapping)}")
            if fecha_mapping:
                all_dates = list(fecha_mapping.keys())
                logging.info(f"📅 Complete date range: {min(all_dates)} to {max(all_dates)}")
                
            return fecha_mapping
            
        except Exception as e:
            logging.error(f"❌ Error creating date mapping: {e}")
            # Fallback to simple sequential mapping
            logging.warning("⚠️ Using fallback sequential date mapping")
            return {}

    def generate_hechos_metricas_csv(
        self, 
        input_file: Optional[str] = None, 
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate HechosMetricas CSV with EXACT structure from reference file.
        Structure: MetricaKey,ModeloKey,MercadoKey,FechaEvaluacion,Periodo,RMSE,MAE,R2,SMAPE,Amplitud_Score,Fase_Score,Ultra_Metric,Hit_Direction,Hyperparametros,NumeroObservaciones,DuracionVentana,FechaInicioPeriodo,FechaFinPeriodo
        """
        
        if input_file is None:
            input_file = self.results_dir / 'all_models_predictions.csv'
        
        if output_file is None:
            output_file = self.results_dir / 'hechos_metricas_modelo.csv'
        
        try:
            logging.info(f"📈 Generating HechosMetricas with EXACT structure from {input_file}")
            
            # Load predictions
            df = pd.read_csv(input_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Create model and market keys mapping (handles trial formats)
            modelo_mapping = {
                'CATBOOST': 1,
                'LIGHTGBM': 2, 
                'XGBOOST': 3,
                'MLP': 4,
                'SVM': 5,
                'LSTM': 6,
                'TTS': 7  # Adding TTS with key 7
            }
            
            # Function to extract base model name from trial format
            def get_base_model_name(model_name: str) -> str:
                """Extract base model name from formats like 'CATBOOST_TRIAL5'"""
                if '_TRIAL' in model_name:
                    return model_name.split('_TRIAL')[0]
                elif '_' in model_name:
                    return model_name.split('_')[0]
                return model_name.upper()
            
            # Function to get model key
            def get_model_key(model_name: str) -> int:
                """Get model key for any model name format"""
                base_name = get_base_model_name(model_name)
                return modelo_mapping.get(base_name, 1)  # Default to 1 if not found
            
            mercado_mapping = {'S&P500': 1}
            
            # Handle NaN values in Periodo by filling with 'Unknown'
            df['Periodo'] = df['Periodo'].fillna('Unknown')
            
            # Debug: Log unique models found
            unique_models = df['Model'].unique()
            logging.info(f"📊 Models found in data: {list(unique_models)}")
            
            # Group by model and periodo for metrics calculation
            hechos_metricas = []
            metrica_key = 1
            fecha_evaluacion = datetime.now().strftime('%Y-%m-%d')
            
            for (modelo, periodo), group in df.groupby(['Model', 'Periodo']):
                
                # Skip if no valid data
                if len(group) == 0:
                    continue
                
                # Get base model name and key
                base_model_name = get_base_model_name(modelo)
                model_key = get_model_key(modelo)
                
                logging.info(f"🔄 Processing: {modelo} -> {base_model_name} (Key: {model_key}) | Periodo: {periodo}")
                
                # Map periodo to exact reference format
                periodo_mapped = self._map_periodo_for_metrics_exact(periodo)
                
                # Basic metrics
                n_predictions = len(group)
                real_values = pd.to_numeric(group['Valor_Real'], errors='coerce').dropna()
                pred_values = pd.to_numeric(group['Valor_Predicho'], errors='coerce').dropna()
                
                # Ensure we have matching pairs
                min_len = min(len(real_values), len(pred_values))
                if min_len > 0:
                    real_values = real_values.iloc[:min_len].values
                    pred_values = pred_values.iloc[:min_len].values
                    
                    # Calculate standard metrics
                    mse = np.mean((real_values - pred_values) ** 2)
                    mae = np.mean(np.abs(real_values - pred_values))
                    rmse = np.sqrt(mse)
                    r2 = self._calculate_r2(real_values, pred_values)
                    smape = self._calculate_smape(real_values, pred_values)
                    
                    # Hilbert Transform Metrics (same as original)
                    amplitud_score = self._calcular_amplitud_score(real_values, pred_values)
                    fase_score = self._calcular_fase_score(real_values, pred_values)
                    ultra_metric = self._calcular_ultra_metric(real_values, pred_values)
                    hit_direction = self._calcular_hit_direction(real_values, pred_values)
                    
                    # Get hyperparameters (try to extract from existing data or use default)
                    hyperparametros = self._get_model_hyperparameters(base_model_name, group)
                    
                    # Calculate duration and date range
                    fecha_inicio = group['date'].min().strftime('%Y-%m-%d')
                    fecha_fin = group['date'].max().strftime('%Y-%m-%d')
                    duracion_ventana = (group['date'].max() - group['date'].min()).days
                    
                else:
                    # Default values for empty data
                    rmse = mae = r2 = smape = np.nan
                    amplitud_score = fase_score = ultra_metric = hit_direction = np.nan
                    hyperparametros = "{}"
                    fecha_inicio = fecha_fin = ""
                    duracion_ventana = 0
                
                # Create metrics record with EXACT structure
                metrica_record = {
                    'MetricaKey': metrica_key,
                    'ModeloKey': model_key,  # Use calculated model key
                    'MercadoKey': mercado_mapping.get('S&P500', 1),
                    'FechaEvaluacion': fecha_evaluacion,
                    'Periodo': periodo_mapped,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'SMAPE': smape,
                    'Amplitud_Score': amplitud_score,
                    'Fase_Score': fase_score,
                    'Ultra_Metric': ultra_metric,
                    'Hit_Direction': hit_direction,
                    'Hyperparametros': hyperparametros,
                    'NumeroObservaciones': n_predictions,
                    'DuracionVentana': duracion_ventana,
                    'FechaInicioPeriodo': fecha_inicio,
                    'FechaFinPeriodo': fecha_fin
                }
                hechos_metricas.append(metrica_record)
                metrica_key += 1
            
            # Create DataFrame with exact column order
            df_hechos = pd.DataFrame(hechos_metricas)
            
            # Ensure exact column order from reference
            column_order = [
                'MetricaKey', 'ModeloKey', 'MercadoKey', 'FechaEvaluacion', 'Periodo',
                'RMSE', 'MAE', 'R2', 'SMAPE', 'Amplitud_Score', 'Fase_Score', 
                'Ultra_Metric', 'Hit_Direction', 'Hyperparametros', 'NumeroObservaciones',
                'DuracionVentana', 'FechaInicioPeriodo', 'FechaFinPeriodo'
            ]
            df_hechos = df_hechos[column_order]
            
            # Sort by ModeloKey, then by Periodo for organized output
            df_hechos = df_hechos.sort_values(['ModeloKey', 'Periodo']).reset_index(drop=True)
            
            # Reset MetricaKey to be sequential after sorting
            df_hechos['MetricaKey'] = range(1, len(df_hechos) + 1)
            
            # Save CSV
            df_hechos.to_csv(output_file, index=False)
            logging.info(f"✅ HechosMetricas saved with EXACT structure: {output_file}")
            logging.info(f"📈 Generated {len(df_hechos)} metrics records")
            
            # Log model confirmation
            unique_model_keys = df_hechos['ModeloKey'].unique()
            logging.info(f"🎯 Model Keys found: {sorted(unique_model_keys)}")
            
            # Log specific model counts
            for key, name in modelo_mapping.items():
                count = len(df_hechos[df_hechos['ModeloKey'] == name])
                if count > 0:
                    logging.info(f"📊 {key} (Key {name}): {count} metrics records")
            
            return df_hechos
            
        except Exception as e:
            logging.error(f"❌ Error generating hechos metricas: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return pd.DataFrame()

    # New mapping methods for exact structure
    def _map_periodo_to_zona_exact(self, periodo: str) -> str:
        """Map periodo to exact zona format from reference."""
        mapping = {
            'Training': 'Zona A',
            'Validation': 'Zona B', 
            'Test': 'Zona C',
            'Forecast': 'Zona D',
            'Unknown': 'Zona A',  # Treat unknown periods as Zona A
            'Evaluacion': 'Zona B'  # Map existing periods
        }
        return mapping.get(periodo, 'Zona A')
    
    def _map_periodo_to_tipo_exact(self, periodo: str) -> str:
        """Map periodo to exact tipo format from reference."""
        mapping = {
            'Training': 'Training',
            'Validation': 'Back-test', 
            'Test': 'Back-test',
            'Forecast': 'Forecast',
            'Unknown': 'Training',  # Treat unknown periods as Training
            'Evaluacion': 'Back-test'  # Map existing periods
        }
        return mapping.get(periodo, 'Training')
    
    def _map_periodo_for_metrics_exact(self, periodo: str) -> str:
        """Map periodo for metrics with exact format from reference."""
        mapping = {
            'Training': 'Training',
            'Validation': 'Back-test', 
            'Test': 'Back-test',
            'Forecast': 'Forecast',
            'Unknown': 'Training',  # Treat unknown periods as Training
            'Evaluacion': 'Back-test'  # Map existing periods
        }
        return mapping.get(periodo, 'Training')
    
    def _get_model_hyperparameters(self, modelo: str, group: pd.DataFrame) -> str:
        """Get hyperparameters for model in JSON format."""
        # Try to get from existing data
        if 'Hyperparámetros' in group.columns:
            hyper_str = group['Hyperparámetros'].iloc[0]
            if pd.notna(hyper_str) and hyper_str != '':
                return hyper_str
        
        # Default hyperparameters for each model
        default_params = {
            'CATBOOST': '{"learning_rate": 0.01, "depth": 6, "iterations": 500}',
            'LIGHTGBM': '{"learning_rate": 0.01, "num_leaves": 31, "n_estimators": 500}',
            'XGBOOST': '{"learning_rate": 0.01, "max_depth": 6, "n_estimators": 500}',
            'MLP': '{"hidden_layer_sizes": [100], "learning_rate_init": 0.001}',
            'SVM': '{"C": 1.0, "epsilon": 0.1, "kernel": "rbf"}',
            'LSTM': '{"units": 50, "dropout_rate": 0.2, "sequence_length": 10}',
            'TTS': '{"d_model": 64, "nhead": 8, "num_encoder_layers": 3, "sequence_length": 15}'
        }
        return default_params.get(modelo, '{}')
    
    def _calculate_smape(self, real: np.ndarray, pred: np.ndarray) -> float:
        """Calculate SMAPE (Symmetric Mean Absolute Percentage Error)."""
        try:
            if len(real) == 0 or len(pred) == 0:
                return np.nan
            
            denominator = (np.abs(real) + np.abs(pred)) / 2
            mask = denominator != 0
            
            if np.sum(mask) == 0:
                return np.nan
            
            smape = np.mean(np.abs(real[mask] - pred[mask]) / denominator[mask]) * 100
            return smape
            
        except Exception:
            return np.nan

    def generate_modelo_dimension_table(
        self, 
        df_hechos_metricas: pd.DataFrame, 
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate dimension table for models."""
        
        if output_file is None:
            output_file = self.results_dir / 'dim_modelo.csv'
        
        try:
            # Create dimension table
            modelos_unicos = df_hechos_metricas['Modelo'].unique()
            
            dim_modelo = []
            for modelo in modelos_unicos:
                dim_record = {
                    'ID_Modelo': modelo,
                    'Nombre_Modelo': modelo,
                    'Categoria': self._get_model_category(modelo),
                    'Tipo_Algoritmo': self._get_algorithm_type(modelo),
                    'Requiere_Escalado': self._requires_scaling(modelo),
                    'Fecha_Creacion': datetime.now()
                }
                dim_modelo.append(dim_record)
            
            df_dim = pd.DataFrame(dim_modelo)
            df_dim.to_csv(output_file, index=False)
            
            logging.info(f"✅ Dimension table saved: {output_file}")
            return df_dim
            
        except Exception as e:
            logging.error(f"❌ Error generating dimension table: {e}")
            return pd.DataFrame()

    def apply_inverse_transform_to_predictions(
        self,
        predictions_file: str,
        original_data_file: Optional[str] = None,
        target_column: str = "pricing_Target",
        lag_days: int = 20
    ) -> Optional[pd.DataFrame]:
        """
        Apply inverse transform to predictions using original prices.
        Matches step_7_0_train_models (1).py logic.
        """
        
        logging.info("🔄 Applying inverse transform to predictions...")
        
        if original_data_file is None:
            original_data_file = self.processed_dir / 'datos_economicos_1month_SP500_TRAINING.xlsx'
        
        try:
            # Load predictions
            df_predictions = pd.read_csv(predictions_file)
            df_predictions['date'] = pd.to_datetime(df_predictions['date'])
            
            # Load original data with prices
            df_original = pd.read_excel(original_data_file)
            df_original['date'] = pd.to_datetime(df_original['date'])
            df_original = df_original.sort_values('date').reset_index(drop=True)
            
            # Find price column
            if target_column not in df_original.columns:
                price_columns = [col for col in df_original.columns if 
                               any(keyword in col.lower() for keyword in ['pricing_target', 'close', 'price'])]
                if price_columns:
                    target_column = price_columns[0]
                    logging.info(f"✅ Using price column: {target_column}")
                else:
                    logging.error("❌ No price column found")
                    return None
            
            # Create price mapping
            price_mapping = dict(zip(df_original['date'], df_original[target_column]))
            
            # Apply inverse transform
            def get_base_price(prediction_date, lag_days):
                base_date = prediction_date - pd.Timedelta(days=lag_days)
                
                # Try exact date first
                if base_date in price_mapping:
                    return price_mapping[base_date]
                
                # Try nearby dates
                for offset in range(1, 6):
                    for direction in [-1, 1]:
                        alt_date = base_date + pd.Timedelta(days=direction * offset)
                        if alt_date in price_mapping:
                            return price_mapping[alt_date]
                return None
            
            # Calculate inverse transformed values
            valor_real_inv = []
            valor_predicho_inv = []
            base_prices_used = []
            
            for _, row in df_predictions.iterrows():
                base_price = get_base_price(row['date'], lag_days)
                
                if base_price is not None and not pd.isna(base_price):
                    # Inverse transform: price[t+lag] = price[t] * (1 + return)
                    real_inv = base_price * (1 + row['Valor_Real']) if not pd.isna(row['Valor_Real']) else np.nan
                    pred_inv = base_price * (1 + row['Valor_Predicho']) if not pd.isna(row['Valor_Predicho']) else np.nan
                    
                    valor_real_inv.append(real_inv)
                    valor_predicho_inv.append(pred_inv)
                    base_prices_used.append(base_price)
                else:
                    valor_real_inv.append(np.nan)
                    valor_predicho_inv.append(np.nan)
                    base_prices_used.append(np.nan)
            
            # Add inverse transform columns
            df_predictions['Valor_Real_Inv'] = valor_real_inv
            df_predictions['Valor_Predicho_Inv'] = valor_predicho_inv
            df_predictions['Precio_Base_Usado'] = base_prices_used
            
            # Save results
            output_file = self.results_dir / "predictions_with_inverse_transform.csv"
            df_predictions.to_csv(output_file, index=False)
            
            logging.info(f"✅ Inverse transform applied and saved: {output_file}")
            return df_predictions
            
        except Exception as e:
            logging.error(f"❌ Error applying inverse transform: {e}")
            return None

    def generate_inverse_transform_plots(self, df_with_inverse: pd.DataFrame, asset_name: str = "S&P500") -> None:
        """Generate plots for inverse transformed predictions."""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{asset_name} - Inverse Transform Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Original returns vs predictions
            axes[0, 0].scatter(df_with_inverse['Valor_Real'], df_with_inverse['Valor_Predicho'], alpha=0.6)
            axes[0, 0].plot([df_with_inverse['Valor_Real'].min(), df_with_inverse['Valor_Real'].max()],
                           [df_with_inverse['Valor_Real'].min(), df_with_inverse['Valor_Real'].max()], 'r--')
            axes[0, 0].set_xlabel('Real Returns')
            axes[0, 0].set_ylabel('Predicted Returns')
            axes[0, 0].set_title('Returns: Real vs Predicted')
            
            # Plot 2: Inverse transformed prices vs predictions
            valid_inv = df_with_inverse.dropna(subset=['Valor_Real_Inv', 'Valor_Predicho_Inv'])
            if not valid_inv.empty:
                axes[0, 1].scatter(valid_inv['Valor_Real_Inv'], valid_inv['Valor_Predicho_Inv'], alpha=0.6)
                axes[0, 1].plot([valid_inv['Valor_Real_Inv'].min(), valid_inv['Valor_Real_Inv'].max()],
                               [valid_inv['Valor_Real_Inv'].min(), valid_inv['Valor_Real_Inv'].max()], 'r--')
                axes[0, 1].set_xlabel('Real Prices')
                axes[0, 1].set_ylabel('Predicted Prices')
                axes[0, 1].set_title('Prices: Real vs Predicted (Inverse Transform)')
            
            # Plot 3: Time series of inverse transformed prices
            df_sorted = df_with_inverse.sort_values('date')
            axes[1, 0].plot(df_sorted['date'], df_sorted['Valor_Real_Inv'], label='Real Prices', alpha=0.7)
            axes[1, 0].plot(df_sorted['date'], df_sorted['Valor_Predicho_Inv'], label='Predicted Prices', alpha=0.7)
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Price')
            axes[1, 0].set_title('Price Time Series')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Prediction errors distribution
            if not valid_inv.empty:
                errors = valid_inv['Valor_Real_Inv'] - valid_inv['Valor_Predicho_Inv']
                axes[1, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
                axes[1, 1].set_xlabel('Prediction Error (Real - Predicted)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Distribution of Prediction Errors')
                axes[1, 1].axvline(errors.mean(), color='red', linestyle='--', label=f'Mean: {errors.mean():.2f}')
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.results_dir / "inverse_transform_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"✅ Inverse transform plots saved: {plot_file}")
            
        except Exception as e:
            logging.error(f"❌ Error generating inverse transform plots: {e}")

    # Hilbert Transform Metrics (from step_7_0_train_models (1).py)
    
    def _calcular_amplitud_score(self, real: np.ndarray, pred: np.ndarray) -> float:
        """Calculate amplitude score using Hilbert transform."""
        try:
            if len(real) != len(pred) or len(real) < 5:
                return np.nan
            
            # Apply Hilbert transform
            analytic_real = hilbert(real)
            analytic_pred = hilbert(pred)
            
            # Get amplitude envelopes
            amplitude_real = np.abs(analytic_real)
            amplitude_pred = np.abs(analytic_pred)
            
            # Calculate correlation between amplitudes
            correlation = np.corrcoef(amplitude_real, amplitude_pred)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return np.nan

    def _calcular_fase_score(self, real: np.ndarray, pred: np.ndarray) -> float:
        """Calculate phase score using Hilbert transform."""
        try:
            if len(real) != len(pred) or len(real) < 5:
                return np.nan
            
            # Apply Hilbert transform
            analytic_real = hilbert(real)
            analytic_pred = hilbert(pred)
            
            # Get instantaneous phases
            phase_real = np.angle(analytic_real)
            phase_pred = np.angle(analytic_pred)
            
            # Calculate phase difference
            phase_diff = np.abs(phase_real - phase_pred)
            
            # Normalize to [0, 1] where 1 is perfect phase alignment
            normalized_score = 1 - np.mean(phase_diff) / np.pi
            return max(0, normalized_score)
            
        except Exception:
            return np.nan

    def _calcular_ultra_metric(self, real: np.ndarray, pred: np.ndarray, w1: float = 0.5, w2: float = 0.3, w3: float = 0.2) -> float:
        """Calculate ultra metric combining multiple scores."""
        try:
            if len(real) != len(pred) or len(real) < 5:
                return np.nan
            
            # Component scores
            amplitude_score = self._calcular_amplitud_score(real, pred)
            fase_score = self._calcular_fase_score(real, pred)
            hit_direction = self._calcular_hit_direction(real, pred)
            
            # Weighted combination
            ultra_metric = (w1 * amplitude_score + w2 * fase_score + w3 * hit_direction)
            return ultra_metric if not np.isnan(ultra_metric) else 0.0
            
        except Exception:
            return np.nan

    def _calcular_hit_direction(self, real: np.ndarray, pred: np.ndarray) -> float:
        """Calculate directional hit rate."""
        try:
            if len(real) < 2 or len(pred) < 2:
                return np.nan
            
            # Calculate directional changes
            real_direction = np.sign(np.diff(real))
            pred_direction = np.sign(np.diff(pred))
            
            # Calculate hit rate
            hits = np.sum(real_direction == pred_direction)
            total = len(real_direction)
            
            return hits / total if total > 0 else 0.0
            
        except Exception:
            return np.nan

    # Helper methods
    
    def _calculate_error_porcentual(self, real: float, predicho: float) -> float:
        """Calculate percentage error."""
        try:
            if pd.isna(real) or pd.isna(predicho) or real == 0:
                return 0.0
            return abs((real - predicho) / real) * 100
        except:
            return 0.0

    def _map_periodo_to_zona(self, periodo: str) -> str:
        """Map periodo to zona classification."""
        mapping = {
            'Training': 'Zona_A_Entrenamiento',
            'Validation': 'Zona_B_Validacion', 
            'Test': 'Zona_C_Prueba',
            'Forecast': 'Zona_D_Pronostico'
        }
        return mapping.get(periodo, 'Zona_Desconocida')

    def _map_periodo_for_metrics(self, periodo: str) -> str:
        """Map periodo for metrics calculation."""
        return self._map_periodo_to_zona(periodo)

    def _calculate_window_duration(self, dates_series: pd.Series) -> int:
        """Calculate duration of time window in days."""
        return (dates_series.max() - dates_series.min()).days

    def _calculate_r2(self, real: np.ndarray, pred: np.ndarray) -> float:
        """Calculate R-squared."""
        try:
            if len(real) == 0 or len(pred) == 0:
                return np.nan
            
            ss_res = np.sum((real - pred) ** 2)
            ss_tot = np.sum((real - np.mean(real)) ** 2)
            
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
            
        except Exception:
            return np.nan

    def _get_model_category(self, modelo: str) -> str:
        """Get model category."""
        categories = {
            'CatBoost': 'Gradient_Boosting',
            'LightGBM': 'Gradient_Boosting',
            'XGBoost': 'Gradient_Boosting',
            'MLP': 'Neural_Network',
            'SVM': 'Support_Vector_Machine',
            'LSTM': 'Deep_Learning',
            'TTS': 'Transformer'
        }
        return categories.get(modelo, 'Unknown')

    def _get_algorithm_type(self, modelo: str) -> str:
        """Get algorithm type."""
        types = {
            'CatBoost': 'Tree_Based',
            'LightGBM': 'Tree_Based',
            'XGBoost': 'Tree_Based',
            'MLP': 'Neural_Network',
            'SVM': 'Kernel_Method',
            'LSTM': 'Recurrent_Neural_Network',
            'TTS': 'Transformer_Neural_Network'
        }
        return types.get(modelo, 'Unknown')

    def _requires_scaling(self, modelo: str) -> bool:
        """Check if model requires feature scaling."""
        scaling_required = {
            'CatBoost': False,
            'LightGBM': False,
            'XGBoost': False,
            'MLP': True,
            'SVM': True,
            'LSTM': True,
            'TTS': True
        }
        return scaling_required.get(modelo, False) 
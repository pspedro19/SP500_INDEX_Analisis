"""
Metrics Service - Handles all metrics generation, fact tables, and inverse transforms
from step_7_0_train_models (1).py in a modular way.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import os
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
        Generate FactPredictions CSV (WITHOUT Hilbert metrics).
        Matches step_7_0_train_models (1).py logic.
        """
        
        if input_file is None:
            input_file = self.results_dir / 'all_models_predictions.csv'
        
        if output_file is None:
            output_file = self.results_dir / 'hechos_predicciones_fields.csv'
        
        try:
            logging.info(f"📊 Generating FactPredictions from {input_file}")
            
            # Load predictions
            df = pd.read_csv(input_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Create fact predictions structure
            fact_predictions = []
            
            for _, row in df.iterrows():
                fact_record = {
                    'ID_Prediccion': f"{row['Model']}_{row['date'].strftime('%Y%m%d')}",
                    'Fecha': row['date'],
                    'Modelo': row['Model'],
                    'Valor_Real': row.get('Valor_Real', np.nan),
                    'Valor_Predicho': row.get('Valor_Predicho', np.nan),
                    'Error_Absoluto': abs(row.get('Valor_Real', 0) - row.get('Valor_Predicho', 0)),
                    'Error_Porcentual': self._calculate_error_porcentual(
                        row.get('Valor_Real', 0), 
                        row.get('Valor_Predicho', 0)
                    ),
                    'Periodo': self._map_periodo_to_zona(row.get('Periodo', 'Unknown')),
                    'Timestamp_Creacion': datetime.now()
                }
                fact_predictions.append(fact_record)
            
            # Create DataFrame
            df_fact = pd.DataFrame(fact_predictions)
            
            # Save CSV
            df_fact.to_csv(output_file, index=False)
            logging.info(f"✅ FactPredictions saved: {output_file}")
            
            return df_fact
            
        except Exception as e:
            logging.error(f"❌ Error generating fact predictions: {e}")
            return pd.DataFrame()

    def generate_hechos_metricas_csv(
        self, 
        input_file: Optional[str] = None, 
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate HechosMetricas CSV (WITH Hilbert metrics).
        Matches step_7_0_train_models (1).py logic.
        """
        
        if input_file is None:
            input_file = self.results_dir / 'all_models_predictions.csv'
        
        if output_file is None:
            output_file = self.results_dir / 'hechos_metricas_modelo.csv'
        
        try:
            logging.info(f"📈 Generating HechosMetricas (WITH Hilbert) from {input_file}")
            
            # Load predictions
            df = pd.read_csv(input_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Group by model and periodo for metrics calculation
            hechos_metricas = []
            
            for (modelo, periodo), group in df.groupby(['Model', 'Periodo']):
                
                # Basic metrics
                n_predictions = len(group)
                mse = np.mean((group['Valor_Real'] - group['Valor_Predicho']) ** 2)
                mae = np.mean(np.abs(group['Valor_Real'] - group['Valor_Predicho']))
                rmse = np.sqrt(mse)
                
                # Hilbert Transform Metrics
                real_values = group['Valor_Real'].dropna().values
                pred_values = group['Valor_Predicho'].dropna().values
                
                if len(real_values) > 0 and len(pred_values) > 0:
                    amplitud_score = self._calcular_amplitud_score(real_values, pred_values)
                    fase_score = self._calcular_fase_score(real_values, pred_values)
                    ultra_metric = self._calcular_ultra_metric(real_values, pred_values)
                    hit_direction = self._calcular_hit_direction(real_values, pred_values)
                else:
                    amplitud_score = fase_score = ultra_metric = hit_direction = np.nan
                
                # Create metrics record
                metrica_record = {
                    'ID_Metrica': f"{modelo}_{periodo}_{datetime.now().strftime('%Y%m%d')}",
                    'Modelo': modelo,
                    'Periodo': self._map_periodo_for_metrics(periodo),
                    'Fecha_Inicio': group['date'].min(),
                    'Fecha_Fin': group['date'].max(),
                    'Duracion_Ventana': self._calculate_window_duration(group['date']),
                    'N_Predicciones': n_predictions,
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': self._calculate_r2(real_values, pred_values),
                    'Amplitud_Score': amplitud_score,
                    'Fase_Score': fase_score,
                    'Ultra_Metric': ultra_metric,
                    'Hit_Direction': hit_direction,
                    'Timestamp_Calculo': datetime.now()
                }
                hechos_metricas.append(metrica_record)
            
            # Create DataFrame
            df_hechos = pd.DataFrame(hechos_metricas)
            
            # Save CSV
            df_hechos.to_csv(output_file, index=False)
            logging.info(f"✅ HechosMetricas saved: {output_file}")
            
            return df_hechos
            
        except Exception as e:
            logging.error(f"❌ Error generating hechos metricas: {e}")
            return pd.DataFrame()

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
        if pd.isna(real) or pd.isna(predicho) or real == 0:
            return np.nan
        return abs((real - predicho) / real) * 100

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
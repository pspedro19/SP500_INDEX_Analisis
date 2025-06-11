#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble de Modelos ARIMA/SARIMAX
=================================
Combina las predicciones de los distintos modelos (ARIMA, SARIMAX_BASIC, SARIMAX_EXTENDED)
para cada instrumento, utilizando ponderación basada en métricas de rendimiento.

Salidas:
- Predicciones de ensemble para cada instrumento
- Métricas comparativas entre modelos individuales y ensemble
- Archivo consolidado con todas las predicciones
- Visualizaciones mejoradas (básicas y avanzadas)
"""

import os
import json
import logging
from sp500_analysis.shared.logging.logger import configurar_logging, setup_logging, get_logger
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Importar módulos de utilidades si están disponibles
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
    from time_series.utils import plots
except ImportError:
    # Intentar con rutas alternativas durante desarrollo
    try:
        sys.path.append(os.path.abspath("../pipelines"))
        from time_series.utils import plots
    except ImportError:
        pass

# Configuración de logging


# Definir constantes del proyecto
class Config:
    PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    TRAIN_DATA_DIR = DATA_DIR / "2_trainingdata_ts"
    RESULTS_DIR = DATA_DIR / "4_results"

    INSTRUMENTS = ["SP500", "EURUSD", "USDJPY"]
    MODEL_TYPES = ["ARIMA", "SARIMAX_BASIC", "SARIMAX_EXTENDED"]

    # Configuración de ensemble
    WEIGHTING_METHOD = "inverse_rmse"  # Opciones: "equal", "inverse_rmse", "inverse_mse", "custom"

    # Pesos personalizados (solo si WEIGHTING_METHOD = "custom")
    CUSTOM_WEIGHTS = {
        "SP500": {"ARIMA": 0.2, "SARIMAX_BASIC": 0.3, "SARIMAX_EXTENDED": 0.5},
        "EURUSD": {"ARIMA": 0.3, "SARIMAX_BASIC": 0.3, "SARIMAX_EXTENDED": 0.4},
        "USDJPY": {"ARIMA": 0.3, "SARIMAX_BASIC": 0.3, "SARIMAX_EXTENDED": 0.4},
    }


def load_metrics(instrument: str, model_type: str, config: Config) -> Dict:
    """Carga las métricas de un modelo específico."""
    logger = logging.getLogger(__name__)

    metrics_file = config.TRAIN_DATA_DIR / instrument / f"{instrument}_{model_type}_metrics.json"

    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        logger.info(f"Métricas cargadas para {instrument}_{model_type}")
        return metrics
    except Exception as e:
        logger.error(f"Error cargando métricas para {instrument}_{model_type}: {str(e)}")
        return {"metrics": {"rmse": float('inf'), "mae": float('inf')}}


def load_forecasts(instrument: str, model_type: str, config: Config) -> pd.DataFrame:
    """Carga las predicciones de un modelo específico."""
    logger = logging.getLogger(__name__)

    forecast_file = config.TRAIN_DATA_DIR / instrument / f"{instrument}_{model_type}_forecast.csv"

    try:
        forecast_df = pd.read_csv(forecast_file)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        forecast_df = forecast_df.set_index('date')
        logger.info(f"Predicciones cargadas para {instrument}_{model_type}: {len(forecast_df)} filas")
        return forecast_df
    except Exception as e:
        logger.error(f"Error cargando predicciones para {instrument}_{model_type}: {str(e)}")
        return pd.DataFrame()


def load_all_predictions(config: Config) -> pd.DataFrame:
    """Carga el archivo consolidado con todas las predicciones."""
    logger = logging.getLogger(__name__)

    all_predictions_file = config.TRAIN_DATA_DIR / "all_models_predictions.csv"

    try:
        all_predictions = pd.read_csv(all_predictions_file)
        all_predictions['date'] = pd.to_datetime(all_predictions['date'])
        logger.info(f"Archivo consolidado cargado: {len(all_predictions)} filas")
        return all_predictions
    except Exception as e:
        logger.error(f"Error cargando archivo consolidado: {str(e)}")
        return pd.DataFrame()


def calculate_ensemble_weights(instrument: str, config: Config) -> Dict[str, float]:
    """Calcula los pesos para el ensemble según el método configurado."""
    logger = logging.getLogger(__name__)

    # Cargar métricas de todos los modelos para este instrumento
    metrics = {}
    for model_type in config.MODEL_TYPES:
        model_metrics = load_metrics(instrument, model_type, config)
        if "metrics" in model_metrics and "rmse" in model_metrics["metrics"]:
            metrics[model_type] = model_metrics["metrics"]["rmse"]
        else:
            # Si no hay RMSE, establecer un valor alto para minimizar la influencia
            metrics[model_type] = float('inf')
            logger.warning(f"No se encontró RMSE para {instrument}_{model_type}, usando peso mínimo")

    # Calcular pesos según el método seleccionado
    weights = {}

    if config.WEIGHTING_METHOD == "equal":
        # Pesos iguales para todos los modelos
        num_models = len(config.MODEL_TYPES)
        for model_type in config.MODEL_TYPES:
            weights[model_type] = 1.0 / num_models

    elif config.WEIGHTING_METHOD == "inverse_rmse":
        # Pesos proporcionales al inverso del RMSE
        sum_inverse_rmse = 0
        inverse_rmse = {}

        for model_type, rmse in metrics.items():
            if rmse == float('inf') or rmse <= 0:
                inverse_rmse[model_type] = 0.0
            else:
                inverse_rmse[model_type] = 1.0 / rmse
                sum_inverse_rmse += inverse_rmse[model_type]

        # Normalizar pesos
        if sum_inverse_rmse > 0:
            for model_type in config.MODEL_TYPES:
                weights[model_type] = inverse_rmse.get(model_type, 0.0) / sum_inverse_rmse
        else:
            # Fallback a pesos iguales si hay problemas
            logger.warning(f"No se pueden calcular pesos por inverse_rmse para {instrument}, usando pesos iguales")
            for model_type in config.MODEL_TYPES:
                weights[model_type] = 1.0 / len(config.MODEL_TYPES)

    elif config.WEIGHTING_METHOD == "inverse_mse":
        # Pesos proporcionales al inverso del MSE (RMSE^2)
        sum_inverse_mse = 0
        inverse_mse = {}

        for model_type, rmse in metrics.items():
            if rmse == float('inf') or rmse <= 0:
                inverse_mse[model_type] = 0.0
            else:
                mse = rmse**2
                inverse_mse[model_type] = 1.0 / mse
                sum_inverse_mse += inverse_mse[model_type]

        # Normalizar pesos
        if sum_inverse_mse > 0:
            for model_type in config.MODEL_TYPES:
                weights[model_type] = inverse_mse.get(model_type, 0.0) / sum_inverse_mse
        else:
            logger.warning(f"No se pueden calcular pesos por inverse_mse para {instrument}, usando pesos iguales")
            for model_type in config.MODEL_TYPES:
                weights[model_type] = 1.0 / len(config.MODEL_TYPES)

    elif config.WEIGHTING_METHOD == "custom":
        # Usar pesos personalizados del archivo de configuración
        if instrument in config.CUSTOM_WEIGHTS:
            weights = config.CUSTOM_WEIGHTS[instrument]
        else:
            logger.warning(f"No hay pesos personalizados para {instrument}, usando pesos iguales")
            for model_type in config.MODEL_TYPES:
                weights[model_type] = 1.0 / len(config.MODEL_TYPES)

    # Garantizar que hay un peso para cada modelo y que suman 1
    for model_type in config.MODEL_TYPES:
        if model_type not in weights:
            weights[model_type] = 0.0

    # Normalizar
    sum_weights = sum(weights.values())
    if sum_weights > 0:
        for model_type in weights:
            weights[model_type] /= sum_weights

    logger.info(f"Pesos calculados para {instrument}: {weights}")
    return weights


def create_ensemble_predictions(instrument: str, config: Config) -> pd.DataFrame:
    """Crea predicciones de ensemble para un instrumento específico."""
    logger = logging.getLogger(__name__)

    # Cargar las predicciones de todos los modelos
    forecasts = {}
    for model_type in config.MODEL_TYPES:
        forecasts[model_type] = load_forecasts(instrument, model_type, config)

    # Verificar que hay predicciones
    if not all(isinstance(df, pd.DataFrame) and not df.empty for df in forecasts.values()):
        logger.error(f"No hay suficientes predicciones válidas para crear ensemble de {instrument}")
        return pd.DataFrame()

    # Calcular pesos
    weights = calculate_ensemble_weights(instrument, config)

    # Encontrar fechas comunes en todos los forecasts
    first_model = list(forecasts.keys())[0]
    dates = forecasts[first_model].index

    # Crear DataFrame para almacenar el ensemble
    ensemble_df = pd.DataFrame(index=dates)
    ensemble_df['date'] = dates

    # Asegurarse de que todos los forecasts tienen las mismas fechas
    for model_type, forecast_df in forecasts.items():
        if not all(date in forecast_df.index for date in dates):
            logger.warning(f"Fechas inconsistentes en {model_type} para {instrument}")

            # Reindexar para alinear fechas
            forecast_df = forecast_df.reindex(dates)

    # Calcular predicción ponderada
    ensemble_df['forecast'] = 0.0
    ensemble_df['lower_95'] = 0.0
    ensemble_df['upper_95'] = 0.0

    for model_type, forecast_df in forecasts.items():
        weight = weights.get(model_type, 0.0)

        if weight > 0:
            # Añadir la contribución ponderada de este modelo
            if 'forecast' in forecast_df.columns:
                ensemble_df['forecast'] += weight * forecast_df['forecast']

            if 'lower_95' in forecast_df.columns:
                ensemble_df['lower_95'] += weight * forecast_df['lower_95']

            if 'upper_95' in forecast_df.columns:
                ensemble_df['upper_95'] += weight * forecast_df['upper_95']

    # Resetear el índice para tener 'date' como columna
    ensemble_df = ensemble_df.reset_index(drop=True)

    logger.info(f"Ensemble creado para {instrument}: {len(ensemble_df)} puntos de predicción")

    # Guardar datos históricos para visualizaciones
    try:
        historical_data_path = config.TRAIN_DATA_DIR / instrument / f"{instrument}_historical.pkl"
        if historical_data_path.parent.exists():
            # Obtener datos históricos para visualizaciones
            all_predictions = load_all_predictions(config)
            if not all_predictions.empty:
                # Filtrar datos históricos (no forecast) para este instrumento
                hist_data = all_predictions[
                    (all_predictions['Tipo_Mercado'] == instrument) & (all_predictions['Periodo'] != 'Forecast')
                ].copy()

                if not hist_data.empty:
                    # Guardar para uso posterior en visualizaciones
                    hist_data.to_pickle(historical_data_path)
                    logger.info(f"Datos históricos guardados para visualizaciones: {historical_data_path}")
    except Exception as e:
        logger.warning(f"No se pudieron guardar datos históricos: {str(e)}")

    return ensemble_df


def create_ensemble_metrics_comparison(instrument: str, config: Config) -> Dict:
    """Crea una comparación de métricas entre modelos individuales y ensemble."""
    logger = logging.getLogger(__name__)

    # Cargar todas las predicciones
    all_predictions = load_all_predictions(config)

    # Filtrar por instrumento y período de test
    instrument_data = all_predictions[all_predictions['Tipo_Mercado'] == instrument]
    test_data = instrument_data[instrument_data['Periodo'] == 'Test']

    if test_data.empty:
        logger.warning(f"No hay datos de test para {instrument}")
        return {}

    # Agrupar por modelo
    models_data = {}
    for model_type in config.MODEL_TYPES:
        model_test = test_data[test_data['Modelo'] == model_type]
        if not model_test.empty:
            models_data[model_type] = model_test

    # Verificar que hay al menos dos modelos para comparar
    if len(models_data) < 2:
        logger.warning(f"No hay suficientes modelos para comparar para {instrument}")
        return {}

    # Calcular métricas para cada modelo
    metrics_comparison = {}

    for model_type, model_data in models_data.items():
        # Asegurarse de que hay valores reales y predichos
        if 'Valor_Real' in model_data.columns and 'Valor_Predicho' in model_data.columns:
            # Convertir a numérico
            try:
                real = pd.to_numeric(model_data['Valor_Real'], errors='coerce')
                pred = pd.to_numeric(model_data['Valor_Predicho'], errors='coerce')

                # Eliminar NaN
                mask = ~(real.isna() | pred.isna())
                real = real[mask]
                pred = pred[mask]

                if len(real) > 0:
                    # Calcular métricas
                    rmse = np.sqrt(mean_squared_error(real, pred))
                    mae = mean_absolute_error(real, pred)

                    # Calcular MAPE evitando divisiones por cero
                    if (real == 0).any():
                        mape = np.nan
                    else:
                        mape = mean_absolute_percentage_error(real, pred) * 100

                    # Calcular hit direction
                    real_diff = real.diff().dropna()
                    pred_diff = pred.diff().dropna()

                    if len(real_diff) > 0:
                        hit_dir = np.mean(np.sign(real_diff) == np.sign(pred_diff)) * 100
                    else:
                        hit_dir = np.nan

                    metrics_comparison[model_type] = {"rmse": rmse, "mae": mae, "mape": mape, "hit_direction": hit_dir}

                    logger.info(
                        f"Métricas para {instrument}_{model_type}: RMSE={rmse:.4f}, MAE={mae:.4f}, Hit={hit_dir:.2f}%"
                    )
            except Exception as e:
                logger.error(f"Error calculando métricas para {instrument}_{model_type}: {str(e)}")

    # Calcular métricas para el ensemble
    # Para el ensemble, simulamos la predicción combinando los modelos con los pesos calculados
    try:
        # Obtener pesos
        weights = calculate_ensemble_weights(instrument, config)

        # Agrupar por fecha para asegurar alineamiento
        grouped_by_date = test_data.groupby(['date', 'Modelo'])

        # Preparar datos para ensemble
        ensemble_data = []

        for date, group in test_data.groupby('date'):
            # Si todas las fechas tienen todos los modelos
            if all(model in group['Modelo'].values for model in config.MODEL_TYPES):
                real_values = group['Valor_Real'].iloc[0]  # Todos deben tener el mismo valor real

                # Calcular predicción ponderada
                pred_ensemble = 0
                for model_type in config.MODEL_TYPES:
                    model_pred = group[group['Modelo'] == model_type]['Valor_Predicho'].iloc[0]
                    pred_ensemble += weights.get(model_type, 0) * float(model_pred)

                ensemble_data.append({'date': date, 'Valor_Real': real_values, 'Valor_Predicho': pred_ensemble})

        # Crear DataFrame de ensemble
        if ensemble_data:
            ensemble_df = pd.DataFrame(ensemble_data)

            # Calcular métricas
            real = pd.to_numeric(ensemble_df['Valor_Real'], errors='coerce')
            pred = pd.to_numeric(ensemble_df['Valor_Predicho'], errors='coerce')

            # Eliminar NaN
            mask = ~(real.isna() | pred.isna())
            real = real[mask]
            pred = pred[mask]

            if len(real) > 0:
                # Calcular métricas
                rmse = np.sqrt(mean_squared_error(real, pred))
                mae = mean_absolute_error(real, pred)

                # Calcular MAPE evitando divisiones por cero
                if (real == 0).any():
                    mape = np.nan
                else:
                    mape = mean_absolute_percentage_error(real, pred) * 100

                # Calcular hit direction
                real_diff = real.diff().dropna()
                pred_diff = pred.diff().dropna()

                if len(real_diff) > 0:
                    hit_dir = np.mean(np.sign(real_diff) == np.sign(pred_diff)) * 100
                else:
                    hit_dir = np.nan

                metrics_comparison['ENSEMBLE'] = {"rmse": rmse, "mae": mae, "mape": mape, "hit_direction": hit_dir}

                logger.info(f"Métricas para {instrument}_ENSEMBLE: RMSE={rmse:.4f}, MAE={mae:.4f}, Hit={hit_dir:.2f}%")
    except Exception as e:
        logger.error(f"Error calculando métricas de ensemble para {instrument}: {str(e)}")

    return metrics_comparison


def visualize_ensemble_comparison(instrument: str, metrics_comparison: Dict, config: Config):
    """Crea visualizaciones mejoradas de la comparación entre modelos y ensemble."""
    try:
        # Solo proceder si hay al menos dos modelos para comparar
        if len(metrics_comparison) < 2:
            return

        # Crear directorio para gráficos
        charts_dir = config.RESULTS_DIR / "charts"
        charts_dir.mkdir(exist_ok=True, parents=True)

        # Usar visualizaciones básicas si el módulo plots no está disponible
        if 'plots' not in globals():
            # Visualización básica (existente)
            # Preparar datos para gráficos
            models = list(metrics_comparison.keys())
            rmse_values = [metrics_comparison[model].get('rmse', 0) for model in models]
            hit_dir_values = [metrics_comparison[model].get('hit_direction', 0) for model in models]

            # Crear figura con dos subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # RMSE comparación (menor es mejor)
            bar_colors = ['#3498db' if model != 'ENSEMBLE' else '#e74c3c' for model in models]
            ax1.bar(models, rmse_values, color=bar_colors)
            ax1.set_title(f'Comparación de RMSE - {instrument}')
            ax1.set_ylabel('RMSE (menor es mejor)')
            ax1.tick_params(axis='x', rotation=45)

            # Añadir valores numéricos sobre las barras
            for i, v in enumerate(rmse_values):
                ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

            # Hit Direction comparación (mayor es mejor)
            ax2.bar(models, hit_dir_values, color=bar_colors)
            ax2.set_title(f'Comparación de Hit Direction - {instrument}')
            ax2.set_ylabel('Hit Direction % (mayor es mejor)')
            ax2.tick_params(axis='x', rotation=45)

            # Añadir valores numéricos sobre las barras
            for i, v in enumerate(hit_dir_values):
                ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(charts_dir / f"{instrument}_ensemble_comparison.png", dpi=300)
            plt.close()

            # Gráfico de radar para comparación completa
            metrics_keys = ['rmse', 'mae', 'mape', 'hit_direction']

            # Normalizar métricas para gráfico de radar
            normalized_metrics = {}
            for metric in metrics_keys:
                if metric == 'hit_direction':  # Mayor es mejor
                    values = [metrics_comparison[model].get(metric, 0) for model in models]
                    max_val = max(values) if values else 1
                    normalized_metrics[metric] = [value / max_val for value in values]
                else:  # Menor es mejor (invertir para que mayor sea mejor)
                    values = [metrics_comparison[model].get(metric, float('inf')) for model in models]
                    max_val = max(values) if values else 1
                    normalized_metrics[metric] = [1 - (value / max_val) for value in values]

            # Crear gráfico de radar
            plt.figure(figsize=(10, 8))

            # Número de variables
            N = len(metrics_keys)

            # Ángulos para cada eje
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Cerrar el polígono

            # Inicializar gráfico
            ax = plt.subplot(111, polar=True)

            # Dibujar para cada modelo
            for i, model in enumerate(models):
                values = []
                for metric in metrics_keys:
                    if metric in normalized_metrics:
                        values.append(normalized_metrics[metric][i])
                    else:
                        values.append(0)

                values += values[:1]  # Cerrar el polígono

                # Dibujar polígono
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
                ax.fill(angles, values, alpha=0.1)

            # Fijar etiquetas
            plt.xticks(angles[:-1], metrics_keys)

            # Añadir leyenda
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

            plt.title(f"Comparación de métricas normalizadas - {instrument}")
            plt.tight_layout()
            plt.savefig(charts_dir / f"{instrument}_radar_comparison.png", dpi=300)
            plt.close()

            return

        # Usar visualizaciones avanzadas si el módulo plots está disponible
        # Preparar datos para visualizaciones mejoradas
        metrics_by_model = {}
        for model, metrics in metrics_comparison.items():
            metrics_by_model[model] = {key.capitalize(): value for key, value in metrics.items()}

        # 1. Comparación de RMSE
        rmse_path = charts_dir / f"{instrument}_ensemble_comparison.png"
        plots.plot_metrics_comparison(
            metrics=metrics_by_model,
            metric_name='Rmse',
            title='Comparación de Modelos',
            figsize=(15, 8),
            save_path=rmse_path,
            instrument=instrument,
            is_lower_better=True,
        )
        logging.info(f"Gráfico de comparación RMSE guardado en {rmse_path}")

        # 2. Comparación de Hit Direction
        hit_dir_path = charts_dir / f"{instrument}_hit_direction_comparison.png"
        plots.plot_metrics_comparison(
            metrics=metrics_by_model,
            metric_name='Hit_Direction',
            title='Comparación de Modelos',
            figsize=(15, 8),
            save_path=hit_dir_path,
            instrument=instrument,
            is_lower_better=False,
        )
        logging.info(f"Gráfico de comparación Hit Direction guardado en {hit_dir_path}")

        # 3. Radar de métricas múltiples
        radar_path = charts_dir / f"{instrument}_radar_comparison.png"
        plots.plot_metrics_radar(
            metrics=metrics_by_model,
            metric_names=['Rmse', 'Mae', 'Mape', 'Hit_Direction', 'R2'],
            title='Comparación Múltiples Métricas',
            figsize=(15, 15),
            save_path=radar_path,
            instrument=instrument,
        )
        logging.info(f"Gráfico radar de métricas guardado en {radar_path}")

    except Exception as e:
        logging.getLogger(__name__).error(f"Error creando visualizaciones para {instrument}: {str(e)}")


def visualize_ensemble_forecasts(
    instrument: str,
    historical_data: pd.DataFrame,
    models_forecasts: Dict[str, pd.DataFrame],
    ensemble_forecast: pd.DataFrame,
    config: Config,
):
    """
    Genera visualizaciones avanzadas para los pronósticos del ensemble.

    Args:
        instrument: Instrumento financiero
        historical_data: DataFrame con datos históricos
        models_forecasts: Diccionario con pronósticos por modelo
        ensemble_forecast: DataFrame con pronóstico del ensemble
        config: Configuración global
    """
    try:
        # Crear directorio para gráficos
        charts_dir = config.RESULTS_DIR / "charts"
        charts_dir.mkdir(exist_ok=True, parents=True)

        # Obtener serie histórica (target)
        target_col = f"{instrument.lower()}_close"
        if target_col not in historical_data.columns:
            # Intentar encontrar la columna del target
            potential_cols = [col for col in historical_data.columns if 'close' in col.lower()]
            if potential_cols:
                target_col = potential_cols[0]
            else:
                # Usar la primera columna como fallback
                target_col = historical_data.columns[0]

        historical_series = historical_data[target_col]

        # Si no tenemos el módulo plots, usar visualización básica
        if 'plots' not in globals():
            # Código de visualización básica existente
            return

        # Usar visualizaciones avanzadas
        # 1. Preparar datos para gráfico de ensemble
        forecast_series = {}
        for model_name, forecast_df in models_forecasts.items():
            if 'forecast' in forecast_df.columns:
                forecast_series[model_name] = pd.Series(
                    forecast_df['forecast'].values,
                    index=(
                        forecast_df.index
                        if isinstance(forecast_df.index, pd.DatetimeIndex)
                        else pd.to_datetime(forecast_df['date'])
                    ),
                )

        # 2. Datos del ensemble
        ensemble_series = pd.Series(
            ensemble_forecast['forecast'].values,
            index=(
                ensemble_forecast.index
                if isinstance(ensemble_forecast.index, pd.DatetimeIndex)
                else pd.to_datetime(ensemble_forecast['date'])
            ),
        )

        # Obtener intervalos de confianza si están disponibles
        ci_lower = None
        ci_upper = None
        if 'lower_95' in ensemble_forecast.columns and 'upper_95' in ensemble_forecast.columns:
            ci_lower = pd.Series(
                ensemble_forecast['lower_95'].values,
                index=(
                    ensemble_forecast.index
                    if isinstance(ensemble_forecast.index, pd.DatetimeIndex)
                    else pd.to_datetime(ensemble_forecast['date'])
                ),
            )
            ci_upper = pd.Series(
                ensemble_forecast['upper_95'].values,
                index=(
                    ensemble_forecast.index
                    if isinstance(ensemble_forecast.index, pd.DatetimeIndex)
                    else pd.to_datetime(ensemble_forecast['date'])
                ),
            )

        # 3. Crear visualización de comparación de ensemble
        ensemble_path = charts_dir / f"{instrument}_ensemble_forecast.png"
        plots.plot_ensemble_comparison(
            historical=historical_series,
            forecasts=forecast_series,
            ensemble_forecast=ensemble_series,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            title="Pronóstico con Ensemble",
            save_path=ensemble_path,
            instrument=instrument,
        )
        logging.info(f"Gráfico de pronóstico ensemble guardado en {ensemble_path}")

    except Exception as e:
        logging.getLogger(__name__).error(f"Error creando visualizaciones de pronóstico para {instrument}: {str(e)}")


def integrate_ensemble_predictions(config: Config):
    """Integra las predicciones de ensemble con las individuales en un solo archivo."""
    logger = logging.getLogger(__name__)

    try:
        # Cargar el archivo consolidado
        all_predictions = load_all_predictions(config)

        # Estructuras para almacenar datos de ensemble
        ensemble_data = []

        # Procesar cada instrumento
        for instrument in config.INSTRUMENTS:
            # Crear predicciones de ensemble
            ensemble_df = create_ensemble_predictions(instrument, config)

            if ensemble_df.empty:
                logger.warning(f"No hay predicciones de ensemble para {instrument}")
                continue

            # Calcular métricas comparativas
            metrics_comparison = create_ensemble_metrics_comparison(instrument, config)

            # Crear visualizaciones
            visualize_ensemble_comparison(instrument, metrics_comparison, config)

            # Visualizar pronósticos del ensemble
            try:
                # Intentar cargar datos históricos para visualizaciones
                historical_data_path = config.TRAIN_DATA_DIR / instrument / f"{instrument}_historical.pkl"
                historical_data = None

                if historical_data_path.exists():
                    try:
                        historical_data = pd.read_pickle(historical_data_path)
                    except:
                        pass

                # Si no pudimos cargar datos históricos, intentar extraerlos de all_predictions
                if historical_data is None or historical_data.empty:
                    historical_data = all_predictions[
                        (all_predictions['Tipo_Mercado'] == instrument) & (all_predictions['Periodo'] != 'Forecast')
                    ].copy()

                    if historical_data.empty:
                        logger.warning(f"No hay datos históricos para visualizaciones de {instrument}")
                    else:
                        # Pivotear para tener formato de series temporales
                        historical_data = historical_data.pivot(
                            index='date', columns='Modelo', values=['Valor_Real', 'Valor_Predicho']
                        )
                        # Simplificar columnas
                        historical_data.columns = [f"{b}_{a}".lower() for a, b in historical_data.columns]

                if not historical_data.empty:
                    # Obtener pronósticos de todos los modelos
                    models_forecasts = {}
                    for model in config.MODEL_TYPES:
                        forecast_file = config.TRAIN_DATA_DIR / instrument / f"{instrument}_{model}_forecast.csv"
                        if forecast_file.exists():
                            models_forecasts[model] = pd.read_csv(forecast_file)

                    # Visualizar ensemble con comparación
                    visualize_ensemble_forecasts(
                        instrument=instrument,
                        historical_data=historical_data,
                        models_forecasts=models_forecasts,
                        ensemble_forecast=ensemble_df,
                        config=config,
                    )
            except Exception as e:
                logger.error(f"Error visualizando pronósticos: {str(e)}")

            # Guardar predicciones de ensemble
            ensemble_file = config.RESULTS_DIR / f"{instrument}_ENSEMBLE_forecast.csv"
            ensemble_df.to_csv(ensemble_file, index=False)
            logger.info(f"Predicciones de ensemble guardadas en {ensemble_file}")

            # Guardar métricas comparativas
            metrics_file = config.RESULTS_DIR / f"{instrument}_ensemble_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_comparison, f, indent=4)
            logger.info(f"Métricas comparativas guardadas en {metrics_file}")

            # Integrar datos de ensemble para el archivo consolidado
            for _, row in ensemble_df.iterrows():
                ensemble_data.append(
                    {
                        "date": row.get('date'),
                        "Valor_Real": None,  # Para predicciones futuras, no hay valor real
                        "Valor_Predicho": row.get('forecast'),
                        "Modelo": "ENSEMBLE",
                        "Version": "1.0",
                        "RMSE": metrics_comparison.get('ENSEMBLE', {}).get('rmse'),
                        "Hyperparámetros": json.dumps({"method": config.WEIGHTING_METHOD}),
                        "Tipo_Mercado": instrument,
                        "Periodo": "Forecast",
                    }
                )

        # Añadir datos de ensemble al archivo consolidado
        if ensemble_data:
            # Convertir a DataFrame
            ensemble_df = pd.DataFrame(ensemble_data)

            # Asegurar que 'date' tiene el formato correcto
            all_predictions['date'] = pd.to_datetime(all_predictions['date'])
            ensemble_df['date'] = pd.to_datetime(ensemble_df['date'])

            # Concatenar con las predicciones existentes
            combined_predictions = pd.concat([all_predictions, ensemble_df], ignore_index=True)

            # Guardar archivo combinado
            output_file = config.RESULTS_DIR / "all_models_with_ensemble.csv"
            combined_predictions.to_csv(output_file, index=False)
            logger.info(f"Archivo combinado con ensemble guardado en {output_file}")

            return output_file
        else:
            logger.warning("No se generaron datos de ensemble para incluir en el archivo consolidado")
            return None

    except Exception as e:
        logger.error(f"Error integrando predicciones de ensemble: {str(e)}")
        return None


def main():
    """Función principal."""
    # Configurar logging
    logger = setup_logging("logs", "arima_ensemble")
    logger.info("Iniciando proceso de ensemble de modelos ARIMA/SARIMAX")

    # Crear config
    config = Config()

    # Crear directorios si no existen
    config.RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # Integrar predicciones de ensemble
    output_file = integrate_ensemble_predictions(config)

    if output_file:
        logger.info(f"✅ Proceso de ensemble completado exitosamente. Resultados guardados en {output_file}")
    else:
        logger.warning("⚠️ Proceso de ensemble completado con advertencias.")


if __name__ == "__main__":
    main()

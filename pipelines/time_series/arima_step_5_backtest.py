#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtesting y Evaluación Avanzada
================================
Realiza análisis de backtesting sobre los modelos entrenados, calculando
métricas avanzadas y generando visualizaciones detalladas del rendimiento.

Características:
- Análisis de amplitud y fase mediante transformada de Hilbert
- Métrica de dirección ponderada por volatilidad
- Comparación con modelos de referencia
- Análisis de robustez por subperíodos
- Visualizaciones avanzadas (cuando el módulo 'plots' está disponible)

Métricas calculadas:
- RMSE, MAE, MAPE, R², R² ajustado
- Amplitud Score (correlación de volatilidad)
- Fase Score (sincronización de ciclos)
- Hit Direction ponderada
"""
import sys
import os
import json
import logging
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.signal import hilbert, detrend
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error,
    r2_score
)

# Intentar importar módulos de utilidades para visualizaciones avanzadas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
try:
    from time_series.utils import plots
    HAS_ADVANCED_PLOTTING = True
    logging.info("Módulo de visualizaciones avanzadas cargado correctamente")
except ImportError:
    # Intentar con rutas alternativas durante desarrollo
    try:
        sys.path.append(os.path.abspath("../pipelines"))
        from time_series.utils import plots
        HAS_ADVANCED_PLOTTING = True
        logging.info("Módulo de visualizaciones avanzadas cargado desde ruta alternativa")
    except ImportError:
        HAS_ADVANCED_PLOTTING = False
        logging.warning("No se pudieron importar los módulos de utilidades. Las visualizaciones avanzadas no estarán disponibles.")

# Configuración del proyecto
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "4_results"
METRICS_DIR = DATA_DIR / "5_metrics"
CHARTS_DIR = METRICS_DIR / "charts"
LOG_DIR = PROJECT_ROOT / "logs"

# Asegurar que existen los directorios
for directory in [RESULTS_DIR, METRICS_DIR, CHARTS_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Configuración de logging
log_file = LOG_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_most_recent_file(directory: Path, pattern: str = '*.csv') -> Optional[Path]:
    """Obtiene el archivo más reciente en un directorio con un patrón dado."""
    try:
        files = list(directory.glob(pattern))
        if not files:
            logging.warning(f"No se encontraron archivos con patrón '{pattern}' en {directory}")
            return None
        
        # Ordenar por fecha de modificación (más reciente primero)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        logging.info(f"Archivo más reciente encontrado: {files[0]}")
        return files[0]
    except Exception as e:
        logging.error(f"Error buscando archivos: {str(e)}")
        return None

def find_predictions_file() -> Path:
    """Busca el archivo de predicciones más reciente."""
    # Opciones en orden de preferencia
    file_options = [
        RESULTS_DIR / "all_models_with_ensemble.csv",
        RESULTS_DIR / "archivo_para_powerbi.csv",
        DATA_DIR / "2_trainingdata_ts" / "all_models_predictions.csv"
    ]
    
    # Verificar cada opción
    for file_path in file_options:
        if file_path.exists():
            logging.info(f"Usando archivo de predicciones: {file_path}")
            return file_path
    
    # Si no se encuentra ninguno específico, buscar cualquier CSV de resultados
    recent_file = get_most_recent_file(RESULTS_DIR, "*predictions*.csv")
    if recent_file:
        return recent_file
    
    # Como fallback, buscar en todo el directorio de datos
    logging.warning("Buscando archivo de predicciones en todo el directorio de datos...")
    
    csv_files = []
    for pattern in ["*predictions*.csv", "*forecast*.csv", "*powerbi*.csv"]:
        for file_path in DATA_DIR.rglob(pattern):
            csv_files.append((file_path, file_path.stat().st_mtime))
    
    if csv_files:
        csv_files.sort(key=lambda x: x[1], reverse=True)  # Ordenar por fecha de modificación
        logging.info(f"Usando como fallback: {csv_files[0][0]}")
        return csv_files[0][0]
    
    raise FileNotFoundError("No se encontró ningún archivo de predicciones en el proyecto")

def load_csv_file(file_path: Path) -> pd.DataFrame:
    """Carga un archivo CSV con detección automática de separador."""
    try:
        # Intentar primero con coma
        df = pd.read_csv(file_path, encoding='utf-8')
        return df
    except:
        try:
            # Si falla, probar con punto y coma
            df = pd.read_csv(file_path, encoding='utf-8', sep=';')
            return df
        except Exception as e:
            logging.error(f"Error leyendo {file_path}: {str(e)}")
            raise

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el DataFrame para análisis (convierte tipos de datos)."""
    # Copiar para no modificar el original
    df_norm = df.copy()
    
    # Convertir fechas
    if 'date' in df_norm.columns:
        df_norm['date'] = pd.to_datetime(df_norm['date'], errors='coerce')
    
    # Convertir valores numéricos
    numeric_cols = ['Valor_Real', 'Valor_Predicho', 'RMSE']
    for col in numeric_cols:
        if col in df_norm.columns:
            # Manejar diferentes formatos de decimales
            if df_norm[col].dtype == object:  # Si es string
                # Reemplazar coma por punto para valores decimales
                df_norm[col] = df_norm[col].astype(str).str.replace(',', '.', regex=False)
            
            # Convertir a numérico
            df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce')
    
    return df_norm

def filter_data_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra los datos para análisis, eliminando NaN y fechas futuras."""
    # Eliminar filas sin fecha
    if 'date' in df.columns:
        df = df.dropna(subset=['date'])
        
        # Filtrar fechas futuras (mantener solo hasta hoy)
        today = pd.Timestamp.now().normalize()
        df = df[df['date'] <= today]
    
    # Asegurarse de que hay valores reales y predichos
    value_cols = ['Valor_Real', 'Valor_Predicho']
    if all(col in df.columns for col in value_cols):
        df = df.dropna(subset=value_cols)
    
    return df

def calculate_basic_metrics(real: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas básicas de rendimiento."""
    if len(real) < 2 or len(pred) < 2:
        logging.warning(f"Insuficientes datos para calcular métricas: {len(real)} valores reales, {len(pred)} predicciones")
        return {}
    
    try:
        # Métricas estándar
        mse = mean_squared_error(real, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(real, pred)
        
        # MAPE (evitar divisiones por cero)
        if np.any(real == 0):
            mape = np.nan
        else:
            mape = mean_absolute_percentage_error(real, pred) * 100
        
        # R² y R² ajustado
        r2 = r2_score(real, pred)
        n = len(real)
        p = 1  # Número de predictores (simplificado)
        r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - p - 1) if n > p + 1 else np.nan
        
        # Hit Direction (acierto en dirección)
        real_diff = np.diff(real)
        pred_diff = np.diff(pred)
        
        if len(real_diff) > 0:
            hit_dir = np.mean(np.sign(real_diff) == np.sign(pred_diff)) * 100
        else:
            hit_dir = np.nan
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'R2_adjusted': r2_adj,
            'Hit_Direction': hit_dir
        }
    
    except Exception as e:
        logging.error(f"Error calculando métricas básicas: {str(e)}")
        return {}

def calculate_amplitude_score(real: np.ndarray, pred: np.ndarray) -> float:
    """
    Calcula el score de amplitud usando la transformada de Hilbert.
    Mide la similitud en la magnitud de las oscilaciones.
    """
    try:
        if len(real) < 10 or len(pred) < 10:
            return np.nan
        
        # Detrend para análisis de ciclos
        real_detrend = detrend(real)
        pred_detrend = detrend(pred)
        
        # Aplicar transformada de Hilbert
        analytic_real = hilbert(real_detrend)
        analytic_pred = hilbert(pred_detrend)
        
        # Calcular envolvente (amplitud)
        amplitude_real = np.abs(analytic_real)
        amplitude_pred = np.abs(analytic_pred)
        
        # Calcular IQR (rango intercuartil) para comparar dispersión
        iqr_real = np.percentile(amplitude_real, 75) - np.percentile(amplitude_real, 25)
        iqr_pred = np.percentile(amplitude_pred, 75) - np.percentile(amplitude_pred, 25)
        
        # Calcular score normalizado (1 = perfecta similitud, 0 = totalmente diferente)
        score = 1 - np.abs(iqr_real - iqr_pred) / max(iqr_real, iqr_pred)
        
        # Limitar a [0, 1]
        return np.clip(score, 0, 1)
    
    except Exception as e:
        logging.error(f"Error calculando score de amplitud: {str(e)}")
        return np.nan

def calculate_phase_score(real: np.ndarray, pred: np.ndarray) -> float:
    """
    Calcula el score de fase usando la transformada de Hilbert.
    Mide la sincronización temporal entre la serie real y predicha.
    """
    try:
        if len(real) < 10 or len(pred) < 10:
            return np.nan
        
        # Detrend para análisis de ciclos
        real_detrend = detrend(real)
        pred_detrend = detrend(pred)
        
        # Aplicar transformada de Hilbert
        analytic_real = hilbert(real_detrend)
        analytic_pred = hilbert(pred_detrend)
        
        # Extraer fase instantánea
        phase_real = np.unwrap(np.angle(analytic_real))
        phase_pred = np.unwrap(np.angle(analytic_pred))
        
        # Calcular diferencia de fase (error)
        phase_error = np.std(phase_real - phase_pred)
        
        # Convertir a score (1 = perfecta sincronización, 0 = completamente desfasado)
        score = 1 / (1 + phase_error)
        
        # Limitar a [0, 1]
        return np.clip(score, 0, 1)
    
    except Exception as e:
        logging.error(f"Error calculando score de fase: {str(e)}")
        return np.nan

def calculate_weighted_hit_direction(real: np.ndarray, pred: np.ndarray) -> float:
    """
    Calcula el hit direction ponderado por la magnitud del cambio.
    Da mayor peso a predecir correctamente cambios grandes.
    """
    try:
        if len(real) < 3 or len(pred) < 3:
            return np.nan
        
        # Calcular cambios
        real_diff = np.diff(real)
        pred_diff = np.diff(pred)
        
        if len(real_diff) == 0:
            return np.nan
        
        # Magnitud de los cambios (normalizada)
        magnitudes = np.abs(real_diff) / np.max(np.abs(real_diff)) if np.max(np.abs(real_diff)) > 0 else np.ones_like(real_diff)
        
        # Aciertos (1 si coincide la dirección, 0 si no)
        hits = (np.sign(real_diff) == np.sign(pred_diff)).astype(float)
        
        # Calcular hit direction ponderado
        weighted_hits = np.sum(hits * magnitudes) / np.sum(magnitudes) if np.sum(magnitudes) > 0 else np.nan
        
        return weighted_hits * 100  # Convertir a porcentaje
    
    except Exception as e:
        logging.error(f"Error calculando hit direction ponderado: {str(e)}")
        return np.nan

def evaluate_models_by_group(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Evalúa los modelos agrupados por instrumento y tipo de modelo.
    """
    result = {}
    
    try:
        # Verificar que estén las columnas mínimas necesarias
        required_cols = ['Modelo', 'Tipo_Mercado', 'Valor_Real', 'Valor_Predicho']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logging.error(f"Faltan columnas requeridas: {missing}")
            return {}
        
        # Filtrar solo datos de test
        if 'Periodo' in df.columns:
            test_data = df[df['Periodo'] == 'Test'].copy()
            if test_data.empty:
                logging.warning("No hay datos de test, usando todos los datos disponibles")
                test_data = df.copy()
        else:
            test_data = df.copy()
            logging.warning("No hay columna 'Periodo', usando todos los datos disponibles")
        
        # Agrupar por instrumento y modelo
        grouped = test_data.groupby(['Tipo_Mercado', 'Modelo'])
        
        for (instrument, model), group in grouped:
            if instrument not in result:
                result[instrument] = {}
            
            # Convertir valores a numérico y eliminar NaN
            real_values = pd.to_numeric(group['Valor_Real'], errors='coerce')
            pred_values = pd.to_numeric(group['Valor_Predicho'], errors='coerce')
            
            # Filtrar datos válidos
            mask = ~(real_values.isna() | pred_values.isna())
            real = real_values[mask].values
            pred = pred_values[mask].values
            
            if len(real) < 5 or len(pred) < 5:
                logging.warning(f"Insuficientes datos para {instrument} - {model}: {len(real)} valores válidos")
                continue
            
            # Calcular métricas
            metrics = calculate_basic_metrics(real, pred)
            
            # Métricas avanzadas
            metrics['Amplitude_Score'] = calculate_amplitude_score(real, pred)
            metrics['Phase_Score'] = calculate_phase_score(real, pred)
            metrics['Weighted_Hit_Direction'] = calculate_weighted_hit_direction(real, pred)
            
            # Guardar métricas
            result[instrument][model] = {
                'metrics': metrics,
                'num_observations': len(real),
                'period': {
                    'start': group['date'].min() if 'date' in group.columns else None,
                    'end': group['date'].max() if 'date' in group.columns else None
                }
            }
            
            logging.info(f"Evaluado {instrument} - {model}: {len(real)} observaciones")
    
    except Exception as e:
        logging.error(f"Error en evaluación por grupo: {str(e)}")
    
    return result

def evaluate_by_subperiods(df: pd.DataFrame, period_length: int = 63) -> Dict[str, Any]:
    """
    Evalúa el rendimiento por subperíodos para detectar cambios en el tiempo.
    
    Args:
        df: DataFrame con los datos
        period_length: Longitud de cada subperíodo en días (default: 63 ≈ 3 meses)
    """
    result = {}
    
    try:
        # Verificar columnas necesarias
        required_cols = ['date', 'Modelo', 'Tipo_Mercado', 'Valor_Real', 'Valor_Predicho']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logging.error(f"Faltan columnas requeridas para análisis por subperíodos: {missing}")
            return {}
        
        # Asegurar que date es datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Filtrar solo datos de test
        if 'Periodo' in df.columns:
            test_data = df[df['Periodo'] == 'Test'].copy()
            if test_data.empty:
                logging.warning("No hay datos de test para análisis por subperíodos")
                return {}
        else:
            test_data = df.copy()
        
        # Crear subperíodos
        min_date = test_data['date'].min()
        max_date = test_data['date'].max()
        
        # Si el rango de fechas es muy corto, no hacer análisis por subperíodos
        total_days = (max_date - min_date).days
        if total_days < period_length:
            logging.warning(f"Rango de fechas demasiado corto para análisis por subperíodos: {total_days} días < {period_length}")
            return {}
        
        # Definir subperíodos
        subperiods = []
        current_start = min_date
        
        while current_start < max_date:
            current_end = current_start + timedelta(days=period_length)
            if current_end > max_date:
                current_end = max_date
            
            subperiods.append({
                'start': current_start,
                'end': current_end,
                'label': f"{current_start.strftime('%Y-%m-%d')} a {current_end.strftime('%Y-%m-%d')}"
            })
            
            current_start = current_end + timedelta(days=1)
        
        # Estructura para resultados
        result = {
            'subperiods': subperiods,
            'instruments': {}
        }
        
        # Evaluar cada instrumento y modelo por subperíodo
        for instrument in test_data['Tipo_Mercado'].unique():
            instrument_data = test_data[test_data['Tipo_Mercado'] == instrument]
            
            result['instruments'][instrument] = {
                'models': {}
            }
            
            for model in instrument_data['Modelo'].unique():
                model_data = instrument_data[instrument_data['Modelo'] == model]
                
                period_metrics = []
                
                for period in subperiods:
                    # Filtrar datos para este período
                    period_data = model_data[
                        (model_data['date'] >= period['start']) & 
                        (model_data['date'] <= period['end'])
                    ]
                    
                    if len(period_data) < 5:
                        # No suficientes datos
                        continue
                    
                    # Calcular métricas
                    real_values = pd.to_numeric(period_data['Valor_Real'], errors='coerce')
                    pred_values = pd.to_numeric(period_data['Valor_Predicho'], errors='coerce')
                    
                    # Filtrar datos válidos
                    mask = ~(real_values.isna() | pred_values.isna())
                    real = real_values[mask].values
                    pred = pred_values[mask].values
                    
                    if len(real) < 5:
                        continue
                    
                    # Calcular métricas
                    metrics = calculate_basic_metrics(real, pred)
                    
                    # Guardar métricas del período
                    period_metrics.append({
                        'period': period['label'],
                        'start': period['start'].strftime('%Y-%m-%d'),
                        'end': period['end'].strftime('%Y-%m-%d'),
                        'num_observations': len(real),
                        'metrics': metrics
                    })
                
                # Solo guardar si hay al menos un período con datos
                if period_metrics:
                    result['instruments'][instrument]['models'][model] = period_metrics
                    logging.info(f"Análisis por subperíodos completado para {instrument} - {model}: {len(period_metrics)} períodos")
    
    except Exception as e:
        logging.error(f"Error en evaluación por subperíodos: {str(e)}")
    
    return result

def generate_metrics_visualizations(results: Dict[str, Dict[str, Dict[str, Any]]], output_dir: Path):
    """
    Genera visualizaciones mejoradas de las métricas por instrumento y modelo.
    """
    try:
        # Crear directorio para gráficos
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(exist_ok=True, parents=True)
        
        # Verificar si el módulo de visualizaciones avanzadas está disponible
        has_advanced_viz = 'plots' in globals()
        
        # Para cada instrumento
        for instrument, models_data in results.items():
            if not models_data:
                continue
            
            # Extraer métricas para todos los modelos
            metrics_data = []
            for model, data in models_data.items():
                if 'metrics' in data:
                    row = {'Model': model}
                    row.update(data['metrics'])
                    metrics_data.append(row)
            
            if not metrics_data:
                continue
            
            # Crear DataFrame con métricas
            df_metrics = pd.DataFrame(metrics_data)
            
            if has_advanced_viz:
                # Usar visualizaciones avanzadas
                # Preparar datos en formato para plots
                metrics_by_model = {}
                for _, row in df_metrics.iterrows():
                    model = row['Model']
                    metrics_by_model[model] = {
                        col: row[col] for col in df_metrics.columns if col != 'Model'
                    }
                
                # 1. Gráfico comparativo de RMSE
                rmse_path = charts_dir / f"{instrument}_rmse_comparison.png"
                plots.plot_metrics_comparison(
                    metrics=metrics_by_model,
                    metric_name='RMSE',
                    title='Comparación de RMSE',
                    save_path=rmse_path,
                    instrument=instrument,
                    is_lower_better=True
                )
                
                # 2. Gráfico comparativo de Hit Direction
                hit_dir_path = charts_dir / f"{instrument}_hit_direction_comparison.png"
                plots.plot_metrics_comparison(
                    metrics=metrics_by_model,
                    metric_name='Hit_Direction',
                    title='Comparación de Hit Direction',
                    save_path=hit_dir_path,
                    instrument=instrument,
                    is_lower_better=False
                )
                
                # 3. Gráfico de radar para métricas avanzadas
                metrics_for_radar = ['RMSE', 'MAE', 'Hit_Direction', 'R2', 'Amplitude_Score', 'Phase_Score', 'Weighted_Hit_Direction']
                available_metrics = [m for m in metrics_for_radar if any(m in model_data for model_data in metrics_by_model.values())]
                
                if len(available_metrics) >= 2:
                    radar_path = charts_dir / f"{instrument}_radar_metrics.png"
                    plots.plot_metrics_radar(
                        metrics=metrics_by_model,
                        metric_names=available_metrics,
                        title='Métricas Múltiples',
                        save_path=radar_path,
                        instrument=instrument
                    )
            else:
                # Usar visualizaciones básicas (código original)
                # 1. Gráfico comparativo de RMSE
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Model', y='RMSE', data=df_metrics, palette='viridis')
                plt.title(f'Comparación de RMSE - {instrument}')
                plt.ylabel('RMSE (menor es mejor)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(charts_dir / f"{instrument}_rmse_comparison.png", dpi=300)
                plt.close()
                
                # 2. Gráfico comparativo de Hit Direction
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Model', y='Hit_Direction', data=df_metrics, palette='viridis')
                plt.title(f'Comparación de Hit Direction - {instrument}')
                plt.ylabel('Hit Direction % (mayor es mejor)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(charts_dir / f"{instrument}_hit_direction_comparison.png", dpi=300)
                plt.close()
                
                # 3. Gráfico de radar para métricas avanzadas
                metrics_for_radar = ['Amplitude_Score', 'Phase_Score', 'Weighted_Hit_Direction', 'R2']
                
                # Verificar que existen estas métricas
                available_metrics = [m for m in metrics_for_radar if m in df_metrics.columns]
                
                if len(available_metrics) >= 2:  # Necesitamos al menos 2 métricas para un radar
                    # Normalizar métricas para el radar
                    df_radar = df_metrics[['Model'] + available_metrics].copy()
                    
                    # Para Weighted_Hit_Direction, normalizar a [0, 1]
                    if 'Weighted_Hit_Direction' in df_radar.columns:
                        df_radar['Weighted_Hit_Direction'] = df_radar['Weighted_Hit_Direction'] / 100.0
                    
                    # Preparar datos para radar
                    categories = available_metrics
                    N = len(categories)
                    
                    # Crear figura
                    plt.figure(figsize=(10, 10))
                    ax = plt.subplot(111, polar=True)
                    
                    # Ángulos para el gráfico
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # Cerrar el polígono
                    
                    # Dibujar para cada modelo
                    for i, model in enumerate(df_radar['Model']):
                        values = df_radar.loc[df_radar['Model'] == model, available_metrics].values.flatten().tolist()
                        values += values[:1]  # Cerrar el polígono
                        
                        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
                        ax.fill(angles, values, alpha=0.1)
                    
                    # Etiquetas y leyendas
                    plt.xticks(angles[:-1], categories, size=10)
                    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
                    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    
                    plt.title(f'Métricas Avanzadas - {instrument}')
                    plt.tight_layout()
                    plt.savefig(charts_dir / f"{instrument}_radar_metrics.png", dpi=300)
                    plt.close()
        
        logging.info(f"Visualizaciones generadas en {charts_dir}")
    
    except Exception as e:
        logging.error(f"Error generando visualizaciones: {str(e)}")

def generate_subperiod_visualizations(subperiod_results: Dict[str, Any], output_dir: Path):
    """
    Genera visualizaciones mejoradas de las métricas por subperíodos.
    """
    try:
        if not subperiod_results or 'instruments' not in subperiod_results:
            logging.warning("No hay datos de subperíodos para visualizar")
            return
        
        # Crear directorio para gráficos
        charts_dir = output_dir / "charts" / "subperiods"
        charts_dir.mkdir(exist_ok=True, parents=True)
        
        # Verificar si el módulo de visualizaciones avanzadas está disponible
        has_advanced_viz = 'plots' in globals()
        
        # Para cada instrumento
        for instrument, instrument_data in subperiod_results['instruments'].items():
            if 'models' not in instrument_data or not instrument_data['models']:
                continue
            
            # Para cada modelo
            for model, periods_data in instrument_data['models'].items():
                if not periods_data:
                    continue
                
                # Extraer datos para gráficos
                df_periods = pd.DataFrame(periods_data)
                
                # Extraer métricas
                metrics_cols = []
                for period in periods_data:
                    if 'metrics' in period:
                        metrics_cols = list(period['metrics'].keys())
                        break
                
                if not metrics_cols:
                    continue
                
                # Crear DataFrame expandido con métricas
                metrics_data = []
                for period in periods_data:
                    if 'metrics' not in period:
                        continue
                    
                    row = {
                        'period': period['period'],
                        'start': period['start'],
                        'end': period['end'],
                        'num_observations': period['num_observations']
                    }
                    
                    for metric, value in period['metrics'].items():
                        row[metric] = value
                    
                    metrics_data.append(row)
                
                df_metrics = pd.DataFrame(metrics_data)
                
                # Ordenar por fecha de inicio
                df_metrics['start'] = pd.to_datetime(df_metrics['start'])
                df_metrics = df_metrics.sort_values('start')
                
                if has_advanced_viz:
                    # Usar visualizaciones avanzadas
                    # Necesitamos formatear como una serie temporal para plot_metric_evolution
                    for metric in ['RMSE', 'MAE', 'Hit_Direction', 'R2']:
                        if metric in df_metrics.columns:
                            metric_values = pd.DataFrame({
                                metric: df_metrics[metric].values
                            }, index=df_metrics['start'])
                            
                            # Crear gráfico de evolución
                            metric_path = charts_dir / f"{instrument}_{model}_{metric.lower()}_evolution.png"
                            plots.plot_metric_evolution(
                                metric_values=metric_values,
                                metric_name=metric,
                                title=f'Evolución por Subperíodo',
                                save_path=metric_path,
                                instrument=instrument,
                                model_name=model
                            )
                    
                    # Crear dashboard con todas las métricas principales
                    dashboard_path = charts_dir / f"{instrument}_{model}_metrics_dashboard.png"
                    
                    # Primero necesitamos extraer las series temporales
                    metrics_to_plot = {}
                    for metric in ['RMSE', 'MAE', 'Hit_Direction', 'R2']:
                        if metric in df_metrics.columns:
                            metrics_to_plot[metric] = pd.Series(
                                df_metrics[metric].values,
                                index=df_metrics['start']
                            )
                    
                    if metrics_to_plot:
                        # Crear figura con múltiples subplots
                        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4*len(metrics_to_plot)))
                        if len(metrics_to_plot) == 1:
                            axes = [axes]  # Asegurar que sea iterable
                        
                        for i, (metric, series) in enumerate(metrics_to_plot.items()):
                            ax = axes[i]
                            # Determinar si valores más altos son mejores
                            color = '#2980B9' if metric in ['Hit_Direction', 'R2'] else '#E74C3C'
                            ax.plot(series.index, series.values, marker='o', color=color, linewidth=2)
                            ax.set_title(f'{metric} - {instrument} - {model}')
                            ax.grid(True, linestyle='--', alpha=0.7)
                            
                            # Ajustar eje Y para Hit Direction
                            if metric == 'Hit_Direction':
                                ax.set_ylim([0, 100])
                        
                        plt.tight_layout()
                        plt.savefig(dashboard_path, dpi=300)
                        plt.close()
                
                else:
                    # Usar visualizaciones básicas (código original)
                    # 1. Gráfico de evolución de RMSE a lo largo del tiempo
                    if 'RMSE' in df_metrics.columns:
                        plt.figure(figsize=(12, 6))
                        plt.plot(df_metrics['start'], df_metrics['RMSE'], marker='o', linewidth=2)
                        plt.title(f'Evolución de RMSE - {instrument} - {model}')
                        plt.ylabel('RMSE')
                        plt.xlabel('Período')
                        plt.xticks(rotation=45)
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        plt.savefig(charts_dir / f"{instrument}_{model}_rmse_evolution.png", dpi=300)
                        plt.close()
                    
                    # 2. Gráfico de evolución de Hit Direction
                    if 'Hit_Direction' in df_metrics.columns:
                        plt.figure(figsize=(12, 6))
                        plt.plot(df_metrics['start'], df_metrics['Hit_Direction'], marker='o', linewidth=2, color='green')
                        plt.title(f'Evolución de Hit Direction - {instrument} - {model}')
                        plt.ylabel('Hit Direction %')
                        plt.xlabel('Período')
                        plt.xticks(rotation=45)
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        plt.savefig(charts_dir / f"{instrument}_{model}_hitdir_evolution.png", dpi=300)
                        plt.close()
                    
                    # 3. Dashboard con múltiples métricas
                    metrics_to_plot = ['RMSE', 'MAE', 'R2', 'Hit_Direction']
                    available_metrics = [m for m in metrics_to_plot if m in df_metrics.columns]
                    
                    if len(available_metrics) >= 2:
                        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 3*len(available_metrics)))
                        
                        for i, metric in enumerate(available_metrics):
                            ax = axes[i] if len(available_metrics) > 1 else axes
                            ax.plot(df_metrics['start'], df_metrics[metric], marker='o', linewidth=2)
                            ax.set_title(f'{metric} - {instrument} - {model}')
                            ax.grid(True, linestyle='--', alpha=0.7)
                            ax.tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        plt.savefig(charts_dir / f"{instrument}_{model}_metrics_dashboard.png", dpi=300)
                        plt.close()
        
        logging.info(f"Visualizaciones de subperíodos generadas en {charts_dir}")
    
    except Exception as e:
        logging.error(f"Error generando visualizaciones de subperíodos: {str(e)}")

def save_metrics_report(results: Dict[str, Dict[str, Dict[str, Any]]], output_dir: Path):
    """
    Guarda el reporte de métricas en formatos CSV y Excel.
    """
    try:
        # Preparar datos para el reporte
        report_data = []
        
        for instrument, models_data in results.items():
            for model, data in models_data.items():
                if 'metrics' not in data:
                    continue
                
                row = {
                    'Instrumento': instrument,
                    'Modelo': model,
                    'Observaciones': data['num_observations']
                }
                
                # Fechas del período si están disponibles
                if 'period' in data:
                    if data['period']['start'] is not None:
                        row['Fecha_Inicio'] = data['period']['start']
                    if data['period']['end'] is not None:
                        row['Fecha_Fin'] = data['period']['end']
                
                # Añadir métricas
                for metric, value in data['metrics'].items():
                    row[metric] = value
                
                report_data.append(row)
        
        if not report_data:
            logging.warning("No hay datos para el reporte de métricas")
            return
        
        # Crear DataFrame
        df_report = pd.DataFrame(report_data)
        
        # Ordenar columnas
        column_order = ['Instrumento', 'Modelo', 'Observaciones', 'Fecha_Inicio', 'Fecha_Fin']
        metric_cols = [col for col in df_report.columns if col not in column_order]
        final_cols = [col for col in column_order if col in df_report.columns] + metric_cols
        
        df_report = df_report[final_cols]
        
        # Guardar como CSV
        csv_file = output_dir / "resultados_totales.csv"
        df_report.to_csv(csv_file, index=False)
        logging.info(f"Reporte guardado en formato CSV: {csv_file}")
        
        # Guardar como Excel con formato
        excel_file = output_dir / "resultados_totales.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            df_report.to_excel(writer, sheet_name='Métricas', index=False)
            
            # Formatear
            workbook = writer.book
            worksheet = writer.sheets['Métricas']
            
            # Formato para números
            num_format = workbook.add_format({'num_format': '0.0000'})
            pct_format = workbook.add_format({'num_format': '0.00%'})
            
            # Aplicar formatos
            for i, col in enumerate(df_report.columns):
                # Ancho de columna basado en longitud del encabezado
                col_width = max(len(str(col)) + 2, 12)
                worksheet.set_column(i, i, col_width)
                
                # Formato numérico para métricas
                if col in ['RMSE', 'MAE', 'MSE', 'R2', 'R2_adjusted', 'Amplitude_Score', 'Phase_Score']:
                    worksheet.set_column(i, i, col_width, num_format)
                elif col in ['MAPE', 'Hit_Direction', 'Weighted_Hit_Direction']:
                    # Estos suelen ser porcentajes
                    worksheet.set_column(i, i, col_width, pct_format)
        
        logging.info(f"Reporte guardado en formato Excel: {excel_file}")
        
        # Guardar archivos individuales por instrumento
        for instrument, models_data in results.items():
            # Filtrar solo este instrumento
            inst_data = [row for row in report_data if row['Instrumento'] == instrument]
            
            if not inst_data:
                continue
            
            df_inst = pd.DataFrame(inst_data)
            
            # Ordenar columnas como antes
            df_inst = df_inst[[col for col in final_cols if col in df_inst.columns]]
            
            # Guardar como CSV
            inst_file = output_dir / f"{instrument}_metricas.csv"
            df_inst.to_csv(inst_file, index=False)
            logging.info(f"Reporte para {instrument} guardado: {inst_file}")
    
    except Exception as e:
        logging.error(f"Error guardando reporte de métricas: {str(e)}")

def save_subperiod_report(subperiod_results: Dict[str, Any], output_dir: Path):
    """
    Guarda el reporte de métricas por subperíodos.
    """
    try:
        if not subperiod_results or 'instruments' not in subperiod_results:
            logging.warning("No hay datos de subperíodos para el reporte")
            return
        
        # Directorio para subperíodos
        subperiod_dir = output_dir / "subperiods"
        subperiod_dir.mkdir(exist_ok=True, parents=True)
        
        # Para cada instrumento y modelo
        for instrument, instrument_data in subperiod_results['instruments'].items():
            if 'models' not in instrument_data or not instrument_data['models']:
                continue
            
            # Guardar un Excel por instrumento
            excel_file = subperiod_dir / f"{instrument}_subperiodos.xlsx"
            
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                for model, periods_data in instrument_data['models'].items():
                    if not periods_data:
                        continue
                    
                    # Crear DataFrame para este modelo
                    report_data = []
                    
                    for period in periods_data:
                        row = {
                            'Período': period['period'],
                            'Fecha_Inicio': period['start'],
                            'Fecha_Fin': period['end'],
                            'Observaciones': period['num_observations']
                        }
                        
                        # Añadir métricas
                        if 'metrics' in period:
                            for metric, value in period['metrics'].items():
                                row[metric] = value
                        
                        report_data.append(row)
                    
                    df_model = pd.DataFrame(report_data)
                    
                    # Ordenar por fecha de inicio
                    if 'Fecha_Inicio' in df_model.columns:
                        df_model['Fecha_Inicio'] = pd.to_datetime(df_model['Fecha_Inicio'])
                        df_model = df_model.sort_values('Fecha_Inicio')
                        df_model['Fecha_Inicio'] = df_model['Fecha_Inicio'].dt.strftime('%Y-%m-%d')
                    
                    # Guardar en hoja propia
                    sheet_name = model[:31]  # Excel limita nombres de hoja a 31 caracteres
                    df_model.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Formatear
                    workbook = writer.book
                    worksheet = writer.sheets[sheet_name]
                    
                    # Formatos
                    num_format = workbook.add_format({'num_format': '0.0000'})
                    pct_format = workbook.add_format({'num_format': '0.00%'})
                    
                    # Aplicar formatos
                    for i, col in enumerate(df_model.columns):
                        col_width = max(len(str(col)) + 2, 12)
                        worksheet.set_column(i, i, col_width)
                        
                        if col in ['RMSE', 'MAE', 'MSE', 'R2', 'R2_adjusted']:
                            worksheet.set_column(i, i, col_width, num_format)
                        elif col in ['MAPE', 'Hit_Direction']:
                            worksheet.set_column(i, i, col_width, pct_format)
            
            logging.info(f"Reporte de subperíodos para {instrument} guardado: {excel_file}")
    
    except Exception as e:
        logging.error(f"Error guardando reporte de subperíodos: {str(e)}")

def main():
    """Función principal del script."""
    try:
        logging.info("Iniciando proceso de backtesting")
        
        # Indicar si se dispone de visualizaciones avanzadas
        if HAS_ADVANCED_PLOTTING:
            logging.info("Utilizando visualizaciones avanzadas")
        else:
            logging.info("Utilizando visualizaciones básicas")
            
        # Buscar archivo de predicciones
        input_file = find_predictions_file()
        
        # Cargar datos
        logging.info(f"Cargando archivo: {input_file}")
        df = load_csv_file(input_file)
        logging.info(f"Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")
        
        # Normalizar DataFrame
        df = normalize_df(df)
        
        # Filtrar datos para análisis
        df_analysis = filter_data_for_analysis(df)
        logging.info(f"Datos filtrados para análisis: {len(df_analysis)} filas")
        
        if df_analysis.empty:
            logging.error("No hay datos válidos para análisis")
            return 1
        
        # Evaluar modelos por grupo
        results = evaluate_models_by_group(df_analysis)
        
        if not results:
            logging.error("No se pudieron evaluar los modelos")
            return 1
        
        # Evaluar por subperíodos
        subperiod_results = evaluate_by_subperiods(df_analysis)
        
        # Generar visualizaciones
        generate_metrics_visualizations(results, METRICS_DIR)
        
        if subperiod_results:
            generate_subperiod_visualizations(subperiod_results, METRICS_DIR)
        
        # Guardar reportes
        save_metrics_report(results, METRICS_DIR)
        
        if subperiod_results:
            save_subperiod_report(subperiod_results, METRICS_DIR)
        
        logging.info(" Proceso de backtesting completado con éxito")
        return 0
    
    except Exception as e:
        logging.error(f"Error en el proceso de backtesting: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
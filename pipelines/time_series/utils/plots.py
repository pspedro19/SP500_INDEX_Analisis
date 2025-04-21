#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidades de visualización para modelos de series temporales
============================================================
Provee funciones estandarizadas para generar gráficos y visualizaciones
para modelos ARIMA/SARIMAX, equivalentes a las del pipeline ML.

Incluye:
- Visualización de series temporales (real vs. predicción)
- Gráficos de diagnóstico (ACF, PACF, residuos)
- Visualización de forecasts con intervalos de confianza
- Comparativas de modelos
- Métricas de rendimiento
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import logging

# Configurar logger básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de estilos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Constantes
DEFAULT_FIGSIZE = (12, 6)
DEFAULT_FORECAST_FIGSIZE = (14, 8)
DEFAULT_DIAGNOSTIC_FIGSIZE = (16, 10)
DEFAULT_DPI = 300
COLORS = {
    'real': '#2C3E50',      # Azul oscuro
    'prediction': '#E74C3C', # Rojo
    'forecast': '#3498DB',   # Azul claro
    'ci_upper': '#AED6F1',   # Azul muy claro
    'ci_lower': '#AED6F1',   # Azul muy claro
    'ensemble': '#9B59B6',   # Púrpura
    'ensemble_ci': '#D2B4DE', # Púrpura claro
    'outlier': '#FF5733'     # Naranja para valores atípicos
}

def configure_axis_date(ax, date_format='%Y-%m-%d'):
    """Configura el eje X para formatear fechas correctamente."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    fig = ax.figure
    fig.autofmt_xdate()
    return ax

def plot_series_comparison(
    real_series: pd.Series, 
    pred_series: pd.Series, 
    title: str = "Real vs Predicción", 
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    save_path: Optional[Path] = None,
    plot_diff: bool = False,
    model_name: str = '',
    instrument: str = '',
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> plt.Figure:
    """
    Genera una visualización comparando la serie real con la predicción.
    
    Args:
        real_series: Serie temporal real
        pred_series: Serie temporal predicha
        title: Título del gráfico
        figsize: Tamaño de figura
        save_path: Ruta para guardar la imagen (opcional)
        plot_diff: Si es True, añade un subplot con la diferencia
        model_name: Nombre del modelo (para título)
        instrument: Instrumento (para título)
        inverse_transform: función para aplicar a los valores antes de graficar (e.g., np.exp)
    
    Returns:
        Figura de matplotlib
    """
    # Aplicar inverse transform si se provee
    if inverse_transform:
        real_plot = pd.Series(inverse_transform(real_series.values), index=real_series.index)
        pred_plot = pd.Series(inverse_transform(pred_series.values), index=pred_series.index)
        ylabel = 'Precio'
    else:
        real_plot = real_series
        pred_plot = pred_series
        ylabel = 'Valor'
    
    if plot_diff:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # Título detallado si se proporcionan detalles
    if model_name and instrument:
        full_title = f"{title} - {instrument} - {model_name}"
    else:
        full_title = title
    
    # Graficar datos
    ax1.plot(real_plot.index, real_plot, label='Real', color=COLORS['real'], linewidth=2)
    ax1.plot(pred_plot.index, pred_plot, label='Predicción', color=COLORS['prediction'], linewidth=2, linestyle='--')
    
    # Configurar gráfico
    ax1.set_title(full_title, fontsize=14)
    ax1.legend(loc='best')
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1 = configure_axis_date(ax1)
    
    # Si se solicita gráfico de diferencias
    if plot_diff:
        # Asegurar alineación de índices
        aligned_real = real_plot.reindex(pred_plot.index)
        diff = aligned_real - pred_plot
        
        # Graficar diferencias
        ax2.plot(diff.index, diff, color='#27AE60', label='Diferencia')
        ax2.axhline(y=0, color='#7F8C8D', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Diferencia')
        ax2.legend(loc='best')
        ax2 = configure_axis_date(ax2)
    
    plt.tight_layout()
    
    # Guardar si se proporciona ruta
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def plot_forecast(
    historical: pd.Series,
    forecast: pd.Series,                      # Required parameter first
    pred_series: Optional[pd.Series] = None,  # Optional parameter after
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
    title: str = "Pronóstico",
    figsize: Tuple[int, int] = DEFAULT_FORECAST_FIGSIZE,
    save_path: Optional[Path] = None,
    instrument: str = '',
    model_name: str = '',
    forecast_periods: int = 21,
    train_end_date: Optional[pd.Timestamp] = None,
    val_end_date: Optional[pd.Timestamp] = None,
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> plt.Figure:
    """
    Genera una visualización del pronóstico con intervalos de confianza.
    
    Args:
        historical: Serie histórica
        forecast: Serie de pronóstico
        ci_lower: Serie con límite inferior del intervalo de confianza
        ci_upper: Serie con límite superior del intervalo de confianza
        title: Título del gráfico
        figsize: Tamaño de figura
        save_path: Ruta para guardar la imagen
        instrument: Instrumento financiero
        model_name: Nombre del modelo
        forecast_periods: Número de períodos para pronóstico
        train_end_date: Fecha de fin de entrenamiento para marcar con línea vertical
        val_end_date: Fecha de fin de validación para marcar con línea vertical
        inverse_transform: función para aplicar a los valores antes de graficar (e.g., np.exp)
    
    Returns:
        Figura de matplotlib
    """
    # Aplicar inverse transform si se provee
    if inverse_transform:
        hist_plot = pd.Series(inverse_transform(historical.values), index=historical.index)
        fc_plot = pd.Series(inverse_transform(forecast.values), index=forecast.index)
        ci_low = pd.Series(inverse_transform(ci_lower.values), index=ci_lower.index) if ci_lower is not None else None
        ci_up = pd.Series(inverse_transform(ci_upper.values), index=ci_upper.index) if ci_upper is not None else None
        ylabel = 'Precio'
    else:
        hist_plot, fc_plot, ci_low, ci_up = historical, forecast, ci_lower, ci_upper
        ylabel = 'Valor'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Título detallado
    if model_name and instrument:
        full_title = f"{title} - {instrument} - {model_name}"
    else:
        full_title = title
    
    # Determinar datos a mostrar (últimos N puntos + pronóstico)
    history_points = min(len(hist_plot), forecast_periods * 3)  # Mostrar 3x más historia que pronóstico
    hist_subset = hist_plot.iloc[-history_points:]
    
    # Graficar datos históricos
    ax.plot(hist_subset.index, hist_subset, 
            label='Histórico', color=COLORS['real'], linewidth=2)
    
        # — Mostrar la predicción en validación/test si existe —
    if pred_series is not None:
        ps = (pd.Series(inverse_transform(pred_series.values), index=pred_series.index)
            if inverse_transform else pred_series)
        ax.plot(ps.index, ps,
                label='Predicción (Val/Test)',
                color=COLORS['prediction'],
                linestyle='--', linewidth=2)

    # — Líneas verticales para cortes —
    if train_end_date is not None:
        ax.axvline(x=train_end_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.annotate('FIN TRAIN',
                xy=(train_end_date, ax.get_ylim()[0]),
                xytext=(5, 15), textcoords='offset points',
                color='gray', fontweight='bold', fontsize=10)
    if val_end_date is not None:
        ax.axvline(x=val_end_date, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.annotate('FIN VALIDACIÓN',
                xy=(val_end_date, ax.get_ylim()[0]),
                xytext=(5, 15), textcoords='offset points',
                color='black', fontweight='bold', fontsize=10)
    
    # Verificar que forecast tenga datos válidos
    if len(fc_plot) == 0:
        logger.warning("El pronóstico está vacío, no hay datos para graficar")
        ax.text(0.5, 0.5, "SIN DATOS DE PRONÓSTICO", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color='red')
    else:
        # Graficar pronóstico con mayor prominencia
        ax.plot(fc_plot.index, fc_plot, 
                label=f'Pronóstico ({len(fc_plot)} días)', 
                color=COLORS['forecast'], 
                linewidth=3.0,
                marker='o',  # Añadir marcador para mayor visibilidad
                markersize=4)
        
        # Verificar valores atípicos o extremos en el pronóstico
        if len(fc_plot) >= 2:  # Solo si hay suficientes datos para calcular std
            forecast_mean = fc_plot.mean()
            forecast_std = fc_plot.std()
            limits = (forecast_mean - 3*forecast_std, forecast_mean + 3*forecast_std)
            
            # Destacar puntos fuera de 3 desviaciones estándar
            outliers = fc_plot[(fc_plot < limits[0]) | (fc_plot > limits[1])]
            if not outliers.empty:
                ax.scatter(outliers.index, outliers, 
                         color=COLORS['outlier'], s=80, zorder=5, label='Valores atípicos')
    
    # Añadir línea vertical más prominente y clara para el inicio del pronóstico
    last_historical_date = hist_plot.index[-1]
    ax.axvline(x=last_historical_date, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
    
    # Etiqueta más visible y mejorada
    ax.annotate('INICIO PRONÓSTICO', 
               xy=(last_historical_date, ax.get_ylim()[0]),
               xytext=(10, 30),
               textcoords='offset points',
               arrowprops=dict(arrowstyle='->', lw=1.5, color='#E74C3C'),
               color='#E74C3C',
               fontweight='bold',
               fontsize=12)
    
    # Añadir líneas verticales para marcar fin de entrenamiento y validación si se proporcionan
    if train_end_date is not None:
        ax.axvline(x=train_end_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.annotate('FIN TRAIN', 
                   xy=(train_end_date, ax.get_ylim()[0]),
                   xytext=(5, 15),
                   textcoords='offset points',
                   color='gray',
                   fontweight='bold',
                   fontsize=10)
    
    if val_end_date is not None:
        ax.axvline(x=val_end_date, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.annotate('FIN VALIDACIÓN', 
                   xy=(val_end_date, ax.get_ylim()[0]),
                   xytext=(5, 15),
                   textcoords='offset points',
                   color='black',
                   fontweight='bold',
                   fontsize=10)
    
    # Añadir intervalos de confianza si están disponibles
    if ci_low is not None and ci_up is not None:
        ax.fill_between(fc_plot.index, ci_low, ci_up, 
                        color=COLORS['ci_lower'], alpha=0.3, label='95% CI')
    
    # Configurar gráfico
    ax.set_title(full_title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax = configure_axis_date(ax)
    
    plt.tight_layout()
    
    # Guardar si se proporciona ruta
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def plot_ensemble_comparison(
    historical: pd.Series,
    forecasts: Dict[str, pd.Series],
    ensemble_forecast: pd.Series,
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
    title: str = "Comparación de Modelos y Ensemble",
    figsize: Tuple[int, int] = DEFAULT_FORECAST_FIGSIZE,
    save_path: Optional[Path] = None,
    instrument: str = '',
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> plt.Figure:
    """
    Genera una visualización comparando pronósticos de diferentes modelos con el ensemble.
    
    Args:
        historical: Serie histórica
        forecasts: Diccionario con series de pronóstico por modelo
        ensemble_forecast: Serie de pronóstico del ensemble
        ci_lower: Límite inferior del intervalo de confianza del ensemble
        ci_upper: Límite superior del intervalo de confianza del ensemble
        title: Título del gráfico
        figsize: Tamaño de figura
        save_path: Ruta para guardar la imagen
        instrument: Instrumento financiero
        inverse_transform: función para aplicar a los valores antes de graficar (e.g., np.exp)
    
    Returns:
        Figura de matplotlib
    """
    # Aplicar inverse transform si se provee
    if inverse_transform:
        hist_plot = pd.Series(inverse_transform(historical.values), index=historical.index)
        ens_plot = pd.Series(inverse_transform(ensemble_forecast.values), index=ensemble_forecast.index)
        fc_plots = {}
        for model_name, fc in forecasts.items():
            fc_plots[model_name] = pd.Series(inverse_transform(fc.values), index=fc.index)
        ci_low = pd.Series(inverse_transform(ci_lower.values), index=ci_lower.index) if ci_lower is not None else None
        ci_up = pd.Series(inverse_transform(ci_upper.values), index=ci_upper.index) if ci_upper is not None else None
        ylabel = 'Precio'
    else:
        hist_plot = historical
        ens_plot = ensemble_forecast
        fc_plots = forecasts
        ci_low = ci_lower
        ci_up = ci_upper
        ylabel = 'Valor'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Título detallado
    if instrument:
        full_title = f"{title} - {instrument}"
    else:
        full_title = title
    
    # Determinar datos a mostrar (últimos N puntos)
    forecast_periods = len(ens_plot)
    history_points = min(len(hist_plot), forecast_periods * 2)  # Mostrar 2x más historia que pronóstico
    hist_subset = hist_plot.iloc[-history_points:]
    
    # Graficar datos históricos
    ax.plot(hist_subset.index, hist_subset, 
            label='Histórico', color=COLORS['real'], linewidth=2.5)
    
    # Graficar pronósticos individuales con líneas más delgadas
    for model_name, forecast in fc_plots.items():
        ax.plot(forecast.index, forecast, 
                label=model_name, alpha=0.7, linewidth=1, linestyle='--')
    
    # Graficar el ensemble con línea más destacada
    ax.plot(ens_plot.index, ens_plot, 
            label='ENSEMBLE', color=COLORS['ensemble'], linewidth=2.5)
    
    # Añadir intervalos de confianza del ensemble si están disponibles
    if ci_low is not None and ci_up is not None:
        ax.fill_between(ens_plot.index, ci_low, ci_up, 
                        color=COLORS['ensemble_ci'], alpha=0.2, label='95% CI (ENSEMBLE)')
    
    # Añadir línea vertical separando histórico y pronóstico
    last_historical_date = hist_plot.index[-1]
    ax.axvline(x=last_historical_date, color='#95A5A6', linestyle='--', alpha=0.5)
    
    # Configurar gráfico
    ax.set_title(full_title, fontsize=14)
    ax.legend(loc='best')
    ax.set_ylabel(ylabel, fontsize=12)
    ax = configure_axis_date(ax)
    
    plt.tight_layout()
    
    # Guardar si se proporciona ruta
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def plot_model_diagnostics(
    model_fit: Any,  # Resultado de ajuste de modelo statsmodels
    residuals: pd.Series,
    title: str = "Diagnóstico de Modelo",
    figsize: Tuple[int, int] = DEFAULT_DIAGNOSTIC_FIGSIZE,
    save_path: Optional[Path] = None,
    instrument: str = '',
    model_name: str = ''
) -> plt.Figure:
    """
    Genera visualizaciones de diagnóstico para el modelo de series temporales.
    
    Args:
        model_fit: Objeto de ajuste de modelo (de statsmodels)
        residuals: Serie de residuos
        title: Título del gráfico
        figsize: Tamaño de figura
        save_path: Ruta para guardar la imagen
        instrument: Instrumento financiero
        model_name: Nombre del modelo
    
    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Título detallado
    if model_name and instrument:
        full_title = f"{title} - {instrument} - {model_name}"
    else:
        full_title = title
    
    fig.suptitle(full_title, fontsize=16)
    
    # 1. Residuos a lo largo del tiempo
    axes[0, 0].plot(residuals.index, residuals, color='#2980B9')
    axes[0, 0].axhline(y=0, color='#E74C3C', linestyle='--')
    axes[0, 0].set_title('Residuos vs Tiempo')
    configure_axis_date(axes[0, 0])
    
    # 2. Histograma y densidad de residuos
    sns.histplot(residuals, kde=True, ax=axes[0, 1], color='#2980B9')
    axes[0, 1].set_title('Distribución de Residuos')
    
    # 3. ACF de residuos
    plot_acf(residuals.values, ax=axes[1, 0], lags=min(30, len(residuals)//4), alpha=0.05)
    axes[1, 0].set_title('ACF de Residuos')
    
    # 4. PACF de residuos
    plot_pacf(residuals.values, ax=axes[1, 1], lags=min(30, len(residuals)//4), alpha=0.05)
    axes[1, 1].set_title('PACF de Residuos')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Ajustar espacio para título general
    
    # Guardar si se proporciona ruta
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str = "RMSE",
    title: str = "Comparación de Modelos",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    instrument: str = '',
    is_lower_better: bool = True
) -> plt.Figure:
    """
    Genera visualización de comparación de métricas entre modelos.
    
    Args:
        metrics: Diccionario de métricas por modelo
        metric_name: Nombre de la métrica a visualizar
        title: Título del gráfico
        figsize: Tamaño de figura
        save_path: Ruta para guardar la imagen
        instrument: Instrumento financiero
        is_lower_better: Si es True, valores más bajos son mejores
    
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Título detallado
    if instrument:
        full_title = f"{title} - {metric_name} - {instrument}"
    else:
        full_title = f"{title} - {metric_name}"
    
    # Preparar datos
    models = []
    values = []
    
    for model_name, model_metrics in metrics.items():
        if metric_name in model_metrics:
            models.append(model_name)
            values.append(model_metrics[metric_name])
    
    if not models:
        plt.close(fig)
        return None
    
    # Ordenar por valor (ascendente o descendente según is_lower_better)
    sorted_indices = np.argsort(values)
    if not is_lower_better:
        sorted_indices = sorted_indices[::-1]
    
    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    # Crear barras con colores especiales para ENSEMBLE
    colors = ['#9B59B6' if model == 'ENSEMBLE' else '#3498DB' for model in sorted_models]
    bars = ax.bar(sorted_models, sorted_values, color=colors)
    
    # Configurar gráfico
    ax.set_title(full_title, fontsize=14)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel('Modelo', fontsize=12)
    
    # Añadir valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(sorted_values),
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardar si se proporciona ruta
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def plot_metrics_radar(
    metrics: Dict[str, Dict[str, float]],
    metric_names: List[str],
    title: str = "Métricas por Modelo",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    instrument: str = '',
    is_lower_better: Dict[str, bool] = None
) -> plt.Figure:
    """
    Genera un gráfico de radar comparando métricas entre modelos.
    
    Args:
        metrics: Diccionario de métricas por modelo
        metric_names: Lista de nombres de métricas a incluir
        title: Título del gráfico
        figsize: Tamaño de figura
        save_path: Ruta para guardar la imagen
        instrument: Instrumento financiero
        is_lower_better: Diccionario indicando si valores más bajos son mejores para cada métrica
    
    Returns:
        Figura de matplotlib
    """
    # Valor predeterminado para is_lower_better
    if is_lower_better is None:
        is_lower_better = {
            'RMSE': True, 'MAE': True, 'MAPE': True, 'MSE': True,
            'R2': False, 'R2_adjusted': False, 'Hit_Direction': False,
            'Amplitude_Score': False, 'Phase_Score': False, 'Weighted_Hit_Direction': False
        }
    
    # Filtrar solo métricas disponibles en todos los modelos
    available_metrics = []
    for metric in metric_names:
        if all(metric in m for m in metrics.values()):
            available_metrics.append(metric)
    
    if not available_metrics:
        return None
    
    # Preparar gráfico de radar
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)
    
    # Título detallado
    if instrument:
        full_title = f"{title} - {instrument}"
    else:
        full_title = title
    
    # Número de variables
    N = len(available_metrics)
    
    # Ángulos para cada métrica
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Cerrar el polígono
    
    # Normalizar métricas para visualización en radar
    normalized_metrics = {}
    
    for model_name, model_metrics in metrics.items():
        normalized_metrics[model_name] = []
        
        for i, metric in enumerate(available_metrics):
            if metric in model_metrics:
                value = model_metrics[metric]
                
                # Almacenar valor para normalización entre todos los modelos
                if i not in normalized_metrics:
                    normalized_metrics[i] = []
                    
                if i not in normalized_metrics:
                    normalized_metrics[i] = []
                
                normalized_metrics[model_name].append(value)
            else:
                normalized_metrics[model_name].append(0)  # Valor predeterminado si falta
    
    # Normalizar todas las métricas de 0 a 1 (1 siendo el mejor)
    for i, metric in enumerate(available_metrics):
        values = [metrics[model][metric] for model in metrics if metric in metrics[model]]
        
        if not values:
            continue
            
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            # Evitar división por cero
            for model_name in normalized_metrics:
                if isinstance(model_name, str):  # Asegurarse de que es nombre de modelo
                    normalized_metrics[model_name][i] = 1.0
        else:
            for model_name in normalized_metrics:
                if isinstance(model_name, str):  # Asegurarse de que es nombre de modelo
                    value = normalized_metrics[model_name][i]
                    # Normalizar según si valores más bajos son mejores
                    if metric in is_lower_better and is_lower_better[metric]:
                        normalized_val = 1 - ((value - min_val) / (max_val - min_val))
                    else:
                        normalized_val = (value - min_val) / (max_val - min_val)
                    
                    normalized_metrics[model_name][i] = normalized_val
    
    # Dibujar para cada modelo
    for model_name in metrics:
        if model_name not in normalized_metrics:
            continue
            
        values = normalized_metrics[model_name]
        values += values[:1]  # Cerrar el polígono
        
        # Dibujar polígono y puntos
        color = '#9B59B6' if model_name == 'ENSEMBLE' else None
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Etiquetar ejes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics)
    
    # Ajustar el título y la leyenda
    plt.title(full_title, size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    
    # Guardar si se proporciona ruta
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def plot_metric_evolution(
    metric_values: pd.DataFrame,
    metric_name: str,
    title: str = "Evolución de Métricas",
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    save_path: Optional[Path] = None,
    instrument: str = '',
    model_name: str = ''
) -> plt.Figure:
    """
    Genera una visualización de la evolución de una métrica a lo largo del tiempo.
    
    Args:
        metric_values: DataFrame con valores de la métrica (índice = fechas)
        metric_name: Nombre de la métrica a visualizar
        title: Título del gráfico
        figsize: Tamaño de figura
        save_path: Ruta para guardar la imagen
        instrument: Instrumento financiero
        model_name: Nombre del modelo
    
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Título detallado
    if model_name and instrument:
        full_title = f"{title} - {metric_name} - {instrument} - {model_name}"
    else:
        full_title = f"{title} - {metric_name}"
    
    # Graficar línea de evolución
    ax.plot(metric_values.index, metric_values[metric_name], 
            marker='o', linewidth=2, color='#3498DB')
    
    # Añadir línea de tendencia (media móvil)
    if len(metric_values) > 3:
        window = min(5, len(metric_values) // 2)
        if window > 0:
            rolling_mean = metric_values[metric_name].rolling(window=window).mean()
            ax.plot(metric_values.index, rolling_mean, 
                    color='#E74C3C', linewidth=2, linestyle='--', 
                    label=f'Media Móvil ({window} períodos)')
    
    # Configurar gráfico
    ax.set_title(full_title, fontsize=14)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if len(metric_values) > 0:
        # Añadir anotación con último valor
        last_value = metric_values[metric_name].iloc[-1]
        ax.annotate(f'{last_value:.4f}', 
                   xy=(metric_values.index[-1], last_value),
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    ax.legend(loc='best')
    configure_axis_date(ax)
    
    plt.tight_layout()
    
    # Guardar si se proporciona ruta
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def create_dashboard(
    real_series: pd.Series,
    pred_series: pd.Series,
    forecast_series: pd.Series,
    metrics: Dict[str, float],
    title: str = "Dashboard de Modelo",
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[Path] = None,
    instrument: str = '',
    model_name: str = '',
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
    train_end_date: Optional[pd.Timestamp] = None,
    val_end_date: Optional[pd.Timestamp] = None,
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> plt.Figure:
    """
    Crea un dashboard completo para un modelo, incluyendo serie histórica,
    pronóstico y métricas clave.
    
    Args:
        real_series: Serie real histórica
        pred_series: Serie predicha (para evaluación)
        forecast_series: Serie de pronóstico futuro
        metrics: Diccionario con métricas del modelo
        title: Título del dashboard
        figsize: Tamaño de figura
        save_path: Ruta para guardar la imagen
        instrument: Instrumento financiero
        model_name: Nombre del modelo
        ci_lower: Límite inferior del intervalo de confianza del pronóstico
        ci_upper: Límite superior del intervalo de confianza del pronóstico
        train_end_date: Fecha de fin de entrenamiento para marcar con línea vertical
        val_end_date: Fecha de fin de validación para marcar con línea vertical
        inverse_transform: función para aplicar a los valores antes de graficar (e.g., np.exp)
    
    Returns:
        Figura de matplotlib
    """
    # Aplicar inverse transform si se provee
    if inverse_transform:
        real_plot = pd.Series(inverse_transform(real_series.values), index=real_series.index)
        pred_plot = pd.Series(inverse_transform(pred_series.values), index=pred_series.index)
        fc_plot = pd.Series(inverse_transform(forecast_series.values), index=forecast_series.index)
        ci_low = pd.Series(inverse_transform(ci_lower.values), index=ci_lower.index) if ci_lower is not None else None
        ci_up = pd.Series(inverse_transform(ci_upper.values), index=ci_upper.index) if ci_upper is not None else None
        ylabel = 'Precio'
    else:
        real_plot = real_series
        pred_plot = pred_series
        fc_plot = forecast_series
        ci_low = ci_lower
        ci_up = ci_upper
        ylabel = 'Valor'
    
    fig = plt.figure(figsize=figsize)
    
    # Definir grid para el dashboard
    gs = fig.add_gridspec(3, 3)
    
    # Título detallado
    if model_name and instrument:
        full_title = f"{title} - {instrument} - {model_name}"
    else:
        full_title = title
    
    fig.suptitle(full_title, fontsize=16, y=0.98)
    
    # 1. Serie histórica y predicción
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(real_plot.index, real_plot, label='Real', color=COLORS['real'], linewidth=2)
    ax1.plot(pred_plot.index, pred_plot, label='Predicción', color=COLORS['prediction'], 
            linewidth=2, linestyle='--')
    
    # Añadir líneas verticales para marcar fin de entrenamiento y validación
    if train_end_date is not None:
        ax1.axvline(x=train_end_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.annotate('FIN TRAIN', 
                   xy=(train_end_date, ax1.get_ylim()[0]),
                   xytext=(5, 15),
                   textcoords='offset points',
                   color='gray',
                   fontweight='bold',
                   fontsize=10)
    
    if val_end_date is not None:
        ax1.axvline(x=val_end_date, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.annotate('FIN VALIDACIÓN', 
                   xy=(val_end_date, ax1.get_ylim()[0]),
                   xytext=(5, 15),
                   textcoords='offset points',
                   color='black',
                   fontweight='bold',
                   fontsize=10)
    
    ax1.set_title('Serie Histórica vs Predicción', fontsize=12)
    ax1.legend(loc='best')
    ax1.set_ylabel(ylabel, fontsize=12)
    configure_axis_date(ax1)
    
    # 2. Pronóstico futuro con intervalo de confianza
    ax2 = fig.add_subplot(gs[1, :])
    
    # Mostrar solo los últimos puntos de la serie real
    history_points = min(len(real_plot), len(fc_plot) * 2)
    hist_subset = real_plot.iloc[-history_points:]
    
    ax2.plot(hist_subset.index, hist_subset, label='Histórico', color=COLORS['real'], linewidth=2)
    
    # Verificar que forecast tenga datos válidos
    if len(fc_plot) == 0:
        logger.warning("El pronóstico está vacío, no hay datos para graficar en el dashboard")
        ax2.text(0.5, 0.5, "SIN DATOS DE PRONÓSTICO", 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14, color='red')
    else:
        # Graficar pronóstico con mayor prominencia
        ax2.plot(fc_plot.index, fc_plot, 
                label=f'Pronóstico ({len(fc_plot)} días)', 
                color=COLORS['forecast'], 
                linewidth=3.0,
                marker='o',
                markersize=4)
                
        if ci_low is not None and ci_up is not None:
            ax2.fill_between(fc_plot.index, ci_low, ci_up, 
                            color=COLORS['ci_lower'], alpha=0.3, label='95% CI')
    
        # Añadir línea vertical más prominente
        last_historical_date = real_plot.index[-1]
        ax2.axvline(x=last_historical_date, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
        
        # Añadir etiqueta para inicio de pronóstico
        ax2.annotate('INICIO PRONÓSTICO', 
                   xy=(last_historical_date, ax2.get_ylim()[0]),
                   xytext=(10, 30),
                   textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#E74C3C'),
                   color='#E74C3C',
                   fontweight='bold',
                   fontsize=12)
    
    # Añadir líneas verticales para marcar fin de entrenamiento y validación
    if train_end_date is not None:
        ax2.axvline(x=train_end_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.annotate('FIN TRAIN', 
                   xy=(train_end_date, ax2.get_ylim()[0]),
                   xytext=(5, 15),
                   textcoords='offset points',
                   color='gray',
                   fontweight='bold',
                   fontsize=10)
    
    if val_end_date is not None:
        ax2.axvline(x=val_end_date, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.annotate('FIN VALIDACIÓN', 
                   xy=(val_end_date, ax2.get_ylim()[0]),
                   xytext=(5, 15),
                   textcoords='offset points',
                   color='black',
                   fontweight='bold',
                   fontsize=10)
    
    ax2.set_title('Pronóstico', fontsize=12)
    ax2.legend(loc='best')
    ax2.set_ylabel(ylabel, fontsize=12)
    configure_axis_date(ax2)
    
    # 3. Barplot de métricas clave
    ax3 = fig.add_subplot(gs[2, 0:2])
    
    # Seleccionar métricas principales para mostrar
    key_metrics = ['RMSE', 'MAE', 'Hit_Direction', 'R2']
    available_metrics = [m for m in key_metrics if m in metrics]
    
    if available_metrics:
        metric_values = [metrics[m] for m in available_metrics]
        bars = ax3.bar(available_metrics, metric_values, color='#3498DB')
        
        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(metric_values),
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax3.set_title('Métricas Clave', fontsize=12)
    
    # 4. Tabla de todas las métricas
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.axis('off')
    
    if metrics:
        metrics_text = "MÉTRICAS:\n\n"
        for metric, value in metrics.items():
            metrics_text += f"{metric}: {value:.4f}\n"
        
        ax4.text(0, 0.9, metrics_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Ajustar espacio para título general
    
    # Guardar si se proporciona ruta
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def ensure_directory(path: Path) -> Path:
    """Asegura que el directorio existe, lo crea si es necesario."""
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# Función de conveniencia para generar gráficos completos para un modelo
# Para plots.py: modificación de la función generate_model_plots para hacerla más compatible

def generate_model_visualizations(
    model: Any,
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    exog_cols: Optional[List[str]],
    test_pred: np.ndarray,
    forecast_df: pd.DataFrame,
    metrics: Dict[str, Any],  # Changed from ModelMetrics to Dict[str, Any]
    model_type: str,
    instrument: str,
    output_dir: Path,
    df_train: Optional[pd.DataFrame] = None,
    df_val: Optional[pd.DataFrame] = None
) -> Dict[str, Path]:
    """
    Genera visualizaciones para el modelo y las guarda en el directorio de salida.
    
    Args:
        model: Modelo entrenado
        df: DataFrame con datos completos
        df_test: DataFrame de test
        target_col: Columna objetivo
        exog_cols: Columnas exógenas
        test_pred: Predicciones en test
        forecast_df: DataFrame con pronóstico
        metrics: Métricas del modelo
        model_type: Tipo de modelo
        instrument: Instrumento
        output_dir: Directorio de salida
        df_train: DataFrame de entrenamiento (opcional)
        df_val: DataFrame de validación (opcional)
    
    Returns:
        Diccionario con rutas a las visualizaciones generadas
    """
    # Verificar si el módulo de visualizaciones está disponible
    if 'plots' not in globals():
        return {}
        
    logger = get_logger(__name__, instrument=instrument, model_type=model_type)
    
    # Crear directorio para gráficos
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True, parents=True)
    
    # Rutas para los gráficos
    paths = {}
    
    try:
        # Series reales y predichas
        y_train = df[target_col]
        y_test = df_test[target_col]
        
        # Test predictions como Series para facilitar visualización
        test_pred_series = pd.Series(test_pred, index=df_test.index)
        
        # Determinar fechas de fin de entrenamiento y validación
        train_end_date = None
        val_end_date = None
        
        if df_train is not None and len(df_train) > 0:
            train_end_date = df_train.index[-1]
        
        if df_val is not None and len(df_val) > 0:
            val_end_date = df_val.index[-1]
        else:
            # Si no se proporciona df_val, podemos inferir la fecha de fin de validación
            # como la fecha justo antes del inicio de test
            if len(df_test) > 0:
                val_end_date = df_test.index[0] - pd.Timedelta(days=1)
        
        # 1. Gráfico de comparación real vs predicción
        comparison_path = charts_dir / f"{instrument}_{model_type}_comparison.png"
        plots.plot_series_comparison(
            real_series=y_test,
            pred_series=test_pred_series,
            title="Real vs Predicción",
            save_path=comparison_path,
            plot_diff=True,
            model_name=model_type,
            instrument=instrument
        )
        paths['comparison'] = comparison_path
        logger.info(f"Gráfico de comparación guardado en {comparison_path}")
        
        # 2. Gráfico de pronóstico
        forecast_series = forecast_df['forecast']
        ci_lower = forecast_df['lower_95'] if 'lower_95' in forecast_df else None
        ci_upper = forecast_df['upper_95'] if 'upper_95' in forecast_df else None
        
        forecast_path = charts_dir / f"{instrument}_{model_type}_forecast.png"
        
        # Primero intentar con la versión más simple (sin train_end_date ni val_end_date)
        try:
            plots.plot_forecast(
                historical=y_train,
                forecast=forecast_series,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                title="Pronóstico",
                save_path=forecast_path,
                instrument=instrument,
                model_name=model_type
            )
        except Exception as e:
            # Si falla, probar con la versión que incluye las fechas
            logger.warning(f"Error en primera llamada a plot_forecast: {str(e)}. Intentando alternativa.")
            try:
                # Si esta versión está disponible, usarla
                plots.plot_forecast(
                    historical=y_train,
                    forecast=forecast_series,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    title="Pronóstico",
                    save_path=forecast_path,
                    instrument=instrument,
                    model_name=model_type,
                    train_end_date=train_end_date,
                    val_end_date=val_end_date
                )
            except Exception as e2:
                # Si ambas fallan, hacer un gráfico básico como respaldo
                logger.warning(f"Error en segunda llamada a plot_forecast: {e2}. Creando gráfico básico.")
                
                # Crear un gráfico simple con matplotlib
                plt.figure(figsize=(12, 6))
                plt.plot(y_train.iloc[-90:].index, y_train.iloc[-90:], label='Histórico', color='blue')
                plt.plot(forecast_series.index, forecast_series, label='Pronóstico', color='red', 
                        linewidth=2, linestyle='--')
                
                if ci_lower is not None and ci_upper is not None:
                    plt.fill_between(forecast_series.index, ci_lower, ci_upper, alpha=0.3, color='lightblue')
                
                plt.title(f"Pronóstico - {instrument} - {model_type}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(forecast_path, dpi=300)
                plt.close()
        
        paths['forecast'] = forecast_path
        logger.info(f"Gráfico de pronóstico guardado en {forecast_path}")
        
        # 3. Gráfico de diagnóstico (residuos)
        try:
            # Obtener residuos para diagnóstico
            residuals = pd.Series(y_test.values - test_pred, index=y_test.index)
            
            diagnostic_path = charts_dir / f"{instrument}_{model_type}_diagnostics.png"
            plots.plot_model_diagnostics(
                model_fit=model,
                residuals=residuals,
                title="Diagnóstico de Modelo",
                save_path=diagnostic_path,
                instrument=instrument,
                model_name=model_type
            )
            paths['diagnostics'] = diagnostic_path
            logger.info(f"Gráfico de diagnóstico guardado en {diagnostic_path}")
        except Exception as e:
            logger.warning(f"Error generando diagnóstico: {str(e)}")
        
        # 4. Dashboard completo
        metrics_dict = metrics.to_dict()
        dashboard_path = charts_dir / f"{instrument}_{model_type}_dashboard.png"
        
        # Primero intentar sin train_end_date ni val_end_date
        try:
            plots.create_dashboard(
                real_series=y_test,
                pred_series=test_pred_series,
                forecast_series=forecast_series,
                metrics=metrics_dict,
                title="Dashboard",
                save_path=dashboard_path,
                instrument=instrument,
                model_name=model_type,
                ci_lower=ci_lower,
                ci_upper=ci_upper
            )
        except Exception as e:
            # Si falla, intentar con la versión que incluye las fechas
            logger.warning(f"Error en primera llamada a create_dashboard: {str(e)}. Intentando alternativa.")
            try:
                plots.create_dashboard(
                    real_series=y_test,
                    pred_series=test_pred_series,
                    forecast_series=forecast_series,
                    metrics=metrics_dict,
                    title="Dashboard",
                    save_path=dashboard_path,
                    instrument=instrument,
                    model_name=model_type,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    train_end_date=train_end_date,
                    val_end_date=val_end_date
                )
            except Exception as e2:
                logger.warning(f"Error en segunda llamada a create_dashboard: {e2}. No se generó el dashboard.")
                
        paths['dashboard'] = dashboard_path
        logger.info(f"Dashboard guardado en {dashboard_path}")
        
        # 5. Línea de tiempo completa
        if df_train is None or df_val is None:
            df_train, df_val, _ = split_data(df)
            
        timeline_path = generate_comprehensive_timeline(
            train_df=df_train,
            val_df=df_val,
            test_df=df_test,
            forecast_df=forecast_df,
            instrument=instrument,
            model_type=model_type,
            output_dir=charts_dir
        )
        paths['timeline'] = timeline_path
        logger.info(f"Línea de tiempo guardada en {timeline_path}")
        
    except Exception as e:
        logger.error(f"Error generando visualizaciones: {str(e)}")
        logger.error(f"Traza detallada: {traceback.format_exc()}")
    
    return paths

def generate_model_plots(
    real_series: pd.Series,
    pred_series: pd.Series,
    forecast_series: pd.Series,
    metrics: Dict[str, float],
    residuals: pd.Series,
    model_fit: Any,  # Resultado de ajuste de modelo statsmodels
    output_dir: Path,
    instrument: str,
    model_name: str,
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Path]:
    """
    Genera y guarda todos los gráficos relevantes para un modelo.
    
    Args:
        real_series: Serie real histórica
        pred_series: Serie predicha (para evaluación)
        forecast_series: Serie de pronóstico futuro
        metrics: Diccionario con métricas del modelo
        residuals: Serie de residuos del modelo
        model_fit: Objeto de ajuste de modelo (de statsmodels)
        output_dir: Directorio base de salida
        instrument: Instrumento financiero
        model_name: Nombre del modelo
        ci_lower: Límite inferior del intervalo de confianza
        ci_upper: Límite superior del intervalo de confianza
        inverse_transform: función para aplicar a los valores antes de graficar (e.g., np.exp)
    
    Returns:
        Diccionario con rutas a los gráficos generados
    """
    # Asegurar que existen los directorios
    charts_dir = ensure_directory(output_dir / "charts")
    
    # Definir rutas para cada gráfico
    paths = {
        'comparison': charts_dir / f"{instrument}_{model_name}_comparison.png",
        'forecast': charts_dir / f"{instrument}_{model_name}_forecast.png",
        'diagnostics': charts_dir / f"{instrument}_{model_name}_diagnostics.png",
        'dashboard': charts_dir / f"{instrument}_{model_name}_dashboard.png"
    }
    
    # Generar gráficos
    plot_series_comparison(
        real_series=real_series,
        pred_series=pred_series,
        title="Real vs Predicción",
        save_path=paths['comparison'],
        plot_diff=True,
        instrument=instrument,
        model_name=model_name,
        inverse_transform=inverse_transform
    )
    
    # Versión modificada para evitar pasar train_end_date y val_end_date
    try:
        plot_forecast(
            historical=real_series,
            forecast=forecast_series,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            title="Pronóstico",
            save_path=paths['forecast'],
            instrument=instrument,
            model_name=model_name,
            inverse_transform=inverse_transform
            # train_end_date y val_end_date eliminados para mayor compatibilidad
        )
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            # Enfoque alternativo si hay incompatibilidad de parámetros
            logger.warning(f"Usando llamada simplificada a plot_forecast debido a error: {str(e)}")
            plot_forecast(
                historical=real_series,
                forecast=forecast_series,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                title="Pronóstico",
                save_path=paths['forecast'],
                instrument=instrument,
                model_name=model_name
            )
        else:
            raise
    
    try:
        plot_model_diagnostics(
            model_fit=model_fit,
            residuals=residuals,
            title="Diagnóstico de Modelo",
            save_path=paths['diagnostics'],
            instrument=instrument,
            model_name=model_name
        )
    except Exception as e:
        logger.warning(f"Error generando diagnósticos: {str(e)}")
    
    # Versión modificada para evitar pasar train_end_date y val_end_date
    try:
        create_dashboard(
            real_series=real_series,
            pred_series=pred_series,
            forecast_series=forecast_series,
            metrics=metrics,
            title="Dashboard",
            save_path=paths['dashboard'],
            instrument=instrument,
            model_name=model_name,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            inverse_transform=inverse_transform
            # train_end_date y val_end_date eliminados para mayor compatibilidad
        )
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            # Enfoque alternativo si hay incompatibilidad de parámetros
            logger.warning(f"Usando llamada simplificada a create_dashboard debido a error: {str(e)}")
            create_dashboard(
                real_series=real_series,
                pred_series=pred_series,
                forecast_series=forecast_series,
                metrics=metrics,
                title="Dashboard",
                save_path=paths['dashboard'],
                instrument=instrument,
                model_name=model_name,
                ci_lower=ci_lower,
                ci_upper=ci_upper
            )
        else:
            raise
    
    return paths
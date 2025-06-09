"""
Utilidades de visualización para el pipeline ML.
Centraliza funciones de creación de gráficos para todas las etapas del proceso.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuración global para consistencia visual
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'real': '#1f77b4',  # Azul
    'predicted': '#ff7f0e',  # Naranja
    'training': '#2ca02c',  # Verde
    'validation': '#d62728',  # Rojo
    'test': '#9467bd',  # Morado
    'forecast': '#8c564b',  # Marrón
    'catboost': '#e377c2',  # Rosa
    'lightgbm': '#7f7f7f',  # Gris
    'xgboost': '#bcbd22',  # Amarillo-verde
    'mlp': '#17becf',  # Cyan
    'svm': '#1f77b4',  # Azul
    'ensemble': '#ff7f0e'  # Naranja
}

def create_directory(directory):
    """Crea un directorio si no existe."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def plot_real_vs_pred(df, date_col='date', real_col='Valor_Real', pred_col='Valor_Predicho', 
                      title=None, metrics=None, periodo=None, model_name=None, 
                      output_path=None, figsize=(12, 6)):
    """
    Genera un gráfico de líneas comparando valores reales vs. predichos.
    
    Args:
        df (DataFrame): DataFrame con columnas de fecha, valores reales y predichos
        date_col (str): Nombre de la columna de fecha
        real_col (str): Nombre de la columna con valores reales
        pred_col (str): Nombre de la columna con valores predichos
        title (str): Título del gráfico (opcional)
        metrics (dict): Diccionario con métricas para mostrar {nombre_metrica: valor}
        periodo (str): Período de los datos (Training, Validation, Test, Forecast)
        model_name (str): Nombre del modelo
        output_path (str): Ruta para guardar el gráfico (si None, solo se muestra)
        figsize (tuple): Tamaño de la figura (ancho, alto)
        
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Asegurar que la fecha está en formato datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    
    # Filtrar NaNs para no romper el gráfico
    mask_real = ~df[real_col].isna()
    mask_pred = ~df[pred_col].isna()
    
    # Plotting
    if mask_real.any():
        ax.plot(df.loc[mask_real, date_col], df.loc[mask_real, real_col], 
                label='Real', color=COLORS['real'], linewidth=2)
    
    if mask_pred.any():
        ax.plot(df.loc[mask_pred, date_col], df.loc[mask_pred, pred_col], 
                label='Predicho', color=COLORS['predicted'], linewidth=2, 
                linestyle='--', marker='o', markersize=4)
    
    # Formateo de ejes
    date_formatter = DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    
    # Título con métricas si están disponibles
    titulo = []
    if model_name:
        titulo.append(f"Modelo: {model_name}")
    if periodo:
        titulo.append(f"Período: {periodo}")
    if title:
        titulo.append(title)
    
    plt.title(" | ".join(titulo))
    
    # Añadir métricas como texto en el gráfico
    if metrics is not None:
        metrics_text = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_text.append(f"{name}: {value:.4f}")
            else:
                metrics_text.append(f"{name}: {value}")
        
        metrics_str = " | ".join(metrics_text)
        plt.figtext(0.5, 0.01, metrics_str, ha='center', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Guardar o mostrar
    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_forecast(df, date_col='date', real_col='Valor_Real', pred_col='Valor_Predicho',
                 inference_date=None, cutoff_date=None, metrics=None, 
                 title=None, model_name=None, output_path=None, figsize=(12, 6)):
    """
    Genera un gráfico de forecast con línea vertical de corte en la fecha de inferencia.
    
    Args:
        df (DataFrame): DataFrame con datos históricos y predicciones
        date_col (str): Nombre de la columna de fecha
        real_col (str): Nombre de la columna con valores reales
        pred_col (str): Nombre de la columna con valores predichos
        inference_date (str): Fecha de inferencia (ej. '2025-04-18')
        cutoff_date (str): Fecha de corte alternativa si es diferente de inference_date
        metrics (dict): Diccionario con métricas para mostrar {nombre_metrica: valor}
        title (str): Título personalizado
        model_name (str): Nombre del modelo
        output_path (str): Ruta para guardar el gráfico
        figsize (tuple): Tamaño de la figura
        
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Asegurar que la fecha está en formato datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    
    # Separar datos históricos de forecast
    if inference_date:
        inference_date = pd.to_datetime(inference_date)
        hist_data = df[df[date_col] <= inference_date]
        forecast_data = df[df[date_col] > inference_date]
    else:
        # Si no hay fecha de inferencia, asumir que el forecast es donde Valor_Real es NaN
        hist_data = df.dropna(subset=[real_col])
        forecast_data = df[df[real_col].isna()]
    
    # Plotting histórico
    if not hist_data.empty:
        ax.plot(hist_data[date_col], hist_data[real_col], 
                label='Histórico (real)', color=COLORS['real'], linewidth=2)
    
    # Plotting valores predichos históricos (validación)
    if not hist_data.empty and pred_col in hist_data.columns:
        ax.plot(hist_data[date_col], hist_data[pred_col], 
                label='Predicho (histórico)', color=COLORS['validation'], 
                linewidth=1.5, linestyle='--')
    
    # Plotting forecast
    if not forecast_data.empty:
        ax.plot(forecast_data[date_col], forecast_data[pred_col], 
                label='Forecast', color=COLORS['forecast'], linewidth=2.5, 
                linestyle='-', marker='o', markersize=5)
    
    # Línea vertical en la fecha de inferencia/corte
    cutoff = cutoff_date if cutoff_date else inference_date
    if cutoff:
        cutoff = pd.to_datetime(cutoff)
        ax.axvline(x=cutoff, color='black', linestyle='--', alpha=0.7,
                   label=f'Fecha de corte: {cutoff.strftime("%Y-%m-%d")}')
    
    # Formateo de ejes
    date_formatter = DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    
    # Título
    titulo = []
    if model_name:
        titulo.append(f"Modelo: {model_name}")
    if title:
        titulo.append(title)
    else:
        titulo.append("Inferencia y Forecast")
    
    plt.title(" | ".join(titulo))
    
    # Añadir métricas si están disponibles
    if metrics is not None:
        metrics_text = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_text.append(f"{name}: {value:.4f}")
            else:
                metrics_text.append(f"{name}: {value}")
        
        metrics_str = " | ".join(metrics_text)
        plt.figtext(0.5, 0.01, metrics_str, ha='center', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Guardar o mostrar
    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_training_curves(df, model_names=None, metric_col='RMSE', trial_col='Trial', 
                        title=None, output_path=None, figsize=(12, 6)):
    """
    Genera un gráfico comparativo de curvas de entrenamiento para diferentes modelos/trials.
    
    Args:
        df (DataFrame): DataFrame con columnas de modelo, trial y métrica
        model_names (list): Lista de nombres de modelos a incluir (si None, todos)
        metric_col (str): Nombre de la columna con la métrica a visualizar
        trial_col (str): Nombre de la columna con el número de trial
        title (str): Título personalizado
        output_path (str): Ruta para guardar el gráfico
        figsize (tuple): Tamaño de la figura
        
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filtrar por modelos si se especifica
    if model_names:
        df_plot = df[df['Modelo'].isin(model_names)].copy()
    else:
        df_plot = df.copy()
    
    # Pivot para tener modelos en columnas
    if 'Modelo' in df_plot.columns and trial_col in df_plot.columns:
        df_pivot = df_plot.pivot(index=trial_col, columns='Modelo', values=metric_col)
        df_pivot.plot(ax=ax, marker='o')
    else:
        # Alternativa si no se puede hacer pivot
        for model in df_plot['Modelo'].unique():
            model_data = df_plot[df_plot['Modelo'] == model]
            ax.plot(model_data[trial_col], model_data[metric_col], 
                    label=model, marker='o')
    
    plt.xlabel('Trial')
    plt.ylabel(metric_col)
    
    if title:
        plt.title(title)
    else:
        plt.title(f"Evolución de {metric_col} por Trial")
    
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    # Guardar o mostrar
    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_feature_importance(importances, feature_names, title=None, top_n=20, 
                           model_name=None, output_path=None, figsize=(12, 8)):
    """
    Visualiza las importancias de features para un modelo.
    
    Args:
        importances (array): Array con valores de importancia
        feature_names (list): Lista con nombres de las features
        title (str): Título personalizado
        top_n (int): Número de features a mostrar
        model_name (str): Nombre del modelo
        output_path (str): Ruta para guardar el gráfico
        figsize (tuple): Tamaño de la figura
        
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    if len(importances) != len(feature_names):
        raise ValueError("Las longitudes de importances y feature_names deben coincidir")
    
    # Ordenar por importancia y tomar las top_n
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Barplot horizontal
    ax.barh(range(len(top_indices)), importances[top_indices], align='center')
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([feature_names[i] for i in top_indices])
    
    # Título
    if title:
        plt.title(title)
    else:
        titulo = "Importancia de Features"
        if model_name:
            titulo += f" - {model_name}"
        plt.title(titulo)
    
    plt.xlabel('Importancia')
    plt.tight_layout()
    
    # Guardar o mostrar
    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_radar_metrics(metrics_dict, model_names=None, title=None, 
                      output_path=None, figsize=(10, 10)):
    """
    Genera un radar chart (araña) para comparar métricas entre modelos.
    
    Args:
        metrics_dict (dict): Diccionario de la forma {modelo: {métrica: valor, ...}, ...}
        model_names (list): Lista de nombres de modelos a incluir (si None, todos)
        title (str): Título personalizado
        output_path (str): Ruta para guardar el gráfico
        figsize (tuple): Tamaño de la figura
        
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    # Filtrar modelos si se especifica
    if model_names:
        metrics_dict = {k: v for k, v in metrics_dict.items() if k in model_names}
    
    # Obtener todas las métricas usadas
    all_metrics = set()
    for model, metrics in metrics_dict.items():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(list(all_metrics))
    
    # Calcular ángulos para el radar
    angles = np.linspace(0, 2*np.pi, len(all_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el polígono
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Para normalizar valores (todos positivos entre 0-1, donde 1 es mejor)
    normalized_metrics = {}
    
    for metric in all_metrics:
        values = [metrics_dict[model].get(metric, np.nan) for model in metrics_dict]
        values = [v for v in values if not np.isnan(v)]
        
        if not values:
            continue
            
        # Determinar si la métrica es "mejor más alta" o "mejor más baja"
        # Por convención, R2, Accuracy, Precision, Recall son "mejor más alto"
        # RMSE, MAE, MSE son "mejor más bajo"
        better_high = any(m.lower() in metric.lower() for m in ["r2", "accuracy", "precision", "recall", "score"])
        
        if better_high:
            # Normalizar: 1 es mejor (más alto), 0 es peor
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                normalized_metrics[metric] = {model: 1.0 for model in metrics_dict}
            else:
                normalized_metrics[metric] = {
                    model: (metrics_dict[model].get(metric, min_val) - min_val) / (max_val - min_val)
                    for model in metrics_dict
                }
        else:
            # Normalizar: 0 es mejor (más bajo), 1 es peor
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                normalized_metrics[metric] = {model: 1.0 for model in metrics_dict}
            else:
                normalized_metrics[metric] = {
                    model: 1 - (metrics_dict[model].get(metric, max_val) - min_val) / (max_val - min_val)
                    for model in metrics_dict
                }
    
    # Dibujar cada modelo
    for i, (model, metrics) in enumerate(metrics_dict.items()):
        values = [normalized_metrics.get(metric, {}).get(model, 0) for metric in all_metrics]
        values += values[:1]  # Cerrar el polígono
        
        color = COLORS.get(model.lower(), f"C{i}")
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Etiquetar ejes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_metrics)
    
    # Título
    if title:
        plt.title(title)
    else:
        plt.title("Comparación de Métricas")
    
    # Leyenda y formato
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    
    # Guardar o mostrar
    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_pipeline_timeline(times_dict, output_path=None, figsize=(14, 8)):
    """
    Genera un gráfico de Gantt con los tiempos de ejecución de cada etapa del pipeline.
    
    Args:
        times_dict (dict): Diccionario con {etapa: tiempo_segundos, ...}
        output_path (str): Ruta para guardar el gráfico
        figsize (tuple): Tamaño de la figura
        
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    # Ordenar por etapa (step_1, step_2, etc.)
    stages = sorted(times_dict.keys())
    times = [times_dict[stage] for stage in stages]
    
    # Calcular tiempos acumulados para posicionamiento
    cumulative_times = np.zeros(len(times))
    for i in range(1, len(times)):
        cumulative_times[i] = cumulative_times[i-1] + times[i-1]
    
    # Formatear etiquetas de tiempo
    time_labels = []
    for t in times:
        if t < 60:
            time_labels.append(f"{t:.1f}s")
        elif t < 3600:
            time_labels.append(f"{t/60:.1f}m")
        else:
            time_labels.append(f"{t/3600:.1f}h")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Crear barras horizontales
    bars = ax.barh(stages, times, left=cumulative_times, height=0.5)
    
    # Añadir etiquetas de tiempo dentro de las barras
    for i, (bar, label) in enumerate(zip(bars, time_labels)):
        width = bar.get_width()
        if width > 0:
            x = bar.get_x() + width/2
            y = bar.get_y() + bar.get_height()/2
            ax.text(x, y, label, ha='center', va='center', color='white', fontweight='bold')
    
    # Calcular tiempo total
    total_time = sum(times)
    total_label = f"{total_time:.1f}s"
    if total_time >= 60:
        minutes = total_time / 60
        total_label = f"{minutes:.1f}m"
        if minutes >= 60:
            total_label = f"{minutes/60:.1f}h {int(minutes%60)}m"
    
    plt.xlabel(f"Tiempo (Total: {total_label})")
    plt.title("Tiempos de ejecución del pipeline")
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Guardar o mostrar
    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_correlation_matrix(df, method='pearson', title=None, 
                           target_col=None, threshold=0.7, figsize=(14, 12),
                           output_path=None):
    """
    Genera una matriz de correlación con heatmap.
    
    Args:
        df (DataFrame): DataFrame con variables
        method (str): Método de correlación ('pearson', 'spearman' o 'kendall')
        title (str): Título personalizado
        target_col (str): Columna objetivo para destacar
        threshold (float): Umbral para destacar correlaciones altas
        figsize (tuple): Tamaño de la figura
        output_path (str): Ruta para guardar el gráfico
        
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    # Calcular matriz de correlación
    corr_matrix = df.corr(method=method)
    
    # Crear máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generar heatmap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    # Destacar correlaciones altas con el target
    if target_col and target_col in corr_matrix.columns:
        target_corrs = corr_matrix[target_col].abs().sort_values(ascending=False)
        high_corrs = target_corrs[target_corrs > threshold].index.tolist()
        
        # Añadir texto con correlaciones altas
        high_corrs_text = "\n".join([
            f"{col}: {corr_matrix.loc[col, target_col]:.3f}" 
            for col in high_corrs if col != target_col
        ])
        
        if high_corrs_text:
            plt.figtext(0.01, 0.01, f"Correlaciones altas con {target_col}:\n{high_corrs_text}", 
                        fontsize=10, ha='left',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Título
    if title:
        plt.title(title)
    else:
        plt.title(f"Matriz de Correlación ({method})")
    
    plt.tight_layout()
    
    # Guardar o mostrar
    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def plot_metrics_by_subperiod(metrics_by_period, metric_col='RMSE', model_col='Modelo', 
                             period_col='Periodo', title=None, output_path=None, figsize=(14, 8)):
    """
    Genera un gráfico comparando métricas por subperíodos.
    
    Args:
        metrics_by_period (DataFrame): DataFrame con métricas por período
        metric_col (str): Columna con la métrica a visualizar
        model_col (str): Columna con el nombre del modelo
        period_col (str): Columna con el período/subperíodo
        title (str): Título personalizado
        output_path (str): Ruta para guardar el gráfico
        figsize (tuple): Tamaño de la figura
        
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Pivot para tener modelos en columnas y períodos en filas
    if model_col in metrics_by_period.columns and period_col in metrics_by_period.columns:
        df_pivot = metrics_by_period.pivot(index=period_col, columns=model_col, values=metric_col)
        df_pivot.plot(kind='bar', ax=ax)
    else:
        # Alternativa si no se puede hacer pivot
        sns.barplot(x=period_col, y=metric_col, hue=model_col, data=metrics_by_period, ax=ax)
    
    plt.xlabel('Período')
    plt.ylabel(metric_col)
    
    if title:
        plt.title(title)
    else:
        plt.title(f"{metric_col} por Período")
    
    plt.legend(title='Modelo')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Guardar o mostrar
    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def generate_report_figures(results_df, metrics_df, output_dir, model_names=None, 
                           prefix="", include_subplots=True):
    """
    Genera un conjunto completo de figuras para un reporte.
    
    Args:
        results_df (DataFrame): DataFrame con resultados (real vs predicción)
        metrics_df (DataFrame): DataFrame con métricas
        output_dir (str): Directorio donde guardar las figuras
        model_names (list): Lista de modelos a incluir (None para todos)
        prefix (str): Prefijo para nombres de archivo
        include_subplots (bool): Si debe generar subplots para cada modelo

    Returns:
        list: Lista de rutas a las figuras generadas
    """
    create_directory(output_dir)
    generated_figures = []
    
    # Filtrar modelos si es necesario
    if model_names:
        results_filtered = results_df[results_df['Modelo'].isin(model_names)]
        metrics_filtered = metrics_df[metrics_df['Modelo'].isin(model_names)]
    else:
        results_filtered = results_df
        metrics_filtered = metrics_df
        model_names = results_df['Modelo'].unique()
    
    # 1. Figura de comparación general (todos los modelos)
    if len(model_names) > 1:
        all_models_fig_path = os.path.join(output_dir, f"{prefix}all_models_comparison.png")
        plot_real_vs_pred(
            results_filtered, 
            title="Comparación de todos los modelos",
            output_path=all_models_fig_path
        )
        generated_figures.append(all_models_fig_path)
    
    # 2. Figura para cada modelo individual
    if include_subplots:
        for model in model_names:
            model_results = results_filtered[results_filtered['Modelo'] == model]
            if model_results.empty:
                continue
                
            # Obtener métricas para este modelo
            model_metrics = {}
            if not metrics_filtered.empty:
                model_metrics_row = metrics_filtered[metrics_filtered['Modelo'] == model]
                if not model_metrics_row.empty:
                    model_metrics = model_metrics_row.iloc[0].to_dict()
            
            model_fig_path = os.path.join(output_dir, f"{prefix}{model.replace(' ', '_')}_comparison.png")
            plot_real_vs_pred(
                model_results,
                title=f"Modelo: {model}",
                metrics=model_metrics,
                model_name=model,
                output_path=model_fig_path
            )
            generated_figures.append(model_fig_path)
    
    # 3. Radar de métricas
    if len(model_names) > 1 and not metrics_filtered.empty:
        # Convertir a formato de diccionario para radar chart
        metrics_dict = {}
        for _, row in metrics_filtered.iterrows():
            model = row['Modelo']
            metrics = {col: row[col] for col in metrics_filtered.columns 
                      if col != 'Modelo' and not pd.isna(row[col])}
            metrics_dict[model] = metrics
        
        radar_path = os.path.join(output_dir, f"{prefix}radar_comparison.png")
        plot_radar_metrics(
            metrics_dict,
            title="Comparación de métricas",
            output_path=radar_path
        )
        generated_figures.append(radar_path)
    
    # 4. Barplot de métricas principales (RMSE, MAE)
    metrics_cols = [col for col in metrics_filtered.columns 
                   if col not in ['Modelo', 'Tipo_de_Mercado']]
    for metric in ['RMSE', 'MAE', 'SMAPE', 'R2']:
        if metric in metrics_cols:
            metric_path = os.path.join(output_dir, f"{prefix}{metric.lower()}_comparison.png")
            plt.figure(figsize=(10, 6))
            metrics_plot = metrics_filtered.sort_values(metric)
            plt.bar(metrics_plot['Modelo'], metrics_plot[metric])
            plt.title(f"Comparación de {metric} entre modelos")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(metric_path, dpi=300)
            plt.close()
            generated_figures.append(metric_path)
    
    return generated_figures
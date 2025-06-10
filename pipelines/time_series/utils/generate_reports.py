#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generación de Reportes para Modelos de Series Temporales
=======================================================
Proporciona funciones para generar reportes estandarizados para
los modelos ARIMA/SARIMAX, equivalentes a los del pipeline ML.

Incluye:
- Informes de rendimiento individual de modelos
- Comparativas entre modelos
- Reportes de validación y backtesting
- Reportes de pronóstico
- Informes ejecutivos
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from jinja2 import Template
import base64
from io import BytesIO

# Importamos nuestras funciones de visualización
try:
    from . import plots
except ImportError:
    # Para permitir importación independiente durante desarrollo
    import plots

def _fig_to_base64(fig):
    """Convierte una figura de matplotlib a una cadena base64 para HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def generate_model_summary(
    model_name: str,
    instrument: str,
    metrics: Dict[str, float],
    model_params: Dict[str, Any],
    training_period: Tuple[str, str],
    output_dir: Path,
    include_plots: bool = True,
    plots_paths: Optional[Dict[str, Path]] = None
) -> Path:
    """
    Genera un resumen en formato JSON para un modelo específico.
    
    Args:
        model_name: Nombre del modelo
        instrument: Instrumento financiero
        metrics: Diccionario con métricas
        model_params: Parámetros del modelo
        training_period: Período de entrenamiento (inicio, fin)
        output_dir: Directorio de salida
        include_plots: Si se deben incluir rutas a plots
        plots_paths: Diccionario con rutas a plots (si include_plots=True)
    
    Returns:
        Ruta al archivo JSON generado
    """
    # Asegurar que existe el directorio
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Construir resumen
    summary = {
        "model": model_name,
        "instrument": instrument,
        "date_generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "training_period": {
            "start": training_period[0],
            "end": training_period[1]
        },
        "metrics": metrics,
        "model_parameters": model_params
    }
    
    # Añadir rutas a plots si corresponde
    if include_plots and plots_paths is not None:
        summary["plots"] = {k: str(v) for k, v in plots_paths.items()}
    
    # Guardar como JSON
    output_file = output_dir / f"{instrument}_{model_name}_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    return output_file

def generate_comparison_report(
    models_data: Dict[str, Dict[str, Any]],
    instrument: str,
    output_dir: Path,
    include_plots: bool = True,
    comparison_plots: Optional[Dict[str, Path]] = None
) -> Path:
    """
    Genera un informe comparativo entre modelos en formato Excel.
    
    Args:
        models_data: Diccionario con datos de cada modelo
        instrument: Instrumento financiero
        output_dir: Directorio de salida
        include_plots: Si se debe incluir referencia a gráficos
        comparison_plots: Diccionario con rutas a gráficos comparativos
    
    Returns:
        Ruta al archivo Excel generado
    """
    # Asegurar que existe el directorio
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Preparar datos para informe
    comparison_data = []
    
    for model_name, model_info in models_data.items():
        row = {
            'Modelo': model_name,
            'Instrumento': instrument
        }
        
        # Añadir métricas
        if 'metrics' in model_info:
            for metric_name, metric_value in model_info['metrics'].items():
                row[metric_name] = metric_value
        
        # Añadir parámetros del modelo
        if 'params' in model_info:
            params_str = json.dumps(model_info['params'])
            row['Parámetros'] = params_str
        
        comparison_data.append(row)
    
    # Crear DataFrame
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # Definir columnas a mostrar
        all_columns = list(df_comparison.columns)
        key_metrics = ['RMSE', 'MAE', 'Hit_Direction', 'R2']
        available_metrics = [m for m in key_metrics if m in all_columns]
        
        # Ordenar columnas: primero info básica, luego métricas clave, luego el resto
        priority_cols = ['Modelo', 'Instrumento'] + available_metrics
        other_cols = [c for c in all_columns if c not in priority_cols]
        ordered_cols = priority_cols + other_cols
        
        df_comparison = df_comparison[ordered_cols]
        
        # Ordenar por RMSE si está disponible
        if 'RMSE' in df_comparison.columns:
            df_comparison = df_comparison.sort_values('RMSE')
        
        # Guardar como Excel
        output_file = output_dir / f"{instrument}_model_comparison.xlsx"
        
        try:
            # Intentar usar xlsxwriter para mejor formato
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                df_comparison.to_excel(writer, sheet_name='Comparación', index=False)
                
                # Dar formato
                workbook = writer.book
                worksheet = writer.sheets['Comparación']
                
                # Formato para números
                num_format = workbook.add_format({'num_format': '0.0000'})
                
                # Aplicar formatos
                for i, col in enumerate(df_comparison.columns):
                    column_width = max(len(str(col)) + 2, 12)
                    worksheet.set_column(i, i, column_width)
                    
                    # Aplicar formato numérico a métricas
                    if col in ['RMSE', 'MAE', 'MSE', 'MAPE', 'R2', 'Hit_Direction']:
                        for row in range(1, len(df_comparison) + 1):
                            worksheet.write_number(row, i, df_comparison[col].iloc[row-1], num_format)
                
                # Si hay gráficos comparativos, añadir hoja con referencias
                if include_plots and comparison_plots:
                    plots_data = []
                    for plot_name, plot_path in comparison_plots.items():
                        plots_data.append({
                            'Nombre': plot_name,
                            'Ruta': str(plot_path)
                        })
                    
                    if plots_data:
                        pd.DataFrame(plots_data).to_excel(writer, sheet_name='Gráficos', index=False)
        
        except ImportError:
            # Fallback si no está xlsxwriter
            df_comparison.to_excel(output_file, index=False)
        
        return output_file
    
    return None

def generate_forecast_report(
    forecast_data: Dict[str, pd.DataFrame],
    instrument: str,
    output_dir: Path,
    include_plots: bool = True,
    forecast_plots: Optional[Dict[str, Path]] = None
) -> Path:
    """
    Genera un informe de pronóstico en formato Excel.
    
    Args:
        forecast_data: Diccionario con DataFrames de pronóstico por modelo
        instrument: Instrumento financiero
        output_dir: Directorio de salida
        include_plots: Si se debe incluir referencia a gráficos
        forecast_plots: Diccionario con rutas a gráficos de pronóstico
    
    Returns:
        Ruta al archivo Excel generado
    """
    # Asegurar que existe el directorio
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Combinar pronósticos en un solo DataFrame
    combined_data = []
    
    for model_name, df in forecast_data.items():
        # Verificar si el DataFrame tiene la estructura esperada
        required_cols = ['forecast']
        if not all(col in df.columns for col in required_cols):
            continue
        
        # Extraer datos
        for idx, row in df.iterrows():
            forecast_date = idx if isinstance(idx, pd.Timestamp) else pd.to_datetime(row.get('date', idx))
            
            row_data = {
                'Fecha': forecast_date,
                'Modelo': model_name,
                'Pronóstico': row['forecast']
            }
            
            # Añadir intervalos de confianza si están disponibles
            if 'lower_95' in row and 'upper_95' in row:
                row_data['Límite Inferior (95%)'] = row['lower_95']
                row_data['Límite Superior (95%)'] = row['upper_95']
            
            combined_data.append(row_data)
    
    if not combined_data:
        return None
    
    # Crear DataFrame
    df_combined = pd.DataFrame(combined_data)
    
    # Convertir fecha a datetime si no lo es
    if 'Fecha' in df_combined.columns and not pd.api.types.is_datetime64_dtype(df_combined['Fecha']):
        df_combined['Fecha'] = pd.to_datetime(df_combined['Fecha'], errors='coerce')
    
    # Ordenar por fecha y modelo
    if 'Fecha' in df_combined.columns:
        df_combined = df_combined.sort_values(['Fecha', 'Modelo'])
    
    # Guardar como Excel
    output_file = output_dir / f"{instrument}_forecast_report.xlsx"
    
    try:
        # Intentar usar xlsxwriter para mejor formato
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # Guardar pronósticos combinados
            df_combined.to_excel(writer, sheet_name='Pronósticos', index=False)
            
            # Dar formato
            workbook = writer.book
            worksheet = writer.sheets['Pronósticos']
            
            # Formatos
            date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
            num_format = workbook.add_format({'num_format': '0.0000'})
            
            # Aplicar formatos
            for i, col in enumerate(df_combined.columns):
                column_width = max(len(str(col)) + 2, 14)
                worksheet.set_column(i, i, column_width)
                
                if col == 'Fecha':
                    worksheet.set_column(i, i, column_width, date_format)
                elif col in ['Pronóstico', 'Límite Inferior (95%)', 'Límite Superior (95%)']:
                    worksheet.set_column(i, i, column_width, num_format)
            
            # Crear una hoja por modelo con pivote
            for model_name in forecast_data.keys():
                model_data = df_combined[df_combined['Modelo'] == model_name].copy()
                
                if not model_data.empty:
                    # Seleccionar columnas relevantes
                    cols_to_keep = ['Fecha', 'Pronóstico']
                    if 'Límite Inferior (95%)' in model_data.columns:
                        cols_to_keep.append('Límite Inferior (95%)')
                    if 'Límite Superior (95%)' in model_data.columns:
                        cols_to_keep.append('Límite Superior (95%)')
                    
                    model_sheet = model_data[cols_to_keep].copy()
                    
                    # Guardar en hoja separada
                    sheet_name = model_name[:31]  # Máximo 31 caracteres para nombre de hoja
                    model_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Dar formato
                    worksheet = writer.sheets[sheet_name]
                    
                    # Aplicar formatos
                    for i, col in enumerate(model_sheet.columns):
                        column_width = max(len(str(col)) + 2, 14)
                        worksheet.set_column(i, i, column_width)
                        
                        if col == 'Fecha':
                            worksheet.set_column(i, i, column_width, date_format)
                        elif col in ['Pronóstico', 'Límite Inferior (95%)', 'Límite Superior (95%)']:
                            worksheet.set_column(i, i, column_width, num_format)
            
            # Si hay gráficos de pronóstico, añadir hoja con referencias
            if include_plots and forecast_plots:
                plots_data = []
                for plot_name, plot_path in forecast_plots.items():
                    plots_data.append({
                        'Nombre': plot_name,
                        'Ruta': str(plot_path)
                    })
                
                if plots_data:
                    pd.DataFrame(plots_data).to_excel(writer, sheet_name='Gráficos', index=False)
    
    except ImportError:
        # Fallback si no está xlsxwriter
        df_combined.to_excel(output_file, index=False)
    
    return output_file

def generate_backtest_report(
    backtest_data: Dict[str, Dict[str, Dict[str, Any]]],
    instrument: str,
    output_dir: Path,
    include_plots: bool = True,
    backtest_plots: Optional[Dict[str, Path]] = None
) -> Path:
    """
    Genera un informe detallado de backtesting en formato Excel.
    
    Args:
        backtest_data: Datos de backtesting por modelo
        instrument: Instrumento financiero
        output_dir: Directorio de salida
        include_plots: Si se debe incluir referencia a gráficos
        backtest_plots: Diccionario con rutas a gráficos de backtesting
    
    Returns:
        Ruta al archivo Excel generado
    """
    # Asegurar que existe el directorio
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Filtrar solo datos del instrumento especificado
    if instrument in backtest_data:
        instrument_data = backtest_data[instrument]
    else:
        return None
    
    # Preparar datos para informe
    summary_data = []
    
    for model_name, model_results in instrument_data.items():
        if 'metrics' not in model_results:
            continue
        
        row = {
            'Modelo': model_name,
            'Instrumento': instrument,
            'Observaciones': model_results.get('num_observations', 0)
        }
        
        # Añadir métricas
        for metric_name, metric_value in model_results['metrics'].items():
            row[metric_name] = metric_value
        
        # Añadir período si está disponible
        if 'period' in model_results:
            if 'start' in model_results['period'] and model_results['period']['start'] is not None:
                row['Fecha Inicio'] = model_results['period']['start']
            if 'end' in model_results['period'] and model_results['period']['end'] is not None:
                row['Fecha Fin'] = model_results['period']['end']
        
        summary_data.append(row)
    
    if not summary_data:
        return None
    
    # Crear DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Ordenar por RMSE si está disponible
    if 'RMSE' in df_summary.columns:
        df_summary = df_summary.sort_values('RMSE')
    
    # Guardar como Excel
    output_file = output_dir / f"{instrument}_backtest_report.xlsx"
    
    try:
        # Intentar usar xlsxwriter para mejor formato
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # Guardar resumen
            df_summary.to_excel(writer, sheet_name='Resumen', index=False)
            
            # Dar formato
            workbook = writer.book
            worksheet = writer.sheets['Resumen']
            
            # Formatos
            num_format = workbook.add_format({'num_format': '0.0000'})
            date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
            
            # Aplicar formatos
            for i, col in enumerate(df_summary.columns):
                column_width = max(len(str(col)) + 2, 14)
                worksheet.set_column(i, i, column_width)
                
                if col in ['Fecha Inicio', 'Fecha Fin']:
                    worksheet.set_column(i, i, column_width, date_format)
                elif col not in ['Modelo', 'Instrumento', 'Observaciones']:
                    worksheet.set_column(i, i, column_width, num_format)
            
            # Si hay datos por subperíodos, añadir hojas adicionales
            for model_name, model_results in instrument_data.items():
                if 'subperiods' in model_results and model_results['subperiods']:
                    subperiod_data = []
                    
                    for subperiod in model_results['subperiods']:
                        subperiod_row = {
                            'Período': subperiod.get('period', ''),
                            'Fecha Inicio': subperiod.get('start', ''),
                            'Fecha Fin': subperiod.get('end', ''),
                            'Observaciones': subperiod.get('num_observations', 0)
                        }
                        
                        # Añadir métricas del subperíodo
                        if 'metrics' in subperiod:
                            for metric_name, metric_value in subperiod['metrics'].items():
                                subperiod_row[metric_name] = metric_value
                        
                        subperiod_data.append(subperiod_row)
                    
                    if subperiod_data:
                        df_subperiods = pd.DataFrame(subperiod_data)
                        
                        # Ordenar por fecha de inicio si está disponible
                        if 'Fecha Inicio' in df_subperiods.columns:
                            df_subperiods['Fecha Inicio'] = pd.to_datetime(df_subperiods['Fecha Inicio'], errors='coerce')
                            df_subperiods = df_subperiods.sort_values('Fecha Inicio')
                        
                        # Guardar en hoja propia para este modelo
                        sheet_name = f"{model_name[:27]}_subp"  # Limitar longitud
                        df_subperiods.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Dar formato
                        worksheet = writer.sheets[sheet_name]
                        
                        # Aplicar formatos
                        for i, col in enumerate(df_subperiods.columns):
                            column_width = max(len(str(col)) + 2, 14)
                            worksheet.set_column(i, i, column_width)
                            
                            if col in ['Fecha Inicio', 'Fecha Fin']:
                                worksheet.set_column(i, i, column_width, date_format)
                            elif col not in ['Período', 'Observaciones']:
                                worksheet.set_column(i, i, column_width, num_format)
            
            # Si hay gráficos de backtesting, añadir hoja con referencias
            if include_plots and backtest_plots:
                plots_data = []
                for plot_name, plot_path in backtest_plots.items():
                    plots_data.append({
                        'Nombre': plot_name,
                        'Ruta': str(plot_path)
                    })
                
                if plots_data:
                    pd.DataFrame(plots_data).to_excel(writer, sheet_name='Gráficos', index=False)
    
    except ImportError:
        # Fallback si no está xlsxwriter
        df_summary.to_excel(output_file, index=False)
    
    return output_file

def generate_pdf_report(
    instrument: str,
    models_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    include_forecast: bool = True,
    forecast_data: Optional[Dict[str, pd.DataFrame]] = None,
    backtest_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
) -> Path:
    """
    Genera un informe PDF completo para un instrumento.
    
    Args:
        instrument: Instrumento financiero
        models_data: Datos de los modelos
        output_dir: Directorio de salida
        include_forecast: Si se debe incluir información de pronóstico
        forecast_data: Datos de pronóstico por modelo
        backtest_data: Datos de backtesting
    
    Returns:
        Ruta al archivo PDF generado
    """
    # Asegurar que existe el directorio
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Nombre del archivo de salida
    output_file = output_dir / f"{instrument}_complete_report.pdf"
    
    # Crear un archivo PDF multipágina
    with PdfPages(output_file) as pdf:
        # 1. Página de título
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.6, f"Informe Completo - {instrument}", 
                fontsize=24, ha='center')
        plt.text(0.5, 0.5, f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                fontsize=14, ha='center')
        plt.text(0.5, 0.4, "Pipeline de Series Temporales", 
                fontsize=14, ha='center')
        pdf.savefig()
        plt.close()
        
        # 2. Resumen de modelos
        if models_data:
            # Crear tabla de métricas
            summary_data = []
            
            for model_name, model_info in models_data.items():
                row = {'Modelo': model_name}
                
                # Añadir métricas principales
                if 'metrics' in model_info:
                    metrics = model_info['metrics']
                    for metric in ['RMSE', 'MAE', 'Hit_Direction', 'R2']:
                        if metric in metrics:
                            row[metric] = metrics[metric]
                
                summary_data.append(row)
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                
                # Ordenar por RMSE si está disponible
                if 'RMSE' in df_summary.columns:
                    df_summary = df_summary.sort_values('RMSE')
                
                # Graficar tabla
                plt.figure(figsize=(11, 8.5))
                plt.axis('off')
                plt.title(f"Resumen de Modelos - {instrument}", fontsize=16, pad=20)
                
                # Añadir tabla
                rows = len(df_summary) + 1  # +1 para encabezados
                cols = len(df_summary.columns)
                
                table = plt.table(
                    cellText=np.vstack([df_summary.columns, df_summary.values]),
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.15] * cols
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                
                # Dar formato a encabezados
                for i in range(cols):
                    table[(0, i)].set_facecolor('#4472C4')
                    table[(0, i)].set_text_props(color='white')
                
                pdf.savefig()
                plt.close()
        
        # 3. Gráficos de comparación de métricas
        if models_data:
            # Preparar datos para gráficos
            model_names = list(models_data.keys())
            metrics_data = {}
            
            for metric in ['RMSE', 'MAE', 'Hit_Direction', 'R2']:
                metrics_data[metric] = []
                for model_name in model_names:
                    if 'metrics' in models_data[model_name] and metric in models_data[model_name]['metrics']:
                        metrics_data[metric].append(models_data[model_name]['metrics'][metric])
                    else:
                        metrics_data[metric].append(np.nan)
            
            # Graficar métricas clave
            for metric in ['RMSE', 'Hit_Direction']:
                if all(np.isnan(metrics_data[metric])):
                    continue
                    
                plt.figure(figsize=(11, 8.5))
                plt.bar(model_names, metrics_data[metric])
                plt.title(f"{metric} por Modelo - {instrument}", fontsize=16)
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        
        # 4. Gráficos de pronóstico
        if include_forecast and forecast_data:
            for model_name, forecast_df in forecast_data.items():
                if 'forecast' not in forecast_df.columns:
                    continue
                
                plt.figure(figsize=(11, 8.5))
                plt.plot(forecast_df.index, forecast_df['forecast'], label='Pronóstico')
                
                if 'lower_95' in forecast_df.columns and 'upper_95' in forecast_df.columns:
                    plt.fill_between(
                        forecast_df.index, 
                        forecast_df['lower_95'], 
                        forecast_df['upper_95'], 
                        alpha=0.3, 
                        label='95% CI'
                    )
                
                plt.title(f"Pronóstico - {instrument} - {model_name}", fontsize=16)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        
        # 5. Resultados de backtesting
        if backtest_data and instrument in backtest_data:
            instrument_backtest = backtest_data[instrument]
            
            for model_name, model_results in instrument_backtest.items():
                if 'subperiods' not in model_results or not model_results['subperiods']:
                    continue
                
                # Extraer datos de subperíodos
                periods = []
                rmse_values = []
                hit_dir_values = []
                
                for subperiod in model_results['subperiods']:
                    if 'period' in subperiod and 'metrics' in subperiod:
                        periods.append(subperiod['period'])
                        
                        if 'RMSE' in subperiod['metrics']:
                            rmse_values.append(subperiod['metrics']['RMSE'])
                        else:
                            rmse_values.append(np.nan)
                            
                        if 'Hit_Direction' in subperiod['metrics']:
                            hit_dir_values.append(subperiod['metrics']['Hit_Direction'])
                        else:
                            hit_dir_values.append(np.nan)
                
                if not periods:
                    continue
                
                # Graficar evolución de métricas
                plt.figure(figsize=(11, 8.5))
                
                # Dos subplots para RMSE y Hit Direction
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
                
                # RMSE
                ax1.plot(periods, rmse_values, marker='o', linestyle='-', linewidth=2)
                ax1.set_title(f"RMSE por Subperíodo - {instrument} - {model_name}", fontsize=14)
                ax1.set_ylabel('RMSE')
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # Hit Direction
                ax2.plot(periods, hit_dir_values, marker='o', linestyle='-', linewidth=2, color='green')
                ax2.set_title(f"Hit Direction por Subperíodo - {instrument} - {model_name}", fontsize=14)
                ax2.set_ylabel('Hit Direction (%)')
                ax2.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
    
    return output_file

def generate_html_report(
    instrument: str,
    models_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    include_forecast: bool = True,
    forecast_data: Optional[Dict[str, pd.DataFrame]] = None,
    backtest_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
) -> Path:
    """
    Genera un informe HTML completo para un instrumento.
    
    Args:
        instrument: Instrumento financiero
        models_data: Datos de los modelos
        output_dir: Directorio de salida
        include_forecast: Si se debe incluir información de pronóstico
        forecast_data: Datos de pronóstico por modelo
        backtest_data: Datos de backtesting
    
    Returns:
        Ruta al archivo HTML generado
    """
    # Asegurar que existe el directorio
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Nombre del archivo de salida
    output_file = output_dir / f"{instrument}_report.html"
    
    # Crear gráficos para incluir en el informe
    charts = {}
    
    # 1. Comparación de métricas
    if models_data:
        try:
            metrics_by_model = {}
            for model_name, model_info in models_data.items():
                if 'metrics' in model_info:
                    metrics_by_model[model_name] = model_info['metrics']
            
            # Gráfico de RMSE
            fig = plots.plot_metrics_comparison(
                metrics=metrics_by_model, 
                metric_name='RMSE', 
                instrument=instrument
            )
            if fig:
                charts['rmse_comparison'] = _fig_to_base64(fig)
                plt.close(fig)
            
            # Gráfico de Hit Direction
            fig = plots.plot_metrics_comparison(
                metrics=metrics_by_model, 
                metric_name='Hit_Direction', 
                instrument=instrument,
                is_lower_better=False
            )
            if fig:
                charts['hit_direction_comparison'] = _fig_to_base64(fig)
                plt.close(fig)
            
            # Gráfico de radar
            fig = plots.plot_metrics_radar(
                metrics=metrics_by_model,
                metric_names=['RMSE', 'MAE', 'Hit_Direction', 'R2', 'Amplitude_Score', 'Phase_Score'],
                instrument=instrument
            )
            if fig:
                charts['radar_metrics'] = _fig_to_base64(fig)
                plt.close(fig)
        except Exception as e:
    
    # 2. Pronósticos
    if include_forecast and forecast_data:
        try:
            # Ejemplo del primer modelo para ilustrar
            if forecast_data and next(iter(forecast_data.values()), None) is not None:
                model_name = next(iter(forecast_data.keys()))
                forecast_df = forecast_data[model_name]
                
                if 'forecast' in forecast_df.columns:
                    # Crear serie histórica simulada para contexto (últimos N puntos)
                    if 'historical' in models_data.get(model_name, {}):
                        historical = models_data[model_name]['historical']
                    else:
                        # Serie simulada para la visualización
                        idx = pd.date_range(end=forecast_df.index[0] - pd.Timedelta(days=1), periods=30, freq='B')
                        historical = pd.Series(np.random.randn(30).cumsum(), index=idx)
                    
                    # Gráfico de pronóstico
                    ci_lower = forecast_df.get('lower_95', None)
                    ci_upper = forecast_df.get('upper_95', None)
                    
                    fig = plots.plot_forecast(
                        historical=historical,
                        forecast=forecast_df['forecast'],
                        ci_lower=ci_lower,
                        ci_upper=ci_upper,
                        instrument=instrument,
                        model_name=model_name
                    )
                    
                    if fig:
                        charts['forecast_example'] = _fig_to_base64(fig)
                        plt.close(fig)
        except Exception as e:
    
    # 3. Backtesting
    if backtest_data and instrument in backtest_data:
        try:
            instrument_backtest = backtest_data[instrument]
            
            # Tomar el primer modelo con subperíodos como ejemplo
            for model_name, model_results in instrument_backtest.items():
                if 'subperiods' in model_results and model_results['subperiods']:
                    # Extraer datos de subperíodos
                    periods = []
                    rmse_values = []
                    hit_dir_values = []
                    
                    for subperiod in model_results['subperiods']:
                        if 'period' in subperiod and 'metrics' in subperiod:
                            periods.append(subperiod['period'])
                            
                            if 'RMSE' in subperiod['metrics']:
                                rmse_values.append(subperiod['metrics']['RMSE'])
                            else:
                                rmse_values.append(np.nan)
                                
                            if 'Hit_Direction' in subperiod['metrics']:
                                hit_dir_values.append(subperiod['metrics']['Hit_Direction'])
                            else:
                                hit_dir_values.append(np.nan)
                    
                    if periods:
                        # Convertir a DataFrame para plot_metric_evolution
                        df_subperiods = pd.DataFrame({
                            'period': periods,
                            'RMSE': rmse_values,
                            'Hit_Direction': hit_dir_values
                        })
                        df_subperiods['date'] = pd.to_datetime([p.split(' ')[0] for p in periods], errors='coerce')
                        df_subperiods = df_subperiods.set_index('date')
                        
                        # Gráfico de evolución de RMSE
                        fig = plots.plot_metric_evolution(
                            metric_values=df_subperiods,
                            metric_name='RMSE',
                            instrument=instrument,
                            model_name=model_name
                        )
                        
                        if fig:
                            charts['rmse_evolution'] = _fig_to_base64(fig)
                            plt.close(fig)
                        
                        # Gráfico de evolución de Hit Direction
                        fig = plots.plot_metric_evolution(
                            metric_values=df_subperiods,
                            metric_name='Hit_Direction',
                            instrument=instrument,
                            model_name=model_name
                        )
                        
                        if fig:
                            charts['hit_dir_evolution'] = _fig_to_base64(fig)
                            plt.close(fig)
                        
                        break  # Solo usar el primer modelo con datos
        except Exception as e:
    
    # Plantilla HTML
    html_template = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Informe {{instrument}} - Series Temporales</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #4472C4;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            .chart-container {
                margin: 20px 0;
                text-align: center;
            }
            .chart {
                max-width: 100%;
                height: auto;
            }
            .section {
                margin-bottom: 40px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }
            .header {
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
            }
            .metric-value {
                font-weight: bold;
                color: #2980b9;
            }
            .forecast-section {
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 5px;
                margin-top: 30px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Informe de Series Temporales - {{instrument}}</h1>
            <p>Generado: {{date_generated}}</p>
        </div>
        
        <div class="section">
            <h2>Resumen de Modelos</h2>
            <table>
                <thead>
                    <tr>
                        <th>Modelo</th>
                        {% for metric in metrics_columns %}
                        <th>{{metric}}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for model in models_summary %}
                    <tr>
                        <td>{{model.name}}</td>
                        {% for metric in metrics_columns %}
                        <td>
                            {% if model.metrics[metric] is defined %}
                            <span class="metric-value">{{model.metrics[metric]|round(4)}}</span>
                            {% else %}
                            -
                            {% endif %}
                        </td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            {% if 'rmse_comparison' in charts %}
            <div class="chart-container">
                <h3>Comparación de RMSE</h3>
                <img class="chart" src="data:image/png;base64,{{charts.rmse_comparison}}" alt="Comparación de RMSE">
            </div>
            {% endif %}
            
            {% if 'hit_direction_comparison' in charts %}
            <div class="chart-container">
                <h3>Comparación de Hit Direction</h3>
                <img class="chart" src="data:image/png;base64,{{charts.hit_direction_comparison}}" alt="Comparación de Hit Direction">
            </div>
            {% endif %}
            
            {% if 'radar_metrics' in charts %}
            <div class="chart-container">
                <h3>Comparación de Métricas Múltiples</h3>
                <img class="chart" src="data:image/png;base64,{{charts.radar_metrics}}" alt="Radar de Métricas">
            </div>
            {% endif %}
        </div>
        
        {% if include_forecast %}
        <div class="section forecast-section">
            <h2>Pronósticos</h2>
            
            {% if 'forecast_example' in charts %}
            <div class="chart-container">
                <h3>Ejemplo de Pronóstico</h3>
                <img class="chart" src="data:image/png;base64,{{charts.forecast_example}}" alt="Ejemplo de Pronóstico">
            </div>
            {% endif %}
            
            {% if forecast_summary %}
            <h3>Resumen de Pronósticos</h3>
            <table>
                <thead>
                    <tr>
                        <th>Fecha</th>
                        {% for model in forecast_summary.models %}
                        <th>{{model}}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for date in forecast_summary.dates %}
                    <tr>
                        <td>{{date}}</td>
                        {% for model in forecast_summary.models %}
                        <td>
                            {% if forecast_summary.values[date][model] is defined %}
                            <span class="metric-value">{{forecast_summary.values[date][model]|round(4)}}</span>
                            {% else %}
                            -
                            {% endif %}
                        </td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
        </div>
        {% endif %}
        
        {% if backtest_data %}
        <div class="section">
            <h2>Análisis de Backtesting</h2>
            
            {% if 'rmse_evolution' in charts %}
            <div class="chart-container">
                <h3>Evolución de RMSE por Subperíodo</h3>
                <img class="chart" src="data:image/png;base64,{{charts.rmse_evolution}}" alt="Evolución de RMSE">
            </div>
            {% endif %}
            
            {% if 'hit_dir_evolution' in charts %}
            <div class="chart-container">
                <h3>Evolución de Hit Direction por Subperíodo</h3>
                <img class="chart" src="data:image/png;base64,{{charts.hit_dir_evolution}}" alt="Evolución de Hit Direction">
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <div class="section">
            <h2>Notas Metodológicas</h2>
            <p>Los modelos ARIMA/SARIMAX se han entrenado siguiendo las mejores prácticas para series temporales financieras:</p>
            <ul>
                <li>Validación temporal (walk-forward) para evitar data leakage</li>
                <li>Optimización de hiperparámetros con metodología híbrida (auto_arima + búsqueda refinada)</li>
                <li>Evaluación de métricas de error (RMSE, MAE) y direccionales (Hit Direction)</li>
                <li>Análisis por subperíodos para detectar cambios en el rendimiento a lo largo del tiempo</li>
            </ul>
        </div>
        
        <footer>
            <p>Informe generado por el pipeline de Series Temporales &copy; {{current_year}}</p>
        </footer>
    </body>
    </html>
    """
    
    # Preparar datos para la plantilla
    template_data = {
        'instrument': instrument,
        'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'current_year': datetime.now().year,
        'charts': charts,
        'include_forecast': include_forecast,
        'backtest_data': backtest_data is not None
    }
    
    # Preparar resumen de modelos
    models_summary = []
    metrics_set = set()
    
    for model_name, model_info in models_data.items():
        model_summary = {
            'name': model_name,
            'metrics': {}
        }
        
        if 'metrics' in model_info:
            model_summary['metrics'] = model_info['metrics']
            metrics_set.update(model_info['metrics'].keys())
        
        models_summary.append(model_summary)
    
    # Priorizar métricas clave
    metrics_columns = []
    for key_metric in ['RMSE', 'MAE', 'Hit_Direction', 'R2']:
        if key_metric in metrics_set:
            metrics_columns.append(key_metric)
    
    # Añadir el resto de métricas
    for metric in sorted(metrics_set):
        if metric not in metrics_columns:
            metrics_columns.append(metric)
    
    template_data['models_summary'] = models_summary
    template_data['metrics_columns'] = metrics_columns
    
    # Preparar resumen de pronósticos
    if include_forecast and forecast_data:
        forecast_summary = {
            'models': list(forecast_data.keys()),
            'dates': set(),
            'values': {}
        }
        
        for model_name, forecast_df in forecast_data.items():
            if 'forecast' in forecast_df.columns:
                for idx, row in forecast_df.iterrows():
                    date_str = idx.strftime('%Y-%m-%d') if isinstance(idx, pd.Timestamp) else str(idx)
                    forecast_summary['dates'].add(date_str)
                    
                    if date_str not in forecast_summary['values']:
                        forecast_summary['values'][date_str] = {}
                    
                    forecast_summary['values'][date_str][model_name] = row['forecast']
        
        # Ordenar fechas
        forecast_summary['dates'] = sorted(forecast_summary['dates'])
        
        template_data['forecast_summary'] = forecast_summary
    
    # Renderizar plantilla
    try:
        template = Template(html_template)
        html_content = template.render(**template_data)
        
        # Guardar archivo HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    except Exception as e:
        return None

def generate_executive_summary(
    instruments_data: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: Path,
    include_charts: bool = True
) -> Path:
    """
    Genera un resumen ejecutivo para todos los instrumentos.
    
    Args:
        instruments_data: Datos por instrumento y modelo
        output_dir: Directorio de salida
        include_charts: Si se deben incluir gráficos
    
    Returns:
        Ruta al archivo Excel generado
    """
    # Asegurar que existe el directorio
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Nombre del archivo de salida
    output_file = output_dir / "resumen_ejecutivo.xlsx"
    
    # Preparar datos para informe
    summary_data = []
    
    for instrument, models in instruments_data.items():
        for model_name, model_info in models.items():
            row = {
                'Instrumento': instrument,
                'Modelo': model_name
            }
            
            # Añadir métricas
            if 'metrics' in model_info:
                for metric, value in model_info['metrics'].items():
                    row[metric] = value
            
            # Añadir período si está disponible
            if 'period' in model_info:
                if 'start' in model_info['period'] and model_info['period']['start'] is not None:
                    row['Fecha Inicio'] = model_info['period']['start']
                if 'end' in model_info['period'] and model_info['period']['end'] is not None:
                    row['Fecha Fin'] = model_info['period']['end']
            
            # Añadir observaciones
            if 'num_observations' in model_info:
                row['Observaciones'] = model_info['num_observations']
            
            summary_data.append(row)
    
    if not summary_data:
        return None
    
    # Crear DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Guardar como Excel
    try:
        # Intentar usar xlsxwriter para mejor formato
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # Guardar resumen completo
            df_summary.to_excel(writer, sheet_name='Resumen Completo', index=False)
            
            # Dar formato
            workbook = writer.book
            worksheet = writer.sheets['Resumen Completo']
            
            # Formatos
            num_format = workbook.add_format({'num_format': '0.0000'})
            date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
            
            # Aplicar formatos
            for i, col in enumerate(df_summary.columns):
                column_width = max(len(str(col)) + 2, 14)
                worksheet.set_column(i, i, column_width)
                
                if col in ['Fecha Inicio', 'Fecha Fin']:
                    worksheet.set_column(i, i, column_width, date_format)
                elif col not in ['Instrumento', 'Modelo', 'Observaciones']:
                    worksheet.set_column(i, i, column_width, num_format)
            
            # Crear una hoja resumen por instrumento
            for instrument in instruments_data.keys():
                instrument_data = df_summary[df_summary['Instrumento'] == instrument].copy()
                
                if not instrument_data.empty:
                    # Ordenar por RMSE si está disponible
                    if 'RMSE' in instrument_data.columns:
                        instrument_data = instrument_data.sort_values('RMSE')
                    
                    # Guardar en hoja propia
                    sheet_name = instrument[:31]  # Limitar longitud
                    instrument_data.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Dar formato
                    worksheet = writer.sheets[sheet_name]
                    
                    # Aplicar formatos
                    for i, col in enumerate(instrument_data.columns):
                        column_width = max(len(str(col)) + 2, 14)
                        worksheet.set_column(i, i, column_width)
                        
                        if col in ['Fecha Inicio', 'Fecha Fin']:
                            worksheet.set_column(i, i, column_width, date_format)
                        elif col not in ['Instrumento', 'Modelo', 'Observaciones']:
                            worksheet.set_column(i, i, column_width, num_format)
            
            # Crear hoja con las mejores métricas por instrumento
            best_by_instrument = []
            
            for instrument, models in instruments_data.items():
                instrument_models = [{'name': k, 'metrics': v.get('metrics', {})} for k, v in models.items()]
                
                # Encontrar mejor modelo por RMSE
                if instrument_models and all('RMSE' in m['metrics'] for m in instrument_models):
                    best_rmse_model = min(instrument_models, key=lambda x: x['metrics'].get('RMSE', float('inf')))
                    
                    if 'RMSE' in best_rmse_model['metrics']:
                        best_by_instrument.append({
                            'Instrumento': instrument,
                            'Mejor Modelo (RMSE)': best_rmse_model['name'],
                            'RMSE': best_rmse_model['metrics']['RMSE']
                        })
            
            if best_by_instrument:
                pd.DataFrame(best_by_instrument).to_excel(writer, sheet_name='Mejores Modelos', index=False)
    
    except ImportError:
        # Fallback si no está xlsxwriter
        df_summary.to_excel(output_file, index=False)
    
    return output_file

# Función principal para generar todos los informes para un instrumento
def generate_all_reports(
    instrument: str,
    models_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    forecast_data: Optional[Dict[str, pd.DataFrame]] = None,
    backtest_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    plots_paths: Optional[Dict[str, Dict[str, Path]]] = None
) -> Dict[str, Path]:
    """
    Genera todos los informes para un instrumento.
    
    Args:
        instrument: Instrumento financiero
        models_data: Datos de los modelos
        output_dir: Directorio de salida
        forecast_data: Datos de pronóstico
        backtest_data: Datos de backtesting
        plots_paths: Rutas a gráficos por modelo
    
    Returns:
        Diccionario con rutas a los informes generados
    """
    # Asegurar que existe el directorio
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Diccionario para almacenar rutas a informes
    reports = {}
    
    # 1. Resúmenes individuales de modelos
    for model_name, model_info in models_data.items():
        if 'metrics' in model_info and 'params' in model_info:
            # Extraer período de entrenamiento
            training_period = ('N/A', 'N/A')
            if 'period' in model_info:
                start = model_info['period'].get('start', 'N/A')
                end = model_info['period'].get('end', 'N/A')
                training_period = (start, end)
            
            # Extraer rutas a gráficos específicos del modelo
            model_plots = None
            if plots_paths and model_name in plots_paths:
                model_plots = plots_paths[model_name]
            
            # Generar resumen
            summary_file = generate_model_summary(
                model_name=model_name,
                instrument=instrument,
                metrics=model_info['metrics'],
                model_params=model_info['params'],
                training_period=training_period,
                output_dir=output_dir,
                include_plots=model_plots is not None,
                plots_paths=model_plots
            )
            
            if summary_file:
                reports[f'summary_{model_name}'] = summary_file
    
    # 2. Informe comparativo
    comparison_plots = {}
    if plots_paths and '_comparison' in plots_paths:
        comparison_plots = plots_paths['_comparison']
    
    comparison_file = generate_comparison_report(
        models_data=models_data,
        instrument=instrument,
        output_dir=output_dir,
        include_plots=comparison_plots is not None,
        comparison_plots=comparison_plots
    )
    
    if comparison_file:
        reports['comparison'] = comparison_file
    
    # 3. Informe de pronóstico
    if forecast_data:
        forecast_plots = {}
        if plots_paths and '_forecast' in plots_paths:
            forecast_plots = plots_paths['_forecast']
        
        forecast_file = generate_forecast_report(
            forecast_data=forecast_data,
            instrument=instrument,
            output_dir=output_dir,
            include_plots=forecast_plots is not None,
            forecast_plots=forecast_plots
        )
        
        if forecast_file:
            reports['forecast'] = forecast_file
    
    # 4. Informe de backtesting
    if backtest_data and instrument in backtest_data:
        backtest_plots = {}
        if plots_paths and '_backtest' in plots_paths:
            backtest_plots = plots_paths['_backtest']
        
        backtest_file = generate_backtest_report(
            backtest_data={instrument: backtest_data[instrument]},
            instrument=instrument,
            output_dir=output_dir,
            include_plots=backtest_plots is not None,
            backtest_plots=backtest_plots
        )
        
        if backtest_file:
            reports['backtest'] = backtest_file
    
    # 5. Informe PDF
    pdf_file = generate_pdf_report(
        instrument=instrument,
        models_data=models_data,
        output_dir=output_dir,
        include_forecast=forecast_data is not None,
        forecast_data=forecast_data,
        backtest_data=backtest_data
    )
    
    if pdf_file:
        reports['pdf'] = pdf_file
    
    # 6. Informe HTML
    html_file = generate_html_report(
        instrument=instrument,
        models_data=models_data,
        output_dir=output_dir,
        include_forecast=forecast_data is not None,
        forecast_data=forecast_data,
        backtest_data=backtest_data
    )
    
    if html_file:
        reports['html'] = html_file
    
    return reports
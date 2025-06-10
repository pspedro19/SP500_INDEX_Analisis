#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preparación de Salida y Generación de Reportes
==============================================
Combina dos funcionalidades principales:
1. Adaptación de resultados para visualización en PowerBI, API y otras herramientas
2. Generación de reportes avanzados con análisis detallado por instrumento y modelo

Salida:
- CSV con formato español (punto y coma, coma decimal) para PowerBI
- Versión JSON para API/microservicios
- Resumen ejecutivo con métricas clave
- Reportes avanzados con análisis detallado por instrumento
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union

# Argumentos para habilitar feature adicional
parser = argparse.ArgumentParser(description="Preparación de salida y generación de reportes")
parser.add_argument("--plot-price-scale", action="store_true",
                    help="Generar gráficos en escala de precios (deshacer log)")
args = parser.parse_args()

# Configuración
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
INPUT_FILE = PROJECT_ROOT / "data" / "4_results" / "all_models_with_ensemble.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "4_results"
LOG_DIR = PROJECT_ROOT / "logs"

# Configuración de logging
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"prepare_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Importar módulos de utilidades para reportes avanzados
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
    from time_series.utils import generate_reports, plots
except ImportError:
    # Intentar con rutas alternativas durante desarrollo
    try:
        sys.path.append(os.path.abspath("../pipelines"))
        from time_series.utils import generate_reports, plots
    except ImportError:
        logging.warning("No se pudieron importar los módulos de generación de reportes avanzados.")

#----------------------------------------------------------------------------
# Funciones auxiliares para manejo seguro de datos
#----------------------------------------------------------------------------

def safe_get_forecast_data(forecast_data, model):
    """Obtiene datos de pronóstico de forma segura."""
    if model not in forecast_data:
        return None
    
    forecast = forecast_data[model]
    # Verificar si es un DataFrame o un diccionario
    if isinstance(forecast, pd.DataFrame):
        return forecast
    elif isinstance(forecast, dict):
        # Convertir a DataFrame si es un diccionario
        return pd.DataFrame(forecast)
    else:
        return None

def safe_format_date(date_obj):
    """Formatea fechas de forma segura."""
    if isinstance(date_obj, str):
        try:
            return pd.to_datetime(date_obj).strftime('%Y-%m-%d')
        except:
            return str(date_obj)
    elif hasattr(date_obj, 'strftime'):  # es un objeto datetime
        return date_obj.strftime('%Y-%m-%d')
    else:
        return str(date_obj)

def count_reports_safely(reports):
    """Cuenta el número de reportes de forma segura."""
    if isinstance(reports, dict):
        return len(reports)
    elif isinstance(reports, Path):
        return 1
    elif hasattr(reports, "__len__"):
        return len(reports)
    else:
        return 1 if reports else 0

#----------------------------------------------------------------------------
# Funciones para preparación de salida
#----------------------------------------------------------------------------

def load_raw_series(instrument: str) -> pd.Series:
    """Carga la serie temporal en escala original (sin transformar)."""
    try:
        raw_file = Path(f"data/processed/{instrument}_raw_data.csv")
        if raw_file.exists():
            df = pd.read_csv(raw_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            # Buscar columna con 'close' o similar
            for col in df.columns:
                if 'close' in col.lower():
                    return df[col]
            # Si no encuentra, usar primera columna numérica
            return df.select_dtypes(include=['number']).iloc[:,0]
        return None
    except Exception as e:
        logging.error(f"Error cargando serie raw para {instrument}: {e}")
        return None

def split_data(df, min_train_size=252):
    """Divide los datos según esquema 7-2-1."""
    total_len = len(df)
    test_size = min(252, int(total_len * 0.1))
    val_size = min(504, int(total_len * 0.2))
    
    if total_len < (val_size + test_size + min_train_size):
        available = total_len - min_train_size
        test_size = max(21, int(available * 0.33))
        val_size = max(42, available - test_size)
    
    train_end = total_len - (val_size + test_size)
    val_end = total_len - test_size
    
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    
    return df_train, df_val, df_test

def verify_input_path(input_file: Path) -> Path:
    """Verifica el path de entrada y busca alternativas si es necesario."""
    if input_file.exists():
        return input_file
    
    # Si no existe el archivo con ensemble, intentar con el original
    alt_input = PROJECT_ROOT / "data" / "2_trainingdata_ts" / "all_models_predictions.csv"
    if alt_input.exists():
        logging.warning(f"Archivo de entrada original no encontrado: {input_file}")
        logging.info(f"Usando archivo alternativo: {alt_input}")
        return alt_input
    
    # Si tampoco existe, buscar el archivo más reciente
    logging.warning(f"Archivo alternativo tampoco encontrado: {alt_input}")
    data_dir = PROJECT_ROOT / "data"
    
    # Buscar recursivamente
    csv_files = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith("predictions.csv") or file.endswith("ensemble.csv"):
                full_path = Path(root) / file
                csv_files.append((full_path, full_path.stat().st_mtime))
    
    if csv_files:
        # Ordenar por tiempo de modificación (más reciente primero)
        csv_files.sort(key=lambda x: x[1], reverse=True)
        newest_file = csv_files[0][0]
        logging.info(f"Usando el archivo más reciente encontrado: {newest_file}")
        return newest_file
    
    logging.error(f"No se encontró ningún archivo de predicciones")
    raise FileNotFoundError(f"No se encontró ningún archivo de predicciones")

def adapt_date_format(df: pd.DataFrame) -> pd.DataFrame:
    """Adapta el formato de la columna de fecha."""
    if 'date' in df.columns:
        # Convertir a datetime si no lo es
        if not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Convertir a formato español (DD/MM/YYYY)
        df['date'] = df['date'].dt.strftime('%d/%m/%Y')
    
    return df

def format_decimal_values(df: pd.DataFrame, decimal_separator: str = ',') -> pd.DataFrame:
    """Convierte los decimales al formato especificado."""
    numeric_cols = ['Valor_Real', 'Valor_Predicho', 'RMSE']
    
    for col in numeric_cols:
        if col in df.columns:
            # Primero asegurar que es string
            df[col] = df[col].astype(str)
            
            # Reemplazar punto por el separador deseado
            if decimal_separator == ',':
                df[col] = df[col].str.replace('.', ',')
            elif decimal_separator == '.':
                df[col] = df[col].str.replace(',', '.')
    
    return df

def sanitize_hyperparameters(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que la columna de hiperparámetros esté bien formateada."""
    # Buscar la columna de hiperparámetros (puede tener varios nombres)
    hyperparam_cols = [col for col in df.columns if 'parámetros' in col.lower() or 'parametros' in col.lower()]
    
    if hyperparam_cols:
        hyperparam_col = hyperparam_cols[0]
        
        # Asegurar que los valores JSON están bien entrecomillados
        df[hyperparam_col] = df[hyperparam_col].apply(
            lambda x: f'"{x}"' if not pd.isna(x) and not str(x).startswith('"') else x
        )
    
    return df

def create_powerbi_version(df: pd.DataFrame, output_dir: Path) -> Path:
    """Crea versión optimizada para PowerBI (CSV con formato español)."""
    # Adaptar formato para PowerBI
    powerbi_df = df.copy()
    
    # Adaptar fecha
    powerbi_df = adapt_date_format(powerbi_df)
    
    # Convertir decimales a formato español (coma)
    powerbi_df = format_decimal_values(powerbi_df, decimal_separator=',')
    
    # Sanitizar hiperparámetros
    powerbi_df = sanitize_hyperparameters(powerbi_df)
    
    # Guardar con separador punto y coma
    output_file = output_dir / "archivo_para_powerbi.csv"
    powerbi_df.to_csv(output_file, index=False, sep=';', encoding='utf-8', quoting=1)
    
    logging.info(f" CSV para Power BI (formato español) guardado en: {output_file}")
    return output_file

def create_api_version(df: pd.DataFrame, output_dir: Path) -> Path:
    """Crea versión para API (JSON)."""
    # Adaptar para API
    api_df = df.copy()
    
    # Convertir fecha a formato ISO
    if 'date' in api_df.columns and not pd.api.types.is_datetime64_dtype(api_df['date']):
        api_df['date'] = pd.to_datetime(api_df['date'], errors='coerce')
    
    if 'date' in api_df.columns:
        api_df['date'] = api_df['date'].dt.strftime('%Y-%m-%d')
    
    # Convertir columnas numéricas a números reales
    for col in ['Valor_Real', 'Valor_Predicho', 'RMSE']:
        if col in api_df.columns:
            api_df[col] = pd.to_numeric(api_df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Parsear hiperparámetros a JSON real
    hyperparam_cols = [col for col in api_df.columns if 'parámetros' in col.lower() or 'parametros' in col.lower()]
    
    if hyperparam_cols:
        hyperparam_col = hyperparam_cols[0]
        
        def parse_params(param_str):
            if pd.isna(param_str):
                return {}
            
            try:
                # Eliminar comillas extras si están presentes
                if param_str.startswith('"') and param_str.endswith('"'):
                    param_str = param_str[1:-1]
                
                return json.loads(param_str)
            except:
                return {"error": "invalid_json"}
        
        api_df[hyperparam_col] = api_df[hyperparam_col].apply(parse_params)
    
    # Guardar como JSON con estructura orientada a registros
    output_file = output_dir / "predictions_api.json"
    
    # Convertir a registros JSON
    result = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "record_count": len(api_df),
            "instruments": api_df['Tipo_Mercado'].unique().tolist() if 'Tipo_Mercado' in api_df.columns else [],
            "models": api_df['Modelo'].unique().tolist() if 'Modelo' in api_df.columns else []
        },
        "predictions": api_df.to_dict(orient='records')
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logging.info(f" JSON para API guardado en: {output_file}")
    return output_file

def create_executive_summary(df: pd.DataFrame, output_dir: Path) -> Path:
    """Crea un resumen ejecutivo con las métricas clave."""
    # Filtrar solo datos de test (para métricas precisas)
    if 'Periodo' in df.columns:
        test_df = df[df['Periodo'] == 'Test'].copy()
    else:
        test_df = df.copy()
        logging.warning("No se encontró columna 'Periodo', usando todos los datos para el resumen")
    
    # Convertir columnas numéricas
    for col in ['Valor_Real', 'Valor_Predicho', 'RMSE']:
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Calcular métricas por instrumento y modelo
    summary_data = []
    
    # Agrupar por instrumento y modelo
    if 'Tipo_Mercado' in test_df.columns and 'Modelo' in test_df.columns:
        grouped = test_df.groupby(['Tipo_Mercado', 'Modelo'])
        
        for (instrument, model), group in grouped:
            # Solo procesar si hay suficientes datos
            if len(group) < 5:
                continue
            
            # Calcular métricas
            if 'Valor_Real' in group.columns and 'Valor_Predicho' in group.columns:
                real = group['Valor_Real'].dropna()
                pred = group['Valor_Predicho'].dropna()
                
                # Solo calcular si hay datos válidos
                if len(real) >= 5 and len(pred) >= 5:
                    # Calcular RMSE
                    rmse = np.sqrt(np.mean((real - pred) ** 2))
                    
                    # Calcular MAE
                    mae = np.mean(np.abs(real - pred))
                    
                    # Calcular MAPE evitando divisiones por cero
                    if (real == 0).any():
                        mape = np.nan
                    else:
                        mape = np.mean(np.abs((real - pred) / real)) * 100
                    
                    # Calcular dirección
                    real_diff = real.diff().dropna()
                    pred_diff = pred.diff().dropna()
                    
                    # Ajustar longitudes si son diferentes
                    min_len = min(len(real_diff), len(pred_diff))
                    if min_len > 0:
                        real_diff = real_diff.iloc[:min_len]
                        pred_diff = pred_diff.iloc[:min_len]
                        
                        hit_direction = np.mean(np.sign(real_diff) == np.sign(pred_diff)) * 100
                    else:
                        hit_direction = np.nan
                    
                    # Añadir al resumen
                    summary_data.append({
                        "Instrumento": instrument,
                        "Modelo": model,
                        "Número de Predicciones": len(group),
                        "RMSE": rmse,
                        "MAE": mae,
                        "MAPE (%)": mape,
                        "Hit Direction (%)": hit_direction
                    })
    
    # Crear DataFrame de resumen
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Ordenar por instrumento y RMSE (ascendente)
        summary_df = summary_df.sort_values(['Instrumento', 'RMSE'])
        
        # Guardar como Excel
        output_file = output_dir / "resumen_ejecutivo.xlsx"
        
        # Crear un Excel con formato
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Resumen', index=False)
            
            # Formatear
            workbook = writer.book
            worksheet = writer.sheets['Resumen']
            
            # Formato para porcentajes
            pct_format = workbook.add_format({'num_format': '0.00%'})
            
            # Formato para números
            num_format = workbook.add_format({'num_format': '0.0000'})
            
            # Aplicar formatos
            for i, col in enumerate(summary_df.columns):
                column_width = max(len(col) + 2, 12)
                worksheet.set_column(i, i, column_width)
                
                if 'RMSE' in col or 'MAE' in col:
                    worksheet.set_column(i, i, column_width, num_format)
                elif '%' in col:
                    worksheet.set_column(i, i, column_width, pct_format)
        
        logging.info(f" Resumen ejecutivo guardado en: {output_file}")
        return output_file
    else:
        logging.warning("No se pudieron calcular métricas para el resumen ejecutivo")
        return None

def create_forecast_summary(df: pd.DataFrame, output_dir: Path) -> Path:
    """Crea un resumen de las predicciones futuras."""
    # Filtrar solo datos de forecast
    if 'Periodo' in df.columns:
        forecast_df = df[df['Periodo'] == 'Forecast'].copy()
    else:
        forecast_df = df.copy()
        logging.warning("No se encontró columna 'Periodo', usando todos los datos para el resumen de forecast")
    
    # Convertir columnas numéricas
    if 'Valor_Predicho' in forecast_df.columns:
        forecast_df['Valor_Predicho'] = pd.to_numeric(forecast_df['Valor_Predicho'].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Formatear la fecha
    if 'date' in forecast_df.columns:
        forecast_df['date'] = pd.to_datetime(forecast_df['date'], errors='coerce')
    
    # Crear resumen agrupado por instrumento, modelo y fecha
    if 'Tipo_Mercado' in forecast_df.columns and 'Modelo' in forecast_df.columns:
        # Pivotar para tener modelos como columnas
        if 'date' in forecast_df.columns:
            pivot_df = forecast_df.pivot_table(
                index=['date', 'Tipo_Mercado'],
                columns='Modelo',
                values='Valor_Predicho',
                aggfunc='first'
            ).reset_index()
            
            # Ordenar por fecha
            pivot_df = pivot_df.sort_values(['Tipo_Mercado', 'date'])
            
            # Guardar como Excel
            output_file = output_dir / "resumen_forecast.xlsx"
            
            # Crear un Excel con formato
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                # Un sheet por instrumento
                for instrument in pivot_df['Tipo_Mercado'].unique():
                    inst_df = pivot_df[pivot_df['Tipo_Mercado'] == instrument].drop(columns=['Tipo_Mercado'])
                    inst_df.to_excel(writer, sheet_name=instrument, index=False)
                    
                    # Formatear
                    workbook = writer.book
                    worksheet = writer.sheets[instrument]
                    
                    # Formato para fecha
                    date_format = workbook.add_format({'num_format': 'dd/mm/yyyy'})
                    
                    # Formato para números
                    num_format = workbook.add_format({'num_format': '0.0000'})
                    
                    # Aplicar formatos
                    worksheet.set_column(0, 0, 12, date_format)  # Fecha
                    
                    for i, col in enumerate(inst_df.columns):
                        if i > 0:  # Saltamos la columna de fecha
                            column_width = max(len(col) + 2, 12)
                            worksheet.set_column(i, i, column_width, num_format)
            
            logging.info(f" Resumen de forecast guardado en: {output_file}")
            return output_file
        else:
            logging.warning("No se encontró columna 'date', no se puede crear resumen de forecast")
            return None
    else:
        logging.warning("No se encontraron columnas necesarias para crear resumen de forecast")
        return None

#----------------------------------------------------------------------------
# Funciones para generación de reportes avanzados
#----------------------------------------------------------------------------

def generate_advanced_reports(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """
    Genera informes avanzados utilizando la biblioteca de generación de reportes.
    
    Args:
        df: DataFrame con predicciones y resultados
        output_dir: Directorio de salida
    
    Returns:
        Diccionario con rutas a los informes generados
    """
    if 'generate_reports' not in globals():
        logging.warning("Módulo de reportes avanzados no disponible. Solo se generarán reportes básicos.")
        return {}
    
    logging.info("Generando reportes avanzados...")
    reports_paths = {}
    
    try:
        # Verificar columnas necesarias
        required_cols = ['date', 'Valor_Real', 'Valor_Predicho', 'Modelo', 'Tipo_Mercado', 'Periodo']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logging.error(f"Faltan columnas necesarias para reportes avanzados: {missing}")
            return {}
        
        # Convertir fecha si es necesario
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Procesar por cada instrumento
        instruments = df['Tipo_Mercado'].unique()
        
        # Directorio para reportes
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(exist_ok=True, parents=True)
        
        for instrument in instruments:
            try:
                logging.info(f"Generando reportes para {instrument}...")
                
                # Datos para este instrumento
                instrument_data = df[df['Tipo_Mercado'] == instrument].copy()
                
                # 1. Preparar datos de modelos
                models_data = {}
                for model in instrument_data['Modelo'].unique():
                    model_data = instrument_data[instrument_data['Modelo'] == model]
                    
                    # Obtener métricas (del período de test)
                    test_data = model_data[model_data['Periodo'] == 'Test']
                    
                    if test_data.empty:
                        continue
                    
                    metrics = {}
                    # Usar primera fila que tenga RMSE como valor no nulo
                    rmse_rows = test_data[~test_data['RMSE'].isna()]
                    if not rmse_rows.empty:
                        metrics['RMSE'] = float(rmse_rows['RMSE'].iloc[0])
                    
                    # Calcular otras métricas si no están disponibles
                    real = pd.to_numeric(test_data['Valor_Real'], errors='coerce')
                    pred = pd.to_numeric(test_data['Valor_Predicho'], errors='coerce')
                    
                    # Filtrar valores válidos
                    mask = ~(real.isna() | pred.isna())
                    real_valid = real[mask].values
                    pred_valid = pred[mask].values
                    
                    if len(real_valid) > 0:
                        # Solo calcular si no están ya en el DataFrame
                        if 'RMSE' not in metrics:
                            metrics['RMSE'] = np.sqrt(np.mean((real_valid - pred_valid) ** 2))
                        
                        metrics['MAE'] = np.mean(np.abs(real_valid - pred_valid))
                        
                        # MAPE (evitar división por cero)
                        if not np.any(real_valid == 0):
                            metrics['MAPE'] = np.mean(np.abs((real_valid - pred_valid) / real_valid)) * 100
                        
                        # R2
                        if len(real_valid) > 1:
                            # Calcular R² manualmente (1 - SSres/SStot)
                            ss_res = np.sum((real_valid - pred_valid) ** 2)
                            ss_tot = np.sum((real_valid - np.mean(real_valid)) ** 2)
                            if ss_tot > 0:
                                metrics['R2'] = 1 - (ss_res / ss_tot)
                        
                        # Hit Direction
                        real_diff = np.diff(real_valid)
                        pred_diff = np.diff(pred_valid)
                        
                        if len(real_diff) > 0:
                            hit_dir = np.mean(np.sign(real_diff) == np.sign(pred_diff)) * 100
                            metrics['Hit_Direction'] = hit_dir
                    
                    # Parámetros del modelo (extraer del campo hyperparámetros si existe)
                    params = {}
                    hyperparam_cols = [c for c in model_data.columns if 'parámetro' in c.lower()]
                    
                    if hyperparam_cols:
                        hyperparam_col = hyperparam_cols[0]
                        # Usar la primera fila no nula
                        param_rows = model_data[~model_data[hyperparam_col].isna()]
                        
                        if not param_rows.empty:
                            param_str = param_rows[hyperparam_col].iloc[0]
                            try:
                                # Intentar parsear como JSON
                                if isinstance(param_str, str):
                                    # Eliminar comillas extras si existen
                                    if param_str.startswith('"') and param_str.endswith('"'):
                                        param_str = param_str[1:-1]
                                    params = json.loads(param_str)
                            except:
                                params = {"raw": str(param_str)}
                    
                    # Datos históricos (serie temporal real y predicción)
                    historical = {}
                    
                    # Serie temporal real (todo excepto forecast)
                    real_series = model_data[model_data['Periodo'] != 'Forecast'].copy()
                    if not real_series.empty:
                        historical['real'] = pd.Series(
                            pd.to_numeric(real_series['Valor_Real'], errors='coerce').values,
                            index=real_series['date']
                        )
                        
                        historical['prediction'] = pd.Series(
                            pd.to_numeric(real_series['Valor_Predicho'], errors='coerce').values,
                            index=real_series['date']
                        )
                    
                    # Almacenar información del modelo
                    models_data[model] = {
                        'metrics': metrics,
                        'params': params,
                        'historical': historical
                    }
                
                # 2. Preparar datos de pronóstico
                forecast_data = {}
                for model in instrument_data['Modelo'].unique():
                    forecast_rows = instrument_data[
                        (instrument_data['Modelo'] == model) & 
                        (instrument_data['Periodo'] == 'Forecast')
                    ].copy()
                    
                    if not forecast_rows.empty:
                        try:
                            # Convertir a formato adecuado
                            forecast_df = pd.DataFrame({
                                'date': forecast_rows['date'],
                                'forecast': pd.to_numeric(forecast_rows['Valor_Predicho'], errors='coerce')
                            })
                            
                            # Establecer fecha como índice
                            forecast_df = forecast_df.set_index('date')
                            
                            # Guardar en diccionario
                            forecast_data[model] = forecast_df
                        except Exception as e:
                            logging.error(f"Error preparando datos de pronóstico para {model}: {str(e)}")
                            # Crear un DataFrame vacío como fallback
                            forecast_data[model] = pd.DataFrame(columns=['forecast'])
                
                # 3. Cargar datos de backtesting si están disponibles
                backtest_data = None
                metrics_dir = output_dir.parent / "5_metrics"
                
                if metrics_dir.exists():
                    # Buscar archivo de backtesting para este instrumento
                    backtest_files = list(metrics_dir.glob(f"{instrument}*metricas.csv"))
                    
                    if backtest_files:
                        try:
                            backtest_df = pd.read_csv(backtest_files[0])
                            
                            # Convertir a formato para generate_reports
                            backtest_data = {instrument: {}}
                            
                            for _, row in backtest_df.iterrows():
                                model = row['Modelo']
                                metrics_dict = {
                                    col: row[col] for col in backtest_df.columns 
                                    if col not in ['Instrumento', 'Modelo', 'Fecha_Inicio', 'Fecha_Fin', 'Observaciones']
                                }
                                
                                # Formatear fechas de forma segura
                                period_data = {
                                    'start': safe_format_date(row['Fecha_Inicio']) if 'Fecha_Inicio' in row else None,
                                    'end': safe_format_date(row['Fecha_Fin']) if 'Fecha_Fin' in row else None
                                }
                                
                                backtest_data[instrument][model] = {
                                    'metrics': metrics_dict,
                                    'num_observations': row['Observaciones'] if 'Observaciones' in row else 0,
                                    'period': period_data
                                }
                        except Exception as e:
                            logging.warning(f"Error cargando datos de backtesting: {str(e)}")
                
                # 4. Buscar gráficos generados
                plots_paths = {}
                charts_dir = output_dir / "charts"
                
                if charts_dir.exists():
                    # Por modelo
                    for model in models_data.keys():
                        model_plots = {}
                        for plot_type in ['comparison', 'forecast', 'diagnostics', 'dashboard']:
                            plot_file = charts_dir / f"{instrument}_{model}_{plot_type}.png"
                            if plot_file.exists():
                                model_plots[plot_type] = plot_file
                        
                        if model_plots:
                            plots_paths[model] = model_plots
                    
                    # Comparativos
                    comparison_plots = {}
                    for plot_type in ['ensemble_comparison', 'radar_comparison', 'hit_direction_comparison', 'rmse_comparison']:
                        plot_file = charts_dir / f"{instrument}_{plot_type}.png"
                        if plot_file.exists():
                            comparison_plots[plot_type] = plot_file
                    
                    if comparison_plots:
                        plots_paths['_comparison'] = comparison_plots
                    
                    # Pronósticos
                    forecast_plots = {}
                    ensemble_forecast_file = charts_dir / f"{instrument}_ensemble_forecast.png"
                    if ensemble_forecast_file.exists():
                        forecast_plots['ensemble_forecast'] = ensemble_forecast_file
                    
                    if forecast_plots:
                        plots_paths['_forecast'] = forecast_plots
                
                # 5. Generar todos los reportes con manejo de errores mejorado
                try:
                    instrument_reports = generate_reports.generate_all_reports(
                        instrument=instrument,
                        models_data=models_data,
                        output_dir=reports_dir / instrument,
                        forecast_data=forecast_data,
                        backtest_data=backtest_data,
                        plots_paths=plots_paths
                    )
                    
                    if instrument_reports:
                        reports_paths[instrument] = instrument_reports
                        logging.info(f"Generados {count_reports_safely(instrument_reports)} reportes para {instrument}")
                except Exception as e:
                    logging.error(f"Error generando reportes para {instrument}: {str(e)}")
                    # Continuar con el siguiente instrumento
            except Exception as e:
                logging.error(f"Error procesando datos para {instrument}: {str(e)}")
                # Continuar con el siguiente instrumento
                continue
        
        # 6. Generar resumen ejecutivo si hay datos de múltiples instrumentos
        if len(reports_paths) > 1:
            try:
                # Combinar datos de todos los instrumentos
                all_models_data = {}
                for instrument, _ in reports_paths.items():
                    all_models_data[instrument] = models_data
                
                # Generar resumen ejecutivo
                exec_summary = generate_reports.generate_executive_summary(
                    instruments_data=all_models_data,
                    output_dir=reports_dir
                )
                
                if exec_summary:
                    reports_paths['executive_summary'] = exec_summary
                    logging.info(f"Generado resumen ejecutivo en {exec_summary}")
            except Exception as e:
                logging.error(f"Error generando resumen ejecutivo: {str(e)}")
    
    except Exception as e:
        logging.error(f"Error general en generación de reportes avanzados: {str(e)}")
    
    return reports_paths

#----------------------------------------------------------------------------
# Función principal
#----------------------------------------------------------------------------

def main():
    """Función principal."""
    try:
        logging.info("Iniciando preparación de salida y generación de reportes")
        
        # Crear directorio de salida si no existe
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        
        # Verificar y obtener path de entrada
        input_file = verify_input_path(INPUT_FILE)
        
        # Verificar existencia del archivo de entrada
        if not input_file.exists():
            logging.error(f"El archivo de entrada no existe: {input_file}")
            return 1
        
        # Leer el archivo
        logging.info(f"Leyendo archivo: {input_file}")
        
        try:
            # Intentar primero con separador estándar (coma)
            df = pd.read_csv(input_file, encoding='utf-8')
        except:
            try:
                # Si falla, probar con punto y coma
                df = pd.read_csv(input_file, encoding='utf-8', sep=';')
            except Exception as e:
                logging.error(f"Error leyendo el archivo: {str(e)}")
                return 1
        
        logging.info(f"Archivo leído correctamente: {len(df)} filas, {len(df.columns)} columnas")
        
        # Crear versiones para diferentes consumidores
        logging.info("Generando versiones para diferentes consumidores...")
        powerbi_file = create_powerbi_version(df, OUTPUT_DIR)
        api_file = create_api_version(df, OUTPUT_DIR)
        summary_file = create_executive_summary(df, OUTPUT_DIR)
        forecast_file = create_forecast_summary(df, OUTPUT_DIR)
        
        # Generar reportes avanzados
        logging.info("Verificando disponibilidad de generación de reportes avanzados...")
        reports_paths = {}
        
        if 'generate_reports' in globals():
            logging.info("Módulo de reportes avanzados disponible, generando reportes...")
            reports_paths = generate_advanced_reports(df, OUTPUT_DIR)
        else:
            logging.warning("Módulo de reportes avanzados no disponible. Solo se generarán reportes básicos.")
        
        # FEATURE ADICIONAL: graficar en escala de precios
        if args.plot_price_scale and 'plots' in globals():
            charts_dir = OUTPUT_DIR / 'charts'
            charts_dir.mkdir(exist_ok=True)
            
            for instrument in df['Tipo_Mercado'].unique():
                # Extraer datos para este instrumento
                inst_data = df[df['Tipo_Mercado'] == instrument]
                
                # Para cada modelo
                for model in inst_data['Modelo'].unique():
                    try:
                        # 1) Carga la serie original sin transformar
                        raw_series = load_raw_series(instrument)
                        if raw_series is None:
                            logging.error(f"No se pudo cargar serie raw para {instrument}")
                            continue
                        
                        # 2) Reconstruye train/val/test para obtener fechas de corte
                        df_model = inst_data[inst_data['Modelo']==model].copy()
                        df_model['date'] = pd.to_datetime(df_model['date'])
                        df_model = df_model.set_index('date')
                        df_tr, df_v, df_te = split_data(df_model[['Valor_Real']])
                        
                        # 3) Recupera predicciones históricas si existen
                        historical_pred = None
                        if 'Valor_Predicho' in df_model.columns:
                            # Filtrar por validación y test
                            mask = df_model['Periodo'].isin(['Evaluacion', 'Test'])
                            if mask.any():
                                historical_pred = pd.Series(
                                    df_model.loc[mask, 'Valor_Predicho'].values,
                                    index=df_model.loc[mask].index
                                )
                        
                        # 4) Obtener datos para el pronóstico
                        frows = inst_data[(inst_data['Modelo']==model)&(inst_data['Periodo']=='Forecast')]
                        if frows.empty:
                            logging.warning(f"No hay datos de pronóstico para {instrument} - {model}")
                            continue
                        
                        # Convertir a formato numérico y aplicar exp() para deshacer log si es necesario
                        future_forecast_series = pd.Series(
                            pd.to_numeric(frows['Valor_Predicho'].astype(str).str.replace(',', '.'), errors='coerce').values,
                            index=pd.to_datetime(frows['date'])
                        )
                        
                        # Preparar intervalos de confianza si existen
                        future_ci_lower = None
                        future_ci_upper = None
                        ci_cols = [c for c in frows.columns if 'lower' in c.lower() or 'upper' in c.lower()]
                        if len(ci_cols) >= 2:
                            lower_col = [c for c in ci_cols if 'lower' in c.lower()][0]
                            upper_col = [c for c in ci_cols if 'upper' in c.lower()][0]
                            
                            future_ci_lower = pd.Series(
                                pd.to_numeric(frows[lower_col].astype(str).str.replace(',', '.'), errors='coerce').values,
                                index=pd.to_datetime(frows['date'])
                            )
                            
                            future_ci_upper = pd.Series(
                                pd.to_numeric(frows[upper_col].astype(str).str.replace(',', '.'), errors='coerce').values,
                                index=pd.to_datetime(frows['date'])
                            )
                        
                        # 4) Generar visualización en escala de precios
                        plots.plot_forecast(
                            historical=raw_series,
                            pred_series=historical_pred,
                            forecast=future_forecast_series,
                            ci_lower=future_ci_lower,
                            ci_upper=future_ci_upper,
                            title=f"Pronóstico Precio - {instrument} - {model}",
                            save_path=charts_dir / f"{instrument}_{model}_forecast_price.png",
                            instrument=instrument,
                            model_name=model,
                            train_end_date=df_tr.index[-1] if df_tr is not None and len(df_tr) > 0 else None,
                            val_end_date=df_v.index[-1] if df_v is not None and len(df_v) > 0 else None,
                            inverse_transform=None  # ya está en escala de precio
                        )
                        logging.info(f"Gráfico price-scale generado para {instrument} - {model}")
                    except Exception as e:
                        logging.error(f"Error generando gráfico price-scale para {instrument} - {model}: {str(e)}")
        
        # Mostrar resultados
        logging.info(" Proceso de preparación de salida y generación de reportes completado")
        
        if summary_file:
        
        if forecast_file:
        
        # Mostrar reportes avanzados generados - CORRECCIÓN AQUÍ
        if reports_paths:
            for instrument, reports in reports_paths.items():
                # Verificar tipo antes de usar len()
                try:
                    num_reports = count_reports_safely(reports)
                except Exception as e:
                    logging.error(f"Error al contar reportes para {instrument}: {str(e)}")
        
        # Mencionar gráficos en escala de precios si fueron generados
        if args.plot_price_scale and 'plots' in globals():
        
        return 0
    
    except Exception as e:
        logging.error(f"Error en el procesamiento: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
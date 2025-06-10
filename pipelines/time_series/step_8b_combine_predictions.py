#!/usr/bin/env python
# coding: utf-8

"""
Script para combinar predicciones de los pipelines ML y ARIMA/SARIMAX,
creando un ensemble ponderado y unificando el formato para PowerBI.

Este script:
1. Lee los archivos de predicciones de ambos pipelines
2. Alinea las fechas y columnas para una correcta integración
3. Crea un ensemble ponderado basado en el rendimiento histórico o pesos definidos
4. Genera intervalos de confianza para las predicciones combinadas
5. Unifica el formato para facilitar la visualización en PowerBI
6. Almacena las predicciones combinadas preservando los modelos individuales

Autor: [Tu Nombre]
Fecha: [Fecha Actual]
"""

import os
import json
import logging
from sp500_analysis.config.settings import settings
from sp500_analysis.shared.logging.logger import configurar_logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Configuración de logging
log_file = os.path.join(settings.log_dir, f"combine_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
configurar_logging(log_file)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Combina predicciones de pipelines ML y ARIMA')
    
    parser.add_argument('--ml_predictions_file', type=str, 
                        default='data/predictions/all_models_predictions.csv',
                        help='Ruta al archivo con predicciones del pipeline ML.')
    
    parser.add_argument('--arima_predictions_file', type=str, 
                        default='data/predictions/arima_for_powerbi.csv',
                        help='Ruta al archivo con predicciones del pipeline ARIMA.')
    
    parser.add_argument('--output_file', type=str, 
                        default='data/predictions/combined_predictions_for_powerbi.csv',
                        help='Ruta al archivo de salida con predicciones combinadas.')
    
    parser.add_argument('--target_col', type=str, 
                        required=True,
                        help='Nombre de la columna objetivo (target).')
    
    parser.add_argument('--ensemble_method', type=str, 
                        default='weighted',
                        choices=['weighted', 'average', 'best', 'custom'],
                        help='Método para combinar las predicciones.')
    
    parser.add_argument('--ml_weight', type=float, 
                        default=0.5,
                        help='Peso para las predicciones ML (0-1) en método weighted o custom.')
    
    parser.add_argument('--weights_file', type=str, 
                        default=None,
                        help='Archivo JSON con pesos personalizados para cada modelo.')
    
    parser.add_argument('--calc_weights_from_valid', action='store_true',
                        help='Calcular pesos basados en el rendimiento en validación.')
    
    parser.add_argument('--validation_periods', type=int, 
                        default=21,
                        help='Número de períodos para evaluar rendimiento y calcular pesos.')
    
    parser.add_argument('--include_all_models', action='store_true',
                        help='Incluir todos los modelos individuales en el archivo de salida.')
    
    parser.add_argument('--ensemble_name', type=str, 
                        default='Ensemble',
                        help='Nombre para el modelo ensamblado.')
    
    parser.add_argument('--date_column', type=str, 
                        default='date',
                        help='Nombre de la columna de fecha en los archivos de entrada.')
    
    parser.add_argument('--drop_duplicates', action='store_true',
                        help='Eliminar predicciones duplicadas para la misma fecha.')
    
    return parser.parse_args()

def read_predictions_file(file_path, date_column='date', required=True):
    """
    Lee un archivo de predicciones y estandariza su formato.
    
    Args:
        file_path: Ruta al archivo de predicciones
        date_column: Nombre de la columna de fecha
        required: Si el archivo es obligatorio o no
        
    Returns:
        DataFrame con predicciones o None si hay error o el archivo no existe
    """
    if not file_path:
        if required:
            logger.error("No se especificó archivo de predicciones.")
            return None
        else:
            logger.info("No se especificó archivo opcional de predicciones.")
            return None
    
    if not os.path.exists(file_path):
        if required:
            logger.error(f"El archivo de predicciones no existe: {file_path}")
            return None
        else:
            logger.info(f"El archivo opcional de predicciones no existe: {file_path}")
            return None
    
    try:
        # Determinar extensión del archivo
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            logger.error(f"Formato de archivo no soportado: {file_ext}")
            return None
        
        logger.info(f"Archivo de predicciones leído exitosamente: {file_path}")
        logger.info(f"Dimensiones: {df.shape}, Columnas: {df.columns.tolist()}")
        
        # Estandarizar la columna de fecha
        if date_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column])
                logger.info(f"Columna {date_column} convertida a datetime")
            
            # Establecer como índice
            df = df.set_index(date_column)
            logger.info(f"Columna {date_column} establecida como índice")
        
        return df
    
    except Exception as e:
        logger.error(f"Error al leer archivo de predicciones {file_path}: {str(e)}")
        return None

def identify_forecast_columns(df, target_col, model_type=None):
    """
    Identifica las columnas de predicción, límites inferior y superior.
    
    Args:
        df: DataFrame con predicciones
        target_col: Nombre base de la columna objetivo
        model_type: Tipo de modelo (si se conoce)
        
    Returns:
        Dict con nombres de columnas identificadas
    """
    # Inicializar diccionario de columnas
    column_info = {
        'forecast': None,
        'lower': None,
        'upper': None,
        'actual': None,
        'model_type': model_type
    }
    
    # Buscar columnas por patrones
    forecast_patterns = ['forecast', 'pred', 'prediction']
    lower_patterns = ['lower', 'lo', 'min', 'bottom']
    upper_patterns = ['upper', 'hi', 'max', 'top']
    actual_patterns = ['actual', 'real', 'observed', 'true']
    
    # Identificar columna de modelo (si existe)
    model_col = None
    for col in df.columns:
        if col.lower() in ['model', 'modeltype', 'model_type']:
            model_col = col
            break
    
    # Obtener tipos de modelo únicos (si hay una columna de modelo)
    model_types = []
    if model_col and model_col in df.columns:
        model_types = df[model_col].unique().tolist()
        column_info['model_column'] = model_col
    
    # Caso 1: Tenemos tipos de modelo específicos
    if model_types:
        # Filtrar para el modelo específico si se proporcionó
        if model_type and model_type in model_types:
            filtered_df = df[df[model_col] == model_type]
        else:
            # Usar el primer tipo de modelo
            model_type = model_types[0]
            filtered_df = df[df[model_col] == model_type]
            column_info['model_type'] = model_type
        
        # Buscar columnas en el DataFrame filtrado
        for col in filtered_df.columns:
            col_lower = col.lower()
            
            # Buscar por patrón en nombre y valor de target
            target_pattern = target_col.lower()
            
            # Identificar columna de predicción
            if any(pattern in col_lower for pattern in forecast_patterns) or f"{target_pattern}_" in col_lower:
                if column_info['forecast'] is None or len(col) < len(column_info['forecast']):
                    column_info['forecast'] = col
            
            # Identificar límite inferior
            if any(pattern in col_lower for pattern in lower_patterns):
                column_info['lower'] = col
            
            # Identificar límite superior
            if any(pattern in col_lower for pattern in upper_patterns):
                column_info['upper'] = col
            
            # Identificar valor real
            if any(pattern in col_lower for pattern in actual_patterns) or col_lower == target_col.lower():
                column_info['actual'] = col
    
    # Caso 2: No tenemos tipos de modelo, buscar en todo el DataFrame
    else:
        for col in df.columns:
            col_lower = col.lower()
            
            # Buscar por patrón en nombre y valor de target
            target_pattern = target_col.lower()
            
            # Identificar columna de predicción
            if any(pattern in col_lower for pattern in forecast_patterns) or f"{target_pattern}_" in col_lower:
                if 'arima' in col_lower and model_type == 'ARIMA':
                    column_info['forecast'] = col
                elif 'hybrid' in col_lower and model_type == 'Hybrid':
                    column_info['forecast'] = col
                elif 'ml' in col_lower and model_type == 'ML':
                    column_info['forecast'] = col
                elif column_info['forecast'] is None:
                    column_info['forecast'] = col
            
            # Identificar límite inferior
            if any(pattern in col_lower for pattern in lower_patterns):
                if 'arima' in col_lower and model_type == 'ARIMA':
                    column_info['lower'] = col
                elif 'hybrid' in col_lower and model_type == 'Hybrid':
                    column_info['lower'] = col
                elif 'ml' in col_lower and model_type == 'ML':
                    column_info['lower'] = col
                elif column_info['lower'] is None:
                    column_info['lower'] = col
            
            # Identificar límite superior
            if any(pattern in col_lower for pattern in upper_patterns):
                if 'arima' in col_lower and model_type == 'ARIMA':
                    column_info['upper'] = col
                elif 'hybrid' in col_lower and model_type == 'Hybrid':
                    column_info['upper'] = col
                elif 'ml' in col_lower and model_type == 'ML':
                    column_info['upper'] = col
                elif column_info['upper'] is None:
                    column_info['upper'] = col
            
            # Identificar valor real
            if any(pattern in col_lower for pattern in actual_patterns) or col_lower == target_col.lower():
                column_info['actual'] = col
    
    # Log de las columnas identificadas
    logger.info(f"Columnas identificadas para {model_type if model_type else 'predicciones'}:")
    for key, value in column_info.items():
        if key != 'model_column':
            logger.info(f"  {key}: {value}")
    
    return column_info

def extract_model_predictions(df, target_col, model_types=None):
    """
    Extrae las predicciones por modelo del DataFrame.
    
    Args:
        df: DataFrame con predicciones
        target_col: Nombre base de la columna objetivo
        model_types: Lista de tipos de modelo a extraer
        
    Returns:
        Dict con DataFrames por modelo
    """
    # Inicializar diccionario de predicciones
    predictions = {}
    
    # Si no se especifican tipos de modelo, intentar identificarlos
    if not model_types:
        # Buscar columna de tipo de modelo
        model_col = None
        for col in df.columns:
            if col.lower() in ['model', 'modeltype', 'model_type']:
                model_col = col
                break
        
        if model_col:
            model_types = df[model_col].unique().tolist()
            logger.info(f"Tipos de modelo identificados: {model_types}")
        else:
            # Buscar patrones en nombres de columnas
            arima_pattern = any('arima' in col.lower() for col in df.columns)
            hybrid_pattern = any('hybrid' in col.lower() for col in df.columns)
            ml_pattern = any(('ml' in col.lower() or 'xgboost' in col.lower() or 'lightgbm' in col.lower()) 
                           for col in df.columns)
            
            model_types = []
            if arima_pattern:
                model_types.append('ARIMA')
            if hybrid_pattern:
                model_types.append('Hybrid')
            if ml_pattern:
                model_types.append('ML')
            
            if not model_types:
                model_types = ['Unknown']
            
            logger.info(f"Tipos de modelo inferidos: {model_types}")
    
    # Extraer predicciones por modelo
    for model_type in model_types:
        # Identificar columnas para este modelo
        column_info = identify_forecast_columns(df, target_col, model_type)
        
        if column_info['forecast'] is None:
            logger.warning(f"No se encontró columna de predicción para modelo {model_type}")
            continue
        
        # Extraer predicciones
        model_df = pd.DataFrame(index=df.index)
        
        # Añadir columnas relevantes
        if column_info['forecast']:
            model_df['forecast'] = df[column_info['forecast']]
        
        if column_info['lower']:
            model_df['lower'] = df[column_info['lower']]
        
        if column_info['upper']:
            model_df['upper'] = df[column_info['upper']]
        
        if column_info['actual']:
            model_df['actual'] = df[column_info['actual']]
        
        # Añadir tipo de modelo
        model_df['model_type'] = model_type
        
        # Guardar en diccionario
        predictions[model_type] = model_df
        logger.info(f"Extraídas predicciones para modelo {model_type}: {len(model_df)} filas")
    
    return predictions

def align_predictions(ml_predictions, arima_predictions, target_col):
    """
    Alinea las predicciones de ML y ARIMA por fecha.
    
    Args:
        ml_predictions: DataFrame con predicciones ML
        arima_predictions: DataFrame con predicciones ARIMA
        target_col: Nombre base de la columna objetivo
        
    Returns:
        Dict con DataFrames de predicciones alineadas por modelo
    """
    # Extraer predicciones por modelo
    ml_models = extract_model_predictions(ml_predictions, target_col)
    arima_models = extract_model_predictions(arima_predictions, target_col)
    
    # Combinar diccionarios
    all_models = {**ml_models, **arima_models}
    
    # Obtener fechas comunes
    common_dates = None
    for model_type, model_df in all_models.items():
        if common_dates is None:
            common_dates = set(model_df.index)
        else:
            common_dates &= set(model_df.index)
    
    # Si no hay fechas comunes, usar todas las fechas
    if not common_dates or len(common_dates) == 0:
        logger.warning("No hay fechas comunes entre modelos. Usando todas las fechas.")
        all_dates = set()
        for model_type, model_df in all_models.items():
            all_dates |= set(model_df.index)
        common_dates = all_dates
    
    # Alinear predicciones a fechas comunes
    aligned_models = {}
    for model_type, model_df in all_models.items():
        # Filtrar a fechas comunes
        aligned_df = model_df.loc[model_df.index.isin(common_dates)].copy()
        
        # Ordenar por fecha
        aligned_df = aligned_df.sort_index()
        
        # Guardar en diccionario
        aligned_models[model_type] = aligned_df
        logger.info(f"Predicciones alineadas para modelo {model_type}: {len(aligned_df)} filas")
    
    return aligned_models

def calculate_weights_from_validation(aligned_models, validation_periods=21):
    """
    Calcula pesos para cada modelo basados en su rendimiento en validación.
    
    Args:
        aligned_models: Dict con DataFrames de predicciones alineadas por modelo
        validation_periods: Número de períodos para evaluar rendimiento
        
    Returns:
        Dict con pesos por modelo
    """
    # Inicializar diccionario de pesos
    weights = {}
    
    # Verificar que haya suficientes datos para validación
    for model_type, model_df in aligned_models.items():
        if len(model_df) <= validation_periods:
            logger.warning(f"Datos insuficientes para modelo {model_type}. Usando pesos iguales.")
            return {model_type: 1.0 / len(aligned_models) for model_type in aligned_models}
    
    # Calcular errores para cada modelo en período de validación
    errors = {}
    for model_type, model_df in aligned_models.items():
        # Verificar que tengamos columnas necesarias
        if 'forecast' not in model_df.columns or 'actual' not in model_df.columns:
            logger.warning(f"Faltan columnas necesarias para modelo {model_type}. Omitiendo.")
            continue
        
        # Usar últimos períodos para validación
        validation_df = model_df.iloc[-validation_periods:].copy()
        
        # Eliminar filas con NaN en actual o forecast
        validation_df = validation_df.dropna(subset=['actual', 'forecast'])
        
        if len(validation_df) == 0:
            logger.warning(f"No hay datos válidos para validación en modelo {model_type}. Omitiendo.")
            continue
        
        # Calcular errores
        mae = mean_absolute_error(validation_df['actual'], validation_df['forecast'])
        rmse = np.sqrt(mean_squared_error(validation_df['actual'], validation_df['forecast']))
        
        # Guardar errores
        errors[model_type] = {
            'mae': mae,
            'rmse': rmse
        }
        
        logger.info(f"Errores para modelo {model_type}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    # Calcular pesos (inverso del error)
    # Usamos 1/RMSE como peso
    weights = {}
    total_weight = 0
    
    for model_type, error in errors.items():
        weights[model_type] = 1.0 / error['rmse']
        total_weight += weights[model_type]
    
    # Normalizar pesos
    if total_weight > 0:
        for model_type in weights:
            weights[model_type] /= total_weight
    else:
        # Si no se pudieron calcular pesos, usar iguales
        for model_type in errors:
            weights[model_type] = 1.0 / len(errors)
    
    logger.info("Pesos calculados a partir de validación:")
    for model_type, weight in weights.items():
        logger.info(f"  {model_type}: {weight:.4f}")
    
    return weights

def read_weights_from_file(weights_file):
    """
    Lee pesos personalizados desde un archivo JSON.
    
    Args:
        weights_file: Ruta al archivo JSON con pesos
        
    Returns:
        Dict con pesos por modelo o None si hay error
    """
    if not weights_file or not os.path.exists(weights_file):
        logger.error(f"El archivo de pesos no existe: {weights_file}")
        return None
    
    try:
        with open(weights_file, 'r') as f:
            weights = json.load(f)
        
        logger.info(f"Pesos leídos desde archivo: {weights_file}")
        
        # Validar estructura de pesos
        if not isinstance(weights, dict):
            logger.error("El archivo de pesos debe contener un diccionario.")
            return None
        
        # Normalizar pesos
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            logger.warning(f"Los pesos no suman 1.0. Normalizando.")
            for model_type in weights:
                weights[model_type] /= total_weight
        
        logger.info("Pesos normalizados:")
        for model_type, weight in weights.items():
            logger.info(f"  {model_type}: {weight:.4f}")
        
        return weights
    
    except Exception as e:
        logger.error(f"Error al leer archivo de pesos: {str(e)}")
        return None

def calculate_ensemble_predictions(aligned_models, weights=None, method='weighted', ml_weight=0.5):
    """
    Calcula predicciones de ensemble basadas en el método especificado.
    
    Args:
        aligned_models: Dict con DataFrames de predicciones alineadas por modelo
        weights: Dict con pesos por modelo (para método weighted o custom)
        method: Método de ensemble (weighted, average, best, custom)
        ml_weight: Peso para modelos ML en método weighted
        
    Returns:
        DataFrame con predicciones de ensemble
    """
    # Verificar que haya modelos para combinar
    if not aligned_models or len(aligned_models) == 0:
        logger.error("No hay modelos para combinar.")
        return None
    
    if len(aligned_models) == 1:
        logger.warning("Solo hay un modelo. No se requiere ensemble.")
        model_type = next(iter(aligned_models))
        ensemble_df = aligned_models[model_type].copy()
        ensemble_df['model_type'] = 'Ensemble'
        return ensemble_df
    
    # Preparar DataFrame para ensemble
    # Usar fechas del primer modelo
    first_model = next(iter(aligned_models.values()))
    ensemble_df = pd.DataFrame(index=first_model.index)
    
    # Añadir columna actual si existe
    if 'actual' in first_model.columns:
        ensemble_df['actual'] = first_model['actual']
    
    # Método 1: Promedio ponderado por pesos
    if method == 'weighted':
        # Si no se proporcionaron pesos, crear por defecto
        if not weights:
            weights = {}
            # Dividir entre modelos ML y ARIMA
            ml_models = [m for m in aligned_models if m.lower() in ['ml', 'xgboost', 'lightgbm']]
            arima_models = [m for m in aligned_models if m.lower() in ['arima', 'sarimax', 'hybrid']]
            
            # Asignar pesos
            if ml_models and arima_models:
                # Dividir peso ML entre modelos ML
                ml_weight_each = ml_weight / len(ml_models)
                arima_weight_each = (1.0 - ml_weight) / len(arima_models)
                
                for model in ml_models:
                    weights[model] = ml_weight_each
                for model in arima_models:
                    weights[model] = arima_weight_each
            else:
                # Pesos iguales si no hay clara distinción
                for model in aligned_models:
                    weights[model] = 1.0 / len(aligned_models)
        
        # Crear predicción ponderada
        forecast_values = np.zeros(len(ensemble_df))
        lower_values = np.zeros(len(ensemble_df))
        upper_values = np.zeros(len(ensemble_df))
        
        for model_type, model_df in aligned_models.items():
            # Obtener peso para este modelo
            weight = weights.get(model_type, 1.0 / len(aligned_models))
            
            # Añadir contribución ponderada
            if 'forecast' in model_df.columns:
                forecast_values += model_df['forecast'].fillna(0).values * weight
            
            if 'lower' in model_df.columns:
                lower_values += model_df['lower'].fillna(0).values * weight
            
            if 'upper' in model_df.columns:
                upper_values += model_df['upper'].fillna(0).values * weight
        
        ensemble_df['forecast'] = forecast_values
        ensemble_df['lower'] = lower_values
        ensemble_df['upper'] = upper_values
        
        logger.info(f"Ensemble creado usando método weighted con {len(weights)} modelos")
    
    # Método 2: Promedio simple
    elif method == 'average':
        forecast_values = np.zeros(len(ensemble_df))
        lower_values = np.zeros(len(ensemble_df))
        upper_values = np.zeros(len(ensemble_df))
        
        # Contar modelos con cada tipo de valor
        forecast_count = 0
        lower_count = 0
        upper_count = 0
        
        for model_type, model_df in aligned_models.items():
            if 'forecast' in model_df.columns:
                forecast_values += model_df['forecast'].fillna(0).values
                forecast_count += 1
            
            if 'lower' in model_df.columns:
                lower_values += model_df['lower'].fillna(0).values
                lower_count += 1
            
            if 'upper' in model_df.columns:
                upper_values += model_df['upper'].fillna(0).values
                upper_count += 1
        
        if forecast_count > 0:
            ensemble_df['forecast'] = forecast_values / forecast_count
        
        if lower_count > 0:
            ensemble_df['lower'] = lower_values / lower_count
        
        if upper_count > 0:
            ensemble_df['upper'] = upper_values / upper_count
        
        logger.info(f"Ensemble creado usando método average con {len(aligned_models)} modelos")
    
    # Método 3: Mejor modelo
    elif method == 'best':
        # Si no se proporcionaron pesos, calcular basados en validación
        if not weights:
            # Usar últimos 21 días como validación
            weights = calculate_weights_from_validation(aligned_models, 21)
        
        # Seleccionar mejor modelo
        best_model = max(weights.items(), key=lambda x: x[1])[0]
        logger.info(f"Mejor modelo seleccionado: {best_model} con peso {weights[best_model]:.4f}")
        
        # Usar predicciones del mejor modelo
        ensemble_df = aligned_models[best_model].copy()
        
        # Renombrar tipo de modelo
        ensemble_df['model_type'] = f"Ensemble (Best: {best_model})"
    
    # Método 4: Personalizado (igual que weighted pero con pesos personalizados)
    elif method == 'custom':
        # Verificar que hay pesos
        if not weights:
            logger.warning("No se proporcionaron pesos personalizados. Usando pesos iguales.")
            weights = {model_type: 1.0 / len(aligned_models) for model_type in aligned_models}
        
        # Crear predicción ponderada
        forecast_values = np.zeros(len(ensemble_df))
        lower_values = np.zeros(len(ensemble_df))
        upper_values = np.zeros(len(ensemble_df))
        
        for model_type, model_df in aligned_models.items():
            # Obtener peso para este modelo
            weight = weights.get(model_type, 1.0 / len(aligned_models))
            
            # Añadir contribución ponderada
            if 'forecast' in model_df.columns:
                forecast_values += model_df['forecast'].fillna(0).values * weight
            
            if 'lower' in model_df.columns:
                lower_values += model_df['lower'].fillna(0).values * weight
            
            if 'upper' in model_df.columns:
                upper_values += model_df['upper'].fillna(0).values * weight
        
        ensemble_df['forecast'] = forecast_values
        ensemble_df['lower'] = lower_values
        ensemble_df['upper'] = upper_values
        
        logger.info(f"Ensemble creado usando método custom con {len(weights)} modelos y pesos personalizados")
    
    # Añadir tipo de modelo
    ensemble_df['model_type'] = 'Ensemble'
    
    return ensemble_df

def format_for_powerbi(models_dict, ensemble_df, target_col, include_all_models=True, ensemble_name='Ensemble'):
    """
    Formatea las predicciones para PowerBI.
    
    Args:
        models_dict: Dict con DataFrames de predicciones por modelo
        ensemble_df: DataFrame con predicciones de ensemble
        target_col: Nombre base de la columna objetivo
        include_all_models: Si se deben incluir todos los modelos en la salida
        ensemble_name: Nombre para el modelo ensamblado
        
    Returns:
        DataFrame formateado para PowerBI
    """
    # Crear lista de DataFrames a combinar
    dfs_to_combine = []
    
    # Añadir ensemble
    if ensemble_df is not None:
        ensemble_copy = ensemble_df.copy()
        ensemble_copy = ensemble_copy.reset_index()
        
        # Renombrar columnas
        column_mapping = {
            'forecast': f'{target_col}_{ensemble_name}_Forecast',
            'lower': f'{target_col}_{ensemble_name}_Lower',
            'upper': f'{target_col}_{ensemble_name}_Upper',
            'actual': f'{target_col}_Actual',
            'model_type': 'ModelType',
            'index': 'date'
        }
        
        ensemble_copy = ensemble_copy.rename(columns={k: v for k, v in column_mapping.items() 
                                                 if k in ensemble_copy.columns})
        
        dfs_to_combine.append(ensemble_copy)
    
    # Añadir modelos individuales si se solicita
    if include_all_models:
        for model_type, model_df in models_dict.items():
            model_copy = model_df.copy()
            model_copy = model_copy.reset_index()
            
            # Renombrar columnas
            column_mapping = {
                'forecast': f'{target_col}_{model_type}_Forecast',
                'lower': f'{target_col}_{model_type}_Lower',
                'upper': f'{target_col}_{model_type}_Upper',
                'actual': f'{target_col}_Actual',
                'model_type': 'ModelType',
                'index': 'date'
            }
            
            model_copy = model_copy.rename(columns={k: v for k, v in column_mapping.items() 
                                                if k in model_copy.columns})
            
            dfs_to_combine.append(model_copy)
    
    # Combinar DataFrames
    if not dfs_to_combine:
        logger.error("No hay DataFrames para combinar.")
        return None
    
    # Unir por fecha y preservar columnas de todos los modelos
    final_df = pd.concat(dfs_to_combine, ignore_index=True, sort=False)
    
    # Añadir columna de tipo de dato si no existe
    if 'DataType' not in final_df.columns:
        # Intentar inferir
        if 'data_type' in final_df.columns:
            final_df['DataType'] = final_df['data_type']
        else:
            final_df['DataType'] = 'forecast'
    
    # Ordenar columnas: date, actual, forecast, intervalos
    column_order = ['date', f'{target_col}_Actual', 'DataType', 'ModelType']
    
    # Añadir columnas de forecast en orden de prioridad
    forecast_cols = [col for col in final_df.columns if 'Forecast' in col]
    forecast_cols.sort(key=lambda x: 0 if ensemble_name in x else 1)  # Poner ensemble primero
    column_order.extend(forecast_cols)
    
    # Añadir columnas de límites
    for col in forecast_cols:
        base = col.replace('Forecast', '')
        lower_col = f"{base}Lower"
        upper_col = f"{base}Upper"
        
        if lower_col in final_df.columns:
            column_order.append(lower_col)
        if upper_col in final_df.columns:
            column_order.append(upper_col)
    
    # Añadir otras columnas
    other_cols = [col for col in final_df.columns if col not in column_order]
    column_order.extend(other_cols)
    
    # Ordenar columnas
    final_df = final_df[column_order]
    
    return final_df

def combine_predictions(ml_predictions_file, arima_predictions_file, output_file, target_col,
                      ensemble_method='weighted', ml_weight=0.5, weights_file=None,
                      calc_weights_from_valid=False, validation_periods=21,
                      include_all_models=True, ensemble_name='Ensemble',
                      date_column='date', drop_duplicates=False):
    """
    Función principal para combinar predicciones de pipelines ML y ARIMA.
    
    Args:
        ml_predictions_file: Ruta al archivo con predicciones ML
        arima_predictions_file: Ruta al archivo con predicciones ARIMA
        output_file: Ruta al archivo de salida
        target_col: Nombre de la columna objetivo
        ensemble_method: Método para combinar predicciones
        ml_weight: Peso para predicciones ML
        weights_file: Archivo con pesos personalizados
        calc_weights_from_valid: Calcular pesos basados en validación
        validation_periods: Períodos para calcular pesos
        include_all_models: Incluir todos los modelos en salida
        ensemble_name: Nombre para el modelo ensamblado
        date_column: Nombre de la columna de fecha
        drop_duplicates: Eliminar predicciones duplicadas
        
    Returns:
        Boolean indicando éxito o fracaso
    """
    # Leer archivos de predicciones
    ml_df = read_predictions_file(ml_predictions_file, date_column, required=True)
    arima_df = read_predictions_file(arima_predictions_file, date_column, required=True)
    
    if ml_df is None or arima_df is None:
        logger.error("Error al leer archivos de predicciones.")
        return False
    
    # Alinear predicciones
    aligned_models = align_predictions(ml_df, arima_df, target_col)
    
    if not aligned_models:
        logger.error("Error al alinear predicciones.")
        return False
    
    # Determinar pesos para ensemble
    weights = None
    
    if weights_file:
        # Leer pesos desde archivo
        weights = read_weights_from_file(weights_file)
    
    if calc_weights_from_valid:
        # Calcular pesos basados en validación
        weights = calculate_weights_from_validation(aligned_models, validation_periods)
    
    # Calcular predicciones de ensemble
    ensemble_df = calculate_ensemble_predictions(aligned_models, weights, ensemble_method, ml_weight)
    
    if ensemble_df is None:
        logger.error("Error al calcular predicciones de ensemble.")
        return False
    
    # Formatear para PowerBI
    output_df = format_for_powerbi(aligned_models, ensemble_df, target_col, include_all_models, ensemble_name)
    
    if output_df is None:
        logger.error("Error al formatear para PowerBI.")
        return False
    
    # Eliminar duplicados si se solicita
    if drop_duplicates:
        n_before = len(output_df)
        output_df = output_df.drop_duplicates(subset=['date', 'ModelType'])
        n_after = len(output_df)
        
        if n_before > n_after:
            logger.info(f"Eliminados {n_before - n_after} duplicados.")
    
    # Guardar resultado
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Determinar formato de archivo
        file_ext = os.path.splitext(output_file)[1].lower()
        
        if file_ext == '.csv':
            output_df.to_csv(output_file, index=False)
        elif file_ext in ['.xlsx', '.xls']:
            output_df.to_excel(output_file, index=False)
        else:
            logger.warning(f"Formato de archivo no reconocido: {file_ext}. Guardando como CSV.")
            output_file = os.path.splitext(output_file)[0] + '.csv'
            output_df.to_csv(output_file, index=False)
        
        logger.info(f"Archivo de salida guardado exitosamente: {output_file}")
        logger.info(f"Dimensiones: {output_df.shape}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error al guardar archivo de salida: {str(e)}")
        return False

def main():
    """Función principal que orquesta el proceso completo."""
    # Obtener argumentos
    args = parse_arguments()
    
    logger.info("=== Iniciando combinación de predicciones ML y ARIMA ===")
    logger.info(f"Archivo de predicciones ML: {args.ml_predictions_file}")
    logger.info(f"Archivo de predicciones ARIMA: {args.arima_predictions_file}")
    logger.info(f"Archivo de salida: {args.output_file}")
    logger.info(f"Columna objetivo: {args.target_col}")
    logger.info(f"Método de ensemble: {args.ensemble_method}")
    
    if args.ensemble_method == 'weighted':
        logger.info(f"Peso ML: {args.ml_weight}")
    
    if args.weights_file:
        logger.info(f"Archivo de pesos: {args.weights_file}")
    
    if args.calc_weights_from_valid:
        logger.info(f"Calculando pesos a partir de validación con {args.validation_periods} períodos")
    
    # Ejecutar combinación de predicciones
    success = combine_predictions(
        args.ml_predictions_file,
        args.arima_predictions_file,
        args.output_file,
        args.target_col,
        args.ensemble_method,
        args.ml_weight,
        args.weights_file,
        args.calc_weights_from_valid,
        args.validation_periods,
        args.include_all_models,
        args.ensemble_name,
        args.date_column,
        args.drop_duplicates
    )
    
    if success:
        logger.info("=== Combinación de predicciones completada con éxito ===")
    else:
        logger.error("=== Error en combinación de predicciones ===")

if __name__ == "__main__":
    main()
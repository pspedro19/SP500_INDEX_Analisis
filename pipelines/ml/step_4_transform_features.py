#!/usr/bin/env python
# coding: utf-8

import re
import os
import time
import pandas as pd
import numpy as np
from ta.trend import ema_indicator, sma_indicator

# Definir la raíz del proyecto para rutas absolutas
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ================================
# CONSTANTES GLOBALES
# ================================
# Horizonte de predicción exactamente igual al pipeline de inferencia
FORECAST_HORIZON_1MONTH = 20  # 20 días hábiles
FORECAST_HORIZON_3MONTHS = 60  # 60 días hábiles
FORECAST_HORIZON = FORECAST_HORIZON_1MONTH  # Horizonte predeterminado
LOCAL_REFINEMENT_DAYS = 225  # Número de días para refinamiento local
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% training, 20% test en refinamiento local

# ================================
# HELPER FUNCTIONS DE PREPROCESAMIENTO
# ================================

def rename_dataframe(dataset, datetime_column, target_columns, date_format):
    renamed = dataset.rename(columns={datetime_column: 'date'})
    
    # Eliminar filas con fechas no válidas (como "Sin categoría")
    renamed = renamed[renamed['date'].apply(lambda x: isinstance(x, str) and not re.search(r'[a-zA-Z]', x))]

    try:
        if date_format:
            renamed['date'] = pd.to_datetime(renamed['date'], format=date_format).dt.strftime('%Y-%m-%d')
        else:
            renamed['date'] = pd.to_datetime(renamed['date'], infer_datetime_format=True).dt.strftime('%Y-%m-%d')
    except Exception as e:
        print("Error al convertir fecha:", e)

    first_col = dataset.columns[0]
    if first_col != datetime_column:
        renamed = renamed.rename(columns={first_col: 'id'})
    else:
        # Si no hay otra columna que renombrar como 'id', creamos una por defecto
        renamed.insert(1, 'id', 'serie_default')

    cols_to_keep = ['date', 'id']
    if target_columns:
        cols_to_keep.extend(target_columns.split(","))

    return renamed[cols_to_keep]


def impute_time_series_ffill(dataset, datetime_column='date', id_column='id'):
    df = dataset.copy()
    if df.isnull().values.any():
        df = df.ffill()
    return df

def resample_to_business_day(dataset, input_frequency, column_date='date', id_column='id', output_frequency='B'):
    df = dataset.copy()
    df[column_date] = pd.to_datetime(df[column_date], format='%Y-%m-%d')
    df = df.drop_duplicates(subset=[column_date], keep='last')
    df = df.set_index(column_date).sort_index()
    if input_frequency != output_frequency and len(df) > 0:
        resampled = df.asfreq(freq=output_frequency, method='ffill')
    else:
        resampled = df.copy()
    resampled = resampled.reset_index()
    resampled[column_date] = resampled[column_date].dt.strftime('%Y-%m-%d')
    return resampled

def convert_dataframe(df, excluded_column, id_column='id', datetime_column='date'):
    cols = df.columns.difference([id_column, datetime_column, excluded_column])
    for col in cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '.', regex=False)
            .str.replace(r'[^\d\.-]', '', regex=True)  # limpia texto no numérico
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')  # convierte a float y deja NaN si no se puede
    return df


# ================================
# NUEVAS FUNCIONES PARA FEATURES ADICIONALES
# ================================

def add_log_and_diff_features(df, target_column):
    # Logaritmo (se suma 1 para evitar problemas con ceros) y diferencia logarítmica (retornos continuos)
    df['log_' + target_column] = np.log(df[target_column] + 1)
    df['log_diff_' + target_column] = df['log_' + target_column] - df['log_' + target_column].shift(1)
    return df

def add_volatility_features(df, target_column, window=FORECAST_HORIZON_1MONTH):
    # Desviación estándar y varianza móvil; Bollinger Bands con k=2
    df['rolling_std_' + target_column] = df[target_column].rolling(window).std()
    df['rolling_var_' + target_column] = df[target_column].rolling(window).var()
    ma = df[target_column].rolling(window).mean()
    std = df[target_column].rolling(window).std()
    df['bollinger_upper_' + target_column] = ma + 2 * std
    df['bollinger_lower_' + target_column] = ma - 2 * std
    return df

def add_rsi(df, target_column, window=14):
    # Cálculo básico del Relative Strength Index (RSI)
    delta = df[target_column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    df['RSI_' + target_column] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, target_column, fast=12, slow=26, signal=9):
    # Calcula la diferencia entre dos EMAs y su señal
    ema_fast = df[target_column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[target_column].ewm(span=slow, adjust=False).mean()
    df['MACD_' + target_column] = ema_fast - ema_slow
    df['MACD_signal_' + target_column] = df['MACD_' + target_column].ewm(span=signal, adjust=False).mean()
    return df

def add_momentum(df, target_column, window=10):
    # Impulso: diferencia entre el valor actual y el valor n períodos atrás
    df['momentum_' + target_column] = df[target_column] - df[target_column].shift(window)
    return df

def add_intermediate_changes(df, target_column):
    # Cambios a 3 meses (60 días hábiles) y 6 meses (120 días hábiles)
    # Usar FORECAST_HORIZON_3MONTHS para 3 meses
    df['3M_change_' + target_column] = (df[target_column] / df[target_column].shift(FORECAST_HORIZON_3MONTHS)) - 1
    df['6M_change_' + target_column] = (df[target_column] / df[target_column].shift(FORECAST_HORIZON_3MONTHS * 2)) - 1
    return df

def add_ytd_performance(df, target_column):
    # Year-to-Date: rendimiento desde el primer día del año
    df['year'] = pd.to_datetime(df['date']).dt.year
    first_day = df.groupby('year')[target_column].transform('first')
    df['YTD_' + target_column] = (df[target_column] / first_day) - 1
    df.drop(columns=['year'], inplace=True)
    return df

def add_zscore(df, target_column, window=60):
    rolling_mean = df[target_column].rolling(window).mean()
    rolling_std = df[target_column].rolling(window).std()

    # Evita divisiones por cero o valores extremadamente pequeños
    rolling_std_safe = rolling_std.replace(0, np.nan)
    df['zscore_' + target_column] = (df[target_column] - rolling_mean) / rolling_std_safe

    return df


def add_minmax_scaling(df, target_column):
    # Escalado MinMax a [0,1] de la serie completa
    min_val = df[target_column].min()
    max_val = df[target_column].max()
    df['minmax_' + target_column] = (df[target_column] - min_val) / (max_val - min_val)
    return df

# ================================
# TRANSFORMACIONES PARA CADA CATEGORÍA (1MONTH)
# Estas funciones ya aplican transformaciones básicas (MoM, YoY, medias móviles y exponenciales)
# y ahora se enriquecen con los indicadores adicionales según la naturaleza de la serie.
# ================================

def transform_bond(df, target_column, id_column='id'):
    # Usar FORECAST_HORIZON para consistencia
    df['MoM_' + target_column] = ((df[target_column] / df[target_column].shift(FORECAST_HORIZON_1MONTH)) - 1).astype(float)
    df['YoY_' + target_column] = ((df[target_column] / df[target_column].shift(252)) - 1).astype(float)
    df['MA_60_' + target_column] = sma_indicator(df[target_column], 60).astype(float)
    df['EMA_60_' + target_column] = ema_indicator(df[target_column], 60).astype(float)
    # Features adicionales
    df = add_log_and_diff_features(df, target_column)
    df = add_volatility_features(df, target_column, window=FORECAST_HORIZON_1MONTH)
    df = add_intermediate_changes(df, target_column)
    df = add_ytd_performance(df, target_column)
    df = add_zscore(df, target_column, window=60)
    df = add_minmax_scaling(df, target_column)
    return df

def transform_business_confidence(df, target_column, id_column='id'):
    df['MoM_' + target_column] = ((df[target_column] / df[target_column].shift(FORECAST_HORIZON_1MONTH)) - 1).astype(float)
    df['YoY_' + target_column] = ((df[target_column] / df[target_column].shift(252)) - 1).astype(float)
    df['MA_20_' + target_column] = sma_indicator(df[target_column], FORECAST_HORIZON_1MONTH).astype(float)
    df['EMA_20_' + target_column] = ema_indicator(df[target_column], FORECAST_HORIZON_1MONTH).astype(float)
    # Features adicionales
    df = add_log_and_diff_features(df, target_column)
    df = add_volatility_features(df, target_column, window=FORECAST_HORIZON_1MONTH)
    df = add_intermediate_changes(df, target_column)
    df = add_ytd_performance(df, target_column)
    df = add_zscore(df, target_column, window=60)
    df = add_minmax_scaling(df, target_column)
    return df

def transform_car_registrations(df, target_column, id_column='id'):
    df['MoM_' + target_column] = ((df[target_column] / df[target_column].shift(FORECAST_HORIZON_1MONTH)) - 1).astype(float)
    df['YoY_' + target_column] = ((df[target_column] / df[target_column].shift(252)) - 1).astype(float)
    df['Abs_Change_' + target_column] = (df[target_column] - df[target_column].shift(1)).astype(float)
    df['MA_60_' + target_column] = sma_indicator(df[target_column], 60).astype(float)
    df['EMA_60_' + target_column] = ema_indicator(df[target_column], 60).astype(float)
    # Features adicionales (útiles para series con fuerte estacionalidad y patrones locales)
    df = add_log_and_diff_features(df, target_column)
    df = add_intermediate_changes(df, target_column)
    df = add_zscore(df, target_column, window=60)
    df = add_minmax_scaling(df, target_column)
    return df

def transform_comm_loans(df, target_column, id_column='id'):
    df['MoM_' + target_column] = ((df[target_column] / df[target_column].shift(FORECAST_HORIZON_1MONTH)) - 1).astype(float)
    df['MA_60_' + target_column] = sma_indicator(df[target_column], 60).astype(float)
    # Features adicionales
    df = add_log_and_diff_features(df, target_column)
    df = add_volatility_features(df, target_column, window=FORECAST_HORIZON_1MONTH)
    df = add_intermediate_changes(df, target_column)
    df = add_zscore(df, target_column, window=60)
    return df

def transform_commodities(df, target_column, id_column='id'):
    df['EMA_5_' + target_column] = ema_indicator(df[target_column], 5).astype(float)
    df['EMA_10_' + target_column] = ema_indicator(df[target_column], 10).astype(float)
    df['EMA_20_' + target_column] = ema_indicator(df[target_column], FORECAST_HORIZON_1MONTH).astype(float)
    df['MA_5_' + target_column] = sma_indicator(df[target_column], 5).astype(float)
    df['MA_10_' + target_column] = sma_indicator(df[target_column], 10).astype(float)
    df['MA_20_' + target_column] = sma_indicator(df[target_column], FORECAST_HORIZON_1MONTH).astype(float)
    # Features adicionales para series financieras
    df = add_log_and_diff_features(df, target_column)
    df = add_volatility_features(df, target_column, window=FORECAST_HORIZON_1MONTH)
    df = add_rsi(df, target_column, window=14)
    df = add_macd(df, target_column, fast=12, slow=26, signal=9)
    df = add_momentum(df, target_column, window=10)
    df = add_intermediate_changes(df, target_column)
    df = add_ytd_performance(df, target_column)
    df = add_zscore(df, target_column, window=60)
    df = add_minmax_scaling(df, target_column)
    return df

def transform_consumer_confidence(df, target_column, id_column='id'):
    df['YoY_' + target_column] = ((df[target_column] / df[target_column].shift(252)) - 1).astype(float)
    df['MoM_' + target_column] = ((df[target_column] / df[target_column].shift(FORECAST_HORIZON_1MONTH)) - 1).astype(float)
    df['MA_60_' + target_column] = sma_indicator(df[target_column], 60).astype(float)
    df['EMA_60_' + target_column] = ema_indicator(df[target_column], 60).astype(float)
    # Features adicionales
    df = add_log_and_diff_features(df, target_column)
    df = add_volatility_features(df, target_column, window=FORECAST_HORIZON_1MONTH)
    df = add_intermediate_changes(df, target_column)
    df = add_ytd_performance(df, target_column)
    df = add_zscore(df, target_column, window=60)
    df = add_minmax_scaling(df, target_column)
    return df

def transform_economics(df, target_column, id_column='id'):
    df['MoM_' + target_column] = ((df[target_column] / df[target_column].shift(FORECAST_HORIZON_1MONTH)) - 1).astype(float)
    df['YoY_' + target_column] = ((df[target_column] / df[target_column].shift(252)) - 1).astype(float)
    df['MA_60_' + target_column] = sma_indicator(df[target_column], 60).astype(float)
    df['EMA_60_' + target_column] = ema_indicator(df[target_column], 60).astype(float)
    # Features adicionales
    df = add_log_and_diff_features(df, target_column)
    df = add_intermediate_changes(df, target_column)
    df = add_ytd_performance(df, target_column)
    df = add_zscore(df, target_column, window=60)
    df = add_minmax_scaling(df, target_column)
    return df

def transform_exchange_rate(df, target_column, id_column='id'):
    df['EMA_5_' + target_column] = ema_indicator(df[target_column], 5).astype(float)
    df['EMA_10_' + target_column] = ema_indicator(df[target_column], 10).astype(float)
    df['EMA_20_' + target_column] = ema_indicator(df[target_column], FORECAST_HORIZON_1MONTH).astype(float)
    df['MA_5_' + target_column] = sma_indicator(df[target_column], 5).astype(float)
    df['MA_10_' + target_column] = sma_indicator(df[target_column], 10).astype(float)
    df['MA_20_' + target_column] = sma_indicator(df[target_column], FORECAST_HORIZON_1MONTH).astype(float)
    # Features adicionales para activos financieros
    df = add_log_and_diff_features(df, target_column)
    df = add_volatility_features(df, target_column, window=FORECAST_HORIZON_1MONTH)
    df = add_rsi(df, target_column, window=14)
    df = add_macd(df, target_column, fast=12, slow=26, signal=9)
    df = add_momentum(df, target_column, window=10)
    df = add_intermediate_changes(df, target_column)
    df = add_ytd_performance(df, target_column)
    df = add_zscore(df, target_column, window=60)
    df = add_minmax_scaling(df, target_column)
    return df

def transform_exports(df, target_column, id_column='id'):
    df['MoM_' + target_column] = ((df[target_column] / df[target_column].shift(FORECAST_HORIZON_1MONTH)) - 1).astype(float)
    df['YoY_' + target_column] = ((df[target_column] / df[target_column].shift(252)) - 1).astype(float)
    df['MA_60_' + target_column] = sma_indicator(df[target_column], 60).astype(float)
    df['EMA_60_' + target_column] = ema_indicator(df[target_column], 60).astype(float)
    # Features adicionales
    df = add_log_and_diff_features(df, target_column)
    df = add_intermediate_changes(df, target_column)
    df = add_ytd_performance(df, target_column)
    df = add_zscore(df, target_column, window=60)
    df = add_minmax_scaling(df, target_column)
    return df

def transform_index_pricing(df, target_column, id_column='id'):
    df['EMA_5_' + target_column] = ema_indicator(df[target_column], 5).astype(float)
    df['EMA_10_' + target_column] = ema_indicator(df[target_column], 10).astype(float)
    df['EMA_20_' + target_column] = ema_indicator(df[target_column], FORECAST_HORIZON_1MONTH).astype(float)
    df['MA_5_' + target_column] = sma_indicator(df[target_column], 5).astype(float)
    df['MA_10_' + target_column] = sma_indicator(df[target_column], 10).astype(float)
    df['MA_20_' + target_column] = sma_indicator(df[target_column], FORECAST_HORIZON_1MONTH).astype(float)
    # Features adicionales para series financieras
    df = add_log_and_diff_features(df, target_column)
    df = add_volatility_features(df, target_column, window=FORECAST_HORIZON_1MONTH)
    df = add_rsi(df, target_column, window=14)
    df = add_macd(df, target_column, fast=12, slow=26, signal=9)
    df = add_momentum(df, target_column, window=10)
    df = add_intermediate_changes(df, target_column)
    df = add_ytd_performance(df, target_column)
    df = add_zscore(df, target_column, window=60)
    df = add_minmax_scaling(df, target_column)
    return df

def transform_leading_economic_index(df, target_column, id_column='id'):
    df['MoM_' + target_column] = ((df[target_column] / df[target_column].shift(FORECAST_HORIZON_1MONTH)) - 1).astype(float)
    df['YoY_' + target_column] = ((df[target_column] / df[target_column].shift(252)) - 1).astype(float)
    df['MA_60_' + target_column] = sma_indicator(df[target_column], 60).astype(float)
    df['MA_120_' + target_column] = sma_indicator(df[target_column], 120).astype(float)
    df['EMA_60_' + target_column] = ema_indicator(df[target_column], 60).astype(float)
    df['EMA_120_' + target_column] = ema_indicator(df[target_column], 120).astype(float)
    # Features adicionales para tendencias a mediano/largo plazo
    df = add_log_and_diff_features(df, target_column)
    df = add_intermediate_changes(df, target_column)
    df = add_ytd_performance(df, target_column)
    df = add_zscore(df, target_column, window=60)
    df = add_minmax_scaling(df, target_column)
    return df

def transform_unemployment_rate(df, target_column, id_column='id'):
    df['MoM_' + target_column] = ((df[target_column] / df[target_column].shift(FORECAST_HORIZON_1MONTH)) - 1).astype(float)
    df['MA_60_' + target_column] = sma_indicator(df[target_column], 60).astype(float)
    df['EMA_60_' + target_column] = ema_indicator(df[target_column], 60).astype(float)
    # Features adicionales (útiles para detectar cambios rápidos en la tasa)
    df = add_log_and_diff_features(df, target_column)
    df = add_intermediate_changes(df, target_column)
    df = add_zscore(df, target_column, window=60)
    df = add_minmax_scaling(df, target_column)
    return df

# ================================
# MAIN PIPELINE PARA 1MONTH
# ================================
def main():
    # Ruta del archivo principal para el procesamiento
    input_file = os.path.join(PROJECT_ROOT, "data", "2_processed", "MERGEDEXCELS_CATEGORIZADO_LIMPIO.xlsx")
    
    # Mostrar las opciones de análisis
    print("\n📊 Selecciona el análisis que deseas realizar:")
    print("1. SP500 (usando todas las variables disponibles)")
    print("2. USD/COP (filtrando según variables de Data Engineering.xlsx)")
    
    analysis_type = None
    
    try:
        selection = int(input("\nEscribe tu selección (1 o 2): "))
        if selection == 1:
            analysis_type = "SP500"
            print(f"\n✅ Análisis seleccionado: SP500 usando todas las variables disponibles")
            # Para SP500, usamos el procesamiento normal sin filtrado adicional
            variables_to_keep = None
        elif selection == 2:
            analysis_type = "USDCOP"
            print(f"\n✅ Análisis seleccionado: USD/COP usando variables filtradas")
            
            # Para USD/COP, leemos las variables del archivo Data Engineering.xlsx
            variables_file = os.path.join(PROJECT_ROOT, "pipelines", "Data Engineering.xlsx")
            
            try:
                # Verificar que el archivo existe
                if not os.path.exists(variables_file):
                    print(f"❌ El archivo {variables_file} no existe.")
                    return
                
                # Leer el archivo de variables, específicamente la hoja "Globales"
                try:
                    variables_df_globales = pd.read_excel(variables_file, sheet_name="Globales")
                    variables_df_usdcop = pd.read_excel(variables_file, sheet_name="USDCOP")
                    print("✅ Hojas 'Globales' y 'USDCOP' cargadas exitosamente.")

                    # Obtener lista limpia de variables desde ambas hojas
                    globales_vars = variables_df_globales.iloc[:, 0].dropna().astype(str).str.strip().tolist()
                    usdcop_vars = variables_df_usdcop.iloc[:, 0].dropna().astype(str).str.strip().tolist()

                    # Unir ambas listas y eliminar duplicados
                    variables_to_keep = list(set(globales_vars + usdcop_vars))

                    print(f"✅ Se encontraron {len(globales_vars)} variables en 'Globales' y {len(usdcop_vars)} en 'USDCOP'.")
                    print(f"📋 Total variables combinadas sin duplicados: {len(variables_to_keep)}")
                    print(f"📋 Ejemplo de variables: {variables_to_keep[:5]}")
                    
                    if not variables_to_keep:
                        print("⚠️ No se encontraron variables válidas en las hojas combinadas.")
                        return

                except Exception as e:
                    print(f"❌ Error al leer las hojas 'Globales' o 'USDCOP' del archivo de variables: {e}")
                    return

            except Exception as e:
                print(f"❌ Error al procesar el archivo de variables: {e}")
                return
        else:
            print("❌ Selección inválida. Se realizará análisis de SP500 por defecto.")
            analysis_type = "SP500"
            variables_to_keep = None
    except Exception as e:
        print(f"❌ Error en la selección: {e}. Se realizará análisis de SP500 por defecto.")
        analysis_type = "SP500"
        variables_to_keep = None
    
    # Verificar qué hojas existen en el archivo principal
    try:
        xlsx = pd.ExcelFile(input_file)
        data_available_sheets = xlsx.sheet_names
        print(f"\n📊 Hojas disponibles en el archivo principal: {data_available_sheets}")
        
        if not data_available_sheets:
            print("❌ El archivo Excel principal no contiene hojas. Verifique el archivo.")
            return
            
    except Exception as e:
        print(f"❌ Error al abrir el archivo Excel principal: {e}")
        print(f"⚠️ Ruta de archivo intentada: {input_file}")
        print("⚠️ Verifique que el archivo existe y está en la ruta correcta.")
        return
    
    # Cargar la primera hoja del archivo principal
    try:
        df_raw = pd.read_excel(input_file, sheet_name=data_available_sheets[0], header=None)
        print(f"✅ Hoja '{data_available_sheets[0]}' del archivo principal cargada exitosamente.")
        print("🧪 Primeras filas del archivo principal:")
        print(df_raw.head())
    except Exception as e:
        print(f"❌ Error al leer la hoja '{data_available_sheets[0]}' del archivo principal:", e)
        return
    
    # Procesar el DataFrame principal
    headers = df_raw.iloc[0].tolist()
    categories_row = df_raw.iloc[1].tolist()
    df = df_raw.iloc[2:].copy()
    df.columns = headers
    
    # Aplicar tipo numérico a las columnas
    df = df.apply(pd.to_numeric, errors='ignore')
    
    # Si estamos en modo USD/COP, filtrar columnas según las variables del primer archivo
    if analysis_type == "USDCOP" and variables_to_keep:
        # Filtrar columnas para mantener solo las variables especificadas
        columnas_disponibles = []
        for col in df.columns:
            if col in ['date', 'id', 'fecha']:
                continue
                
            # Verificar si esta columna contiene alguna de las variables a conservar
            keep_column = False
            for var in variables_to_keep:
                if var.lower() in col.lower():  # Comparación no sensible a mayúsculas/minúsculas
                    keep_column = True
                    break
            
            if keep_column:
                columnas_disponibles.append(col)
        
        print(f"\n🎯 Se procesarán {len(columnas_disponibles)} columnas para análisis de {analysis_type}.")
        print(f"🎯 Primeras columnas seleccionadas: {columnas_disponibles[:5]}... (y más)")
        print(f"⚠️ Se eliminarán {len(df.columns) - len(columnas_disponibles) - 2} columnas que no están en la lista de variables.")
    else:
        # Para SP500, usamos todas las columnas excepto date e id
        columnas_disponibles = [col for col in df.columns if col not in ['date', 'id', 'fecha']]
        print(f"\n🎯 Se procesarán {len(columnas_disponibles)} columnas para análisis de {analysis_type}.")
        print(f"🎯 Primeras columnas seleccionadas: {columnas_disponibles[:5]}... (y más)")
    
    # Buscar columna objetivo según el tipo de análisis y asignarla automáticamente
    if analysis_type == "SP500":
        # Para SP500, buscar una columna con S&P500 automáticamente
        posibles_cols_sp500 = [col for col in columnas_disponibles if 'S&P' in col or 'SP500' in col or 'S&P500' in col]
        variable_objetivo = posibles_cols_sp500[0] if posibles_cols_sp500 else None
        
        if variable_objetivo:
            print(f"🎯 Variable objetivo para SP500: {variable_objetivo}")
        else:
            print(f"⚠️ No se encontró automáticamente una variable objetivo para SP500.")
            print("❌ Se requiere una variable objetivo para continuar.")
            return
    else:  # USDCOP
        # Para USD/COP, buscar una columna con USD/COP o USD_COP automáticamente
        posibles_cols_usdcop = [col for col in columnas_disponibles if ('USD' in col and 'COP' in col)]
        
        # Si no encontramos USD_COP, tomamos S&P500 como variable objetivo por defecto para USD/COP
        if not posibles_cols_usdcop:
            posibles_cols_sp500 = [col for col in columnas_disponibles if 'S&P' in col or 'SP500' in col or 'S&P500' in col]
            variable_objetivo = posibles_cols_sp500[0] if posibles_cols_sp500 else None
            
            if variable_objetivo:
                print(f"🎯 No se encontró USD/COP, se usará {variable_objetivo} como variable objetivo para USD/COP")
            else:
                print(f"⚠️ No se encontró automáticamente una variable objetivo para USD/COP.")
                print("❌ Se requiere una variable objetivo para continuar.")
                return
        else:
            variable_objetivo = posibles_cols_usdcop[0]
            print(f"🎯 Variable objetivo para USD/COP: {variable_objetivo}")
    
    print("\n✅ Headers corregidos:")
    print(df.columns[:10].tolist())

    if 'categories_row' in locals() and len(categories_row) > 0:
        print("\n✅ Categorías detectadas:")
        print(categories_row[:10])
    else:
        # Si no tenemos categorías, intentamos inferirlas de los nombres de columnas
        print("\n⚠️ No se detectaron categorías. Se inferirán de los nombres de columnas.")
        categories_row = []
        for col in df.columns:
            if 'bond' in col.lower():
                categories_row.append('bond')
            elif 'confidence' in col.lower() and 'business' in col.lower():
                categories_row.append('business_confidence')
            elif 'car' in col.lower() and 'registration' in col.lower():
                categories_row.append('car_registrations')
            elif 'loan' in col.lower():
                categories_row.append('comm_loans')
            elif 'commodity' in col.lower() or any(c in col.lower() for c in ['gold', 'silver', 'oil', 'copper']):
                categories_row.append('commodities')
            elif 'confidence' in col.lower() and 'consumer' in col.lower():
                categories_row.append('consumer_confidence')
            elif any(e in col.lower() for e in ['gdp', 'cpi', 'ppi', 'inflation']):
                categories_row.append('economics')
            elif 'exchange' in col.lower() or 'usd' in col.lower() or 'eur' in col.lower():
                categories_row.append('exchange_rate')
            elif 'export' in col.lower():
                categories_row.append('exports')
            elif 'index' in col.lower() or any(i in col.lower() for i in ['s&p', 'nasdaq', 'dow', 'ftse']):
                categories_row.append('index_pricing')
            elif 'lead' in col.lower() and 'economic' in col.lower():
                categories_row.append('leading_economic_index')
            elif 'unemployment' in col.lower() or 'jobless' in col.lower():
                categories_row.append('unemployment_rate')
            else:
                categories_row.append('economics')  # Categoría por defecto
    
    # Asegurarse de que la columna 'date' existe y está en formato correcto
    if 'fecha' in df.columns and 'date' not in df.columns:
        df = df.rename(columns={"fecha": "date"})
    
    df = df[df['date'].apply(lambda x: isinstance(x, str) or isinstance(x, pd.Timestamp))]
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    except Exception as e:
        print("❌ Error al convertir fechas:", e)

    df = df[df['date'].notna()]

    # Asegurarse de que la columna 'id' existe
    if 'id' not in df.columns:
        df.insert(1, 'id', 'serie_default')

    # Crear un subconjunto del DataFrame con solo date, id y las columnas seleccionadas
    cols_to_keep = ['date', 'id'] + columnas_disponibles
    df = df[[col for col in cols_to_keep if col in df.columns]]

    # Convertir columnas a tipo numérico y limpiar valores
    df = convert_dataframe(df, excluded_column=None, id_column='id', datetime_column='date')
    
    # Imputar valores faltantes y convertir a frecuencia diaria de negocio
    df = impute_time_series_ffill(df, datetime_column="date", id_column="id")
    df = resample_to_business_day(df, input_frequency="D", column_date="date", id_column="id", output_frequency="B")

    # Crear diccionario de categorías para las columnas
    if len(categories_row) == len(headers):
        column_categories = dict(zip(headers, categories_row))
    else:
        # Si las longitudes no coinciden, crear un diccionario vacío
        column_categories = {}
        for i, col in enumerate(df.columns):
            if i < len(categories_row):
                column_categories[col] = categories_row[i]
            else:
                # Asignar una categoría basada en el nombre si está fuera de rango
                if 'bond' in col.lower():
                    column_categories[col] = 'bond'
                elif any(i in col.lower() for i in ['s&p', 'nasdaq', 'dow', 'ftse']):
                    column_categories[col] = 'index_pricing'
                elif 'exchange' in col.lower() or 'usd' in col.lower() or 'eur' in col.lower():
                    column_categories[col] = 'exchange_rate'
                else:
                    column_categories[col] = 'economics'  # Categoría por defecto

    # ⚠️ Definir manualmente la frecuencia aquí (se mantiene igual)
    column_frequencies = {
        "PRICE_Australia_10Y_Bond_bond": "D",
        "PRICE_Italy_10Y_Bond_bond": "D",
        "PRICE_Japan_10Y_Bond_bond": "D",
        "PRICE_UK_10Y_Bond_bond": "D",
        "PRICE_Germany_10Y_Bond_bond": "D",
        "PRICE_Canada_10Y_Bond_bond": "D",
        "PRICE_China_10Y_Bond_bond": "D",
        "PRICE_CrudeOil_WTI_commodities": "D",
        "PRICE_Gold_Spot_commodities": "D",
        "PRICE_Silver_Spot_commodities": "D",
        "PRICE_Copper_Futures_commodities": "D",
        "PRICE_Platinum_Spot_commodities": "D",
        "PRICE_EUR_USD_Spot_exchange_rate": "D",
        "PRICE_GBP_USD_Spot_exchange_rate": "D",
        "PRICE_JPY_USD_Spot_exchange_rate": "D",
        "PRICE_CNY_USD_Spot_exchange_rate": "D",
        "PRICE_AUD_USD_Spot_exchange_rate": "D",
        "PRICE_CAD_USD_Spot_exchange_rate": "D",
        "PRICE_MXN_USD_Spot_exchange_rate": "D",
        "PRICE_EUR_GBP_Cross_exchange_rate": "D",
        "PRICE_USD_COP_Spot_exchange_rate": "D",  # Añadido USD/COP
        "ULTIMO_S&P500_Index_index_pricing": "D",
        "ULTIMO_NASDAQ_Composite_index_pricing": "D",
        "ULTIMO_Russell_2000_index_pricing": "D",
        "ULTIMO_FTSE_100_index_pricing": "D",
        "ULTIMO_Nikkei_225_index_pricing": "D",
        "ULTIMO_DAX_30_index_pricing": "D",
        "PRICE_Shanghai_Composite_index_pricing": "D",
        "ULTIMO_VIX_VolatilityIndex_index_pricing": "D",
        "ESI_GACDISA_US_Empire_State_Index_business_confidence": "M",
        "ESI_AWCDISA_US_Empire_State_Index_business_confidence": "M",
        "AAII_Bearish_AAII_Investor_Sentiment_consumer_confidence": "W",
        "AAII_Bull-Bear Spread_AAII_Investor_Sentiment_consumer_confidence": "W",
        "AAII_Bullish_AAII_Investor_Sentiment_consumer_confidence": "W",
        "PutCall_strike_Put_Call_Ratio_SPY_consumer_confidence": "W",
        "PutCall_bid_Put_Call_Ratio_SPY_consumer_confidence": "W",
        "PutCall_ask_Put_Call_Ratio_SPY_consumer_confidence": "W",
        "PutCall_vol_Put_Call_Ratio_SPY_consumer_confidence": "W",
        "PutCall_delta_Put_Call_Ratio_SPY_consumer_confidence": "W",
        "PutCall_gamma_Put_Call_Ratio_SPY_consumer_confidence": "W",
        "PutCall_theta_Put_Call_Ratio_SPY_consumer_confidence": "W",
        "PutCall_vega_Put_Call_Ratio_SPY_consumer_confidence": "W",
        "PutCall_rho_Put_Call_Ratio_SPY_consumer_confidence": "W",
        "NFCI_Chicago_Fed_NFCI_leading_economic_index": "M",
        "ANFCI_Chicago_Fed_NFCI_leading_economic_index": "M",
        "Actual_US_ISM_Manufacturing_business_confidence": "M",
        "Actual_US_ISM_Services_business_confidence": "M",
        "Actual_US_Philly_Fed_Index_business_confidence": "M",
        "Actual_France_Business_Climate_business_confidence": "M",
        "Actual_EuroZone_Business_Climate_business_confidence": "M",
        "Actual_US_Consumer_Confidence_consumer_confidence": "M",
        "Actual_China_PMI_Manufacturing_economics": "Q",
        "Actual_US_ConferenceBoard_LEI_leading_economic_index": "D",
        "Actual_Japan_Leading_Indicator_leading_economic_index": "D",
        "DGS10_US_10Y_Treasury_bond": "M",
        "DGS2_US_2Y_Treasury_bond": "M",
        "AAA_Corporate_Bond_AAA_Spread_bond": "D",
        "BAA10YM_Corporate_Bond_BBB_Spread_bond": "M",
        "BAMLH0A0HYM2_High_Yield_Bond_Spread_bond": "M",
        "DNKSLRTCR03GPSAM_Denmark_Car_Registrations_MoM": "M",
        "USASLRTCR03GPSAM_US_Car_Registrations_MoM": "M",
        "ZAFSLRTCR03GPSAM_SouthAfrica_Car_Registrations_MoM": "M",
        "GBRSLRTCR03GPSAM_United_Kingdom_Car_Registrations_MoM": "M",
        "ESPSLRTCR03GPSAM_Spain_Car_Registrations_MoM": "M",
        "BUSLOANS_US_Commercial_Loans_comm_loans": "M",
        "CREACBM027NBOG_US_RealEstate_Commercial_Loans_comm_loans": "M",
        "TOTALSL_US_Consumer_Credit_comm_loans": "M",
        "CSCICP02EZM460S_EuroZone_Consumer_Confidence_consumer_confidence": "M",
        "CSCICP02CHQ460S_Switzerland_Consumer_Confidence_consumer_confidence": "M",
        "UMCSENT_Michigan_Consumer_Sentiment_consumer_confidence": "M",
        "CPIAUCSL_US_CPI_economics": "M",
        "CPILFESL_US_Core_CPI_economics": "M",
        "PCE_US_PCE_economics": "M",
        "PCEPILFE_US_Core_PCE_economics": "M",
        "PPIACO_US_PPI_economics": "M",
        "INDPRO_US_Industrial_Production_MoM_economics": "M",
        "CSUSHPINSA_US_CaseShiller_HomePrice_economics": "M",
        "GDP_US_GDP_Growth_economics": "M",
        "TCU_US_Capacity_Utilization_economics": "M",
        "PERMIT_US_Building_Permits_economics": "M",
        "HOUST_US_Housing_Starts_economics": "M",
        "FEDFUNDS_US_FedFunds_Rate_economics": "M",
        "ECBDFR_ECB_Deposit_Rate_economics": "M",
        "WALCL_Fed_Balance_Sheet_economics": "M",
        "Price_Dollar_Index_DXY_index_pricing": "M",
        "PRICE_US_Unemployment_Rate_unemployment_rate": "M",
        "PRICE_US_Nonfarm_Payrolls_unemployment_rate": "M",
        "PRICE_US_Initial_Jobless_Claims_unemployment_rate": "M",
        "PRICE_US_JOLTS_unemployment_rate": "M"
    }
    
    # Permitir inferir frecuencia para columnas no listadas explícitamente
    for col in df.columns:
        if col not in column_frequencies and col not in ['date', 'id']:
            # Por defecto, asumir frecuencia diaria para columnas no listadas
            column_frequencies[col] = "D"

    # Definición de funciones de transformación por categoría (sin cambios)
    categories = {
        'bond': transform_bond,
        'business_confidence': transform_business_confidence,
        'car_registrations': transform_car_registrations,
        'comm_loans': transform_comm_loans,
        'commodities': transform_commodities,
        'consumer_confidence': transform_consumer_confidence,
        'economics': transform_economics,
        'exchange_rate': transform_exchange_rate,
        'exports': transform_exports,
        'index_pricing': transform_index_pricing,
        'leading_economic_index': transform_leading_economic_index,
        'unemployment_rate': transform_unemployment_rate
    }

    # Agrupar columnas por categoría
    grouped = {cat: [] for cat in categories.keys()}
    for col in df.columns:
        if col not in ['date', 'id']:
            # Obtener categoría del diccionario o inferir del nombre de la columna
            cat = column_categories.get(col, None)
            if cat is None:
                # Inferir categoría si no está en el diccionario
                if 'bond' in col.lower():
                    cat = 'bond'
                elif any(i in col.lower() for i in ['s&p', 'nasdaq', 'dow', 'ftse']):
                    cat = 'index_pricing'
                elif 'exchange' in col.lower() or any(c in col.lower() for c in ['usd', 'eur', 'jpy', 'gbp']):
                    cat = 'exchange_rate'
                else:
                    cat = 'economics'  # Categoría por defecto
            
            if cat in grouped:
                grouped[cat].append(col)
            else:
                grouped['economics'].append(col)  # Si la categoría no existe, usar 'economics'

    print("\n📊 Columnas agrupadas por categoría:")
    for cat, cols in grouped.items():
        print(f"{cat}: {len(cols)} columnas")

    # Transformar las columnas según su categoría
    transformed_dfs = []
    untouched_cols = []

    for cat, cols in grouped.items():
        func = categories[cat]
        for col in cols:
            if col not in df.columns:
                continue
            temp_df = df[['date', 'id', col]].copy()
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
            if temp_df[col].dropna().empty:
                print(f"⚠️  Columna '{col}' está vacía o no tiene valores numéricos. Se omite.")
                continue
            freq = column_frequencies.get(col, "D")  # Por defecto usar "D" si no está en el diccionario
            if freq != 'D':
                print(f"⏭️  Columna '{col}' con frecuencia '{freq}' no es diaria. No se transforma.")
                untouched_cols.append(temp_df)
                continue
            print(f"⚙️  Transformando '{col}' en categoría '{cat}'")
            try:
                transformed = func(temp_df, target_column=col, id_column='id')
                transformed_dfs.append(transformed)
            except Exception as e:
                print(f"❌ Error al transformar '{col}': {e}")
                untouched_cols.append(temp_df)  # Agregar como no transformada si falla

    # Combinar todos los DataFrames transformados
    all_dfs = transformed_dfs + untouched_cols
    if not all_dfs:
        print("❌ No se encontraron series válidas para procesar.")
        return

    final_df = all_dfs[0]
    for tdf in all_dfs[1:]:
        final_df = pd.merge(final_df, tdf, on=['date', 'id'], how='outer')

    # CAMBIO IMPORTANTE: Modificar esta parte para usar imputación en lugar de eliminación
    print(f"🔍 Debug: Cantidad de filas antes de procesar datos faltantes: {len(final_df)}")

    # Identificar columnas de datos (todas excepto 'date' e 'id')
    data_columns = [col for col in final_df.columns if col not in ['date', 'id']]

    # En lugar de eliminar filas, rellenar valores NaN con el último valor válido
    print(f"🔍 Debug: Aplicando forward fill y backward fill para preservar todas las fechas")



    # Si todavía quedan NaN después del ffill/bfill, rellenar con ceros
    # Esto puede ocurrir en columnas que están completamente vacías


    # CAMBIO IMPORTANTE: Para las variables objetivo, también usamos imputación
    if variable_objetivo and variable_objetivo in final_df.columns:
        print(f"🔍 Debug: Creando columna target {variable_objetivo}_Target")
        # Usar shift(-FORECAST_HORIZON) para que coincida con tu pipeline de inferencia
        final_df[variable_objetivo + "_Target"] = final_df[variable_objetivo].shift(-FORECAST_HORIZON_1MONTH)
        
        # Añadir la transformación a retorno para la variable target
        print(f"🔍 Debug: Creando columna de retorno {variable_objetivo}_Return_Target")
        final_df[variable_objetivo + "_Return_Target"] = (final_df[variable_objetivo + "_Target"] / 
                                                        final_df[variable_objetivo]) - 1
        
        # Asegurarnos de que ambas columnas se muevan al final
        columnas = final_df.columns.tolist()
        for col in [variable_objetivo + "_Target", variable_objetivo + "_Return_Target"]:
            if col in columnas:
                columnas.remove(col)
                columnas.append(col)
        final_df = final_df[columnas]
        
        # En lugar de eliminar las filas con NaN en las columnas target,
        # simplemente marcamos la información para conocimiento del usuario
        target_columns = [variable_objetivo + "_Target", variable_objetivo + "_Return_Target"]
        null_count_target = final_df[target_columns].isna().sum().sum()
        
        print(f"🔍 Debug: Hay {null_count_target} valores NaN en las columnas target (esto es normal al final de la serie)")
        print(f"\n🎯 Columnas objetivo '{variable_objetivo}_Target' (valor absoluto) y '{variable_objetivo}_Return_Target' (retorno) añadidas al final (con horizonte de {FORECAST_HORIZON_1MONTH} días).")
    else:
        if variable_objetivo:
            print(f"\n⚠️ Advertencia: La variable objetivo '{variable_objetivo}' no existe en el DataFrame final.")
            print(f"⚠️ Columnas disponibles: {', '.join(final_df.columns[:10])}... (y más)")
        else:
            print("\n⚠️ No se seleccionó ninguna variable objetivo.")
    
    print(f"🔍 Debug: Últimas 5 columnas después = {final_df.columns[-5:].tolist()}")
    print(f"🔍 Debug: Cantidad final de filas en el DataFrame: {len(final_df)}")

    # Crear el nombre del archivo de salida con el tipo de análisis
    output_file = os.path.join(PROJECT_ROOT, "data", "2_processed", 
                              f"datos_economicos_1month_{analysis_type}.xlsx")
    
        # Eliminar cualquier fila que tenga al menos un valor vacío

    # Esto te mostrará exactamente en qué columnas se generan NaNs:
    null_counts = final_df.isnull().sum()
    print("🔍 Valores nulos por columna:")
    print(null_counts[null_counts > 0])
    # Mostrar exactamente dónde y cuántos NaN tienes
    print(final_df.isnull().sum().sort_values(ascending=False).head(20))

    
    final_df.dropna(inplace=True)

    # Confirmación
    print(f"🔍 Debug: Cantidad de filas después de eliminar filas con datos faltantes: {len(final_df)}")

# 1️⃣ Reemplazar infinitos por NaN
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2️⃣ Imputar valores faltantes restantes (forward y backward fill)
    final_df = final_df.ffill().bfill()

# 3️⃣ Si aún queda algún NaN (por columnas completamente vacías), reemplazar por 0
    final_df.fillna(0, inplace=True)

    try:
        final_df.to_excel(output_file, index=False)
        print(f"✅ Proceso completado para análisis de {analysis_type}. Archivo guardado en: {output_file}")
        print(f"✅ El archivo contiene {len(final_df)} filas y {len(final_df.columns)} columnas.")
    except Exception as e:
        print(f"❌ Error al guardar el archivo:", e)

if __name__ == "__main__":
    main()
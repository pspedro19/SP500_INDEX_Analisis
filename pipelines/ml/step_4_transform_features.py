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
    # z-score basado en ventana móvil
    rolling_mean = df[target_column].rolling(window).mean()
    rolling_std = df[target_column].rolling(window).std()
    df['zscore_' + target_column] = (df[target_column] - rolling_mean) / rolling_std
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
    input_file = os.path.join(PROJECT_ROOT, "data", "2_processed", "MERGEDEXCELS_CATEGORIZADO_LIMPIO.xlsx")
    output_file = os.path.join(PROJECT_ROOT, "data", "2_processed", "datos_economicos_1month_procesados.xlsx")
    
    try:
        df_raw = pd.read_excel(input_file, header=None)
    except Exception as e:
        return

    headers = df_raw.iloc[0].tolist()
    categories_row = df_raw.iloc[1].tolist()
    df = df_raw.iloc[2:].copy()  # Saltamos fila de frecuencia
    df = df.apply(pd.to_numeric, errors='ignore')
    df.columns = headers

    columnas_disponibles = [col for col in df.columns if col not in ['date', 'id']]
    for idx, col in enumerate(columnas_disponibles):

    variable_objetivo = None
    try:
        seleccion = int(input("\nEscribe el número de la columna que deseas usar como variable objetivo: "))
        if 1 <= seleccion <= len(columnas_disponibles):
            variable_objetivo = columnas_disponibles[seleccion - 1]
        else:
    except Exception as e:



    df = df.rename(columns={"fecha": "date"})
    df = df[df['date'].apply(lambda x: isinstance(x, str) or isinstance(x, pd.Timestamp))]
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    except Exception as e:

    df = df[df['date'].notna()]

    if 'id' not in df.columns:
        df.insert(1, 'id', 'serie_default')

    df = convert_dataframe(df, excluded_column=None, id_column='id', datetime_column='date')
    df = impute_time_series_ffill(df, datetime_column="date", id_column="id")
    df = resample_to_business_day(df, input_frequency="D", column_date="date", id_column="id", output_frequency="B")

    column_categories = dict(zip(headers, categories_row))

    # ⚠️ Definir manualmente la frecuencia aquí:
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

    grouped = {cat: [] for cat in categories.keys()}
    for col, cat in column_categories.items():
        if cat in grouped:
            grouped[cat].append(col)

    for cat, cols in grouped.items():

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
                continue
            freq = column_frequencies.get(col, None)
            if freq != 'D':
                untouched_cols.append(temp_df)
                continue
            try:
                transformed = func(temp_df, target_column=col, id_column='id')
                transformed_dfs.append(transformed)
            except Exception as e:

    all_dfs = transformed_dfs + untouched_cols
    if not all_dfs:
        return

    final_df = all_dfs[0]
    for tdf in all_dfs[1:]:
        final_df = pd.merge(final_df, tdf, on=['date', 'id'], how='outer')

    # ================================
    # ✅ AGREGAR VARIABLE OBJETIVO COMO *_Target AL FINAL - CON DEPURACIÓN
    # ================================
    
    if variable_objetivo and variable_objetivo in final_df.columns:
        # Usar shift(-FORECAST_HORIZON) para que coincida con tu pipeline de inferencia
        final_df[variable_objetivo + "_Target"] = final_df[variable_objetivo].shift(-FORECAST_HORIZON_1MONTH)
        # Asegurarnos de que la columna se mueva al final
        columnas = final_df.columns.tolist()
        columnas.remove(variable_objetivo + "_Target")
        columnas.append(variable_objetivo + "_Target")
        final_df = final_df[columnas]
    else:
        if variable_objetivo:
        else:
    

    try:
        final_df.to_excel(output_file, index=False)
    except Exception as e:


if __name__ == "__main__":
    main()
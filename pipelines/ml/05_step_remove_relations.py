import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
import os
from matplotlib import pyplot as plt
from sp500_analysis.shared.logging.logger import configurar_logging
from sp500_analysis.application.feature_engineering import (
    FeatureSelector,
    detect_feature_frequency as detect_freq,
    remove_correlated_features_by_frequency as remove_corr_by_freq,
    compute_vif_by_frequency as compute_vif,
    iterative_vif_reduction_by_frequency as vif_reduction,
)

# Importar constantes centralizadas
from sp500_analysis.config.settings import settings

PROJECT_ROOT = settings.project_root
PROCESSED_DIR = settings.processed_dir
DATE_COL = settings.date_col
TARGET_SUFFIX = settings.target_suffix
CONSTANT_THRESHOLD = settings.constant_threshold
CORR_THRESHOLD = settings.corr_threshold
VIF_THRESHOLD = settings.vif_threshold

# ------------------------------
# CONFIGURACIÓN DE LOGGING
# ------------------------------
log_file = os.path.join(PROJECT_ROOT, "logs", f"remove_relations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
configurar_logging(log_file)


# Wrapper for frequency detection
def detect_feature_frequency(df, col, date_col=DATE_COL):
    return detect_freq(df, col, date_col)


# FUNCIÓN REMOVIDA: create_lagged_target
# Esta función ya no se necesita ya que no aplicaremos LAG


def remove_derived_target_features(df, target_col, date_col=DATE_COL):
    """
    Elimina features derivadas del target para evitar leakage.

    Args:
        df (DataFrame): DataFrame con features y target
        target_col (str): Nombre de la columna target
        date_col (str): Nombre de la columna de fecha

    Returns:
        tuple: (DataFrame sin columnas derivadas, lista de columnas eliminadas)
    """
    cols_to_drop = []
    if target_col.endswith(TARGET_SUFFIX):
        base = target_col[: -len(TARGET_SUFFIX)]
        pattern = rf'^(log_diff|momentum)_{re.escape(base)}'
        for col in df.columns:
            if col in [target_col, date_col]:
                continue
            if re.search(pattern, col):
                cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop, errors='ignore'), cols_to_drop


def drop_constant_features(df, threshold=CONSTANT_THRESHOLD, date_col=DATE_COL):
    """Wrapper around :class:`FeatureSelector` to drop near constant columns."""
    selector = FeatureSelector(date_col=date_col, constant_threshold=threshold)
    return selector.drop_constant_features(df)


def analyze_lags_with_target(df, target_col, date_col=DATE_COL, max_lag=10):
    """
    Analiza correlaciones con lags entre variables y el target.

    Args:
        df (DataFrame): DataFrame con series temporales
        target_col (str): Nombre de la columna objetivo
        date_col (str): Nombre de la columna de fecha
        max_lag (int): Máximo lag a considerar

    Returns:
        DataFrame: Correlaciones máximas con el lag óptimo
    """
    df_sorted = df.sort_values(date_col).copy()
    numeric_cols = [
        col
        for col in df_sorted.columns
        if pd.api.types.is_numeric_dtype(df_sorted[col]) and col not in [date_col, target_col]
    ]

    # Para cada columna, encontrar el lag con mayor correlación
    results = []

    for col in numeric_cols:
        max_corr = 0
        best_lag = 0

        # Probar diferentes lags
        for lag in range(max_lag + 1):
            # Correlación con lag
            if lag == 0:
                corr = df_sorted[col].corr(df_sorted[target_col])
            else:
                corr = df_sorted[col].corr(df_sorted[target_col].shift(-lag))

            if abs(corr) > abs(max_corr):
                max_corr = corr
                best_lag = lag

        # Detectar frecuencia
        freq = detect_feature_frequency(df_sorted, col, date_col)

        results.append({'column': col, 'frequency': freq, 'max_correlation': max_corr, 'optimal_lag': best_lag})

    # Crear DataFrame con resultados
    lag_corr_df = pd.DataFrame(results)
    lag_corr_df = lag_corr_df.sort_values('max_correlation', key=abs, ascending=False)

    return lag_corr_df


# Wrapper for stationarity testing
def test_feature_stationarity(df, date_col=DATE_COL, significance=0.05):
    selector = FeatureSelector(date_col=date_col)
    return selector.test_stationarity(df, significance)


# Wrapper for correlation removal
def remove_correlated_features_by_frequency(df_features, target, date_col=DATE_COL, threshold=CORR_THRESHOLD):
    return remove_corr_by_freq(df_features, target, date_col=date_col, threshold=threshold)


# Wrapper for VIF computation
def compute_vif_by_frequency(df_features, date_col=DATE_COL):
    return compute_vif(df_features, date_col=date_col)


# Wrapper for iterative VIF reduction
def iterative_vif_reduction_by_frequency(df_features, date_col=DATE_COL, threshold=VIF_THRESHOLD):
    return vif_reduction(df_features, date_col=date_col, threshold=threshold)


def create_feature_summary_report(df, target_col, date_col=DATE_COL, output_file=None):
    """
    Genera un informe detallado de las características de las variables.

    Args:
        df (DataFrame): DataFrame con series temporales
        target_col (str): Nombre de la columna objetivo
        date_col (str): Nombre de la columna de fecha
        output_file (str): Ruta para guardar el informe (opcional)

    Returns:
        DataFrame: Informe detallado
    """
    report_data = []

    # Análisis de lags
    lag_df = analyze_lags_with_target(df, target_col, date_col)

    # Análisis de estacionariedad
    stat_df = test_feature_stationarity(df, date_col)

    # Crear informe combinado
    for col in df.columns:
        if col in [date_col, target_col]:
            continue

        # Estadísticas básicas
        data_type = str(df[col].dtype)
        missing = df[col].isna().sum()
        missing_pct = missing / len(df) * 100

        # Detectar frecuencia
        freq = detect_feature_frequency(df, col, date_col)

        # Estadísticas para variables numéricas
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()

            # Buscar correlación óptima
            corr_data = lag_df[lag_df['column'] == col]
            if not corr_data.empty:
                max_corr = corr_data.iloc[0]['max_correlation']
                best_lag = int(corr_data.iloc[0]['optimal_lag'])
            else:
                max_corr = float('nan')
                best_lag = float('nan')

            # Buscar resultado estacionariedad
            stat_data = stat_df[stat_df['column'] == col]
            if not stat_data.empty and stat_data.iloc[0]['is_stationary'] is not None:
                is_stationary = stat_data.iloc[0]['is_stationary']
                p_value = stat_data.iloc[0]['p_value']
            else:
                is_stationary = None
                p_value = None
        else:
            # Para variables no numéricas
            mean_val = std_val = min_val = max_val = float('nan')
            max_corr = best_lag = float('nan')
            is_stationary = None
            p_value = None

        # Contar cambios
        changes = (df[col] != df[col].shift(1)).sum()
        change_ratio = changes / len(df) * 100

        report_data.append(
            {
                'column': col,
                'frequency': freq,
                'data_type': data_type,
                'missing_count': missing,
                'missing_percent': missing_pct,
                'changes_count': changes,
                'changes_percent': change_ratio,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'max_correlation': max_corr,
                'optimal_lag': best_lag,
                'is_stationary': is_stationary,
                'stationarity_p_value': p_value,
            }
        )

    # Crear DataFrame
    report_df = pd.DataFrame(report_data)

    # Ordenar por frecuencia y luego por correlación
    report_df['freq_order'] = report_df['frequency'].map({'D': 0, 'W': 1, 'M': 2, 'Q': 3})
    report_df = report_df.sort_values(
        ['freq_order', 'max_correlation'],
        key=lambda x: x if x.name != 'max_correlation' else abs(x),
        ascending=[True, False],
    )
    report_df = report_df.drop(columns=['freq_order'])

    # Guardar informe si se especifica archivo
    if output_file:
        report_df.to_excel(output_file, index=False)
        logging.info(f"Informe de variables guardado en: {output_file}")

    return report_df


def main():
    """
    Función principal que ejecuta el proceso de reducción de dimensionalidad
    con mejoras para series temporales financieras.

    Proceso:
    1. Eliminar features derivadas del target
    2. Eliminar features constantes con ajuste por frecuencia
    3. Análisis de correlación temporal con el target
    4. Eliminar features correlacionadas por grupos de frecuencia
    5. Reducir multicolinealidad mediante VIF adaptado a frecuencias
    6. Generar informe detallado de variables
    7. NOTA: Se removió la aplicación de LAG al target
    """
    logging.info("Iniciando proceso de selección y filtrado de features para series temporales...")

    # Usar rutas definidas en config
    archivo_entrada = os.path.join(PROCESSED_DIR, "datos_economicos_1month_SP500_TRAINING.xlsx")
    archivo_salida = os.path.join(PROCESSED_DIR, "ULTIMO_S&P500_final.xlsx")
    archivo_informe = os.path.join(PROCESSED_DIR, "informe_variables.xlsx")

    # Verificar existencia del archivo
    if not os.path.exists(archivo_entrada):
        logging.error(f"El archivo de entrada no existe: {archivo_entrada}")
        return

    # Cargar datos
    df = pd.read_excel(archivo_entrada)
    logging.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Identificar target (última columna)
    target_col = df.columns[-1]
    logging.info(f"Target identificado: {target_col}")

    # 1. Eliminar features derivadas del target
    df, dropped_derived = remove_derived_target_features(df, target_col)
    if dropped_derived:
        logging.info(f"Features derivadas eliminadas: {len(dropped_derived)}")
        logging.info(f"Ejemplos: {dropped_derived[:5]}...")

    # Separar target y features
    target = df[target_col]
    features = df.drop(columns=[target_col])

    # 2. Eliminar features constantes con consideración de frecuencia
    features, dropped_const = drop_constant_features(features, CONSTANT_THRESHOLD, DATE_COL)
    if dropped_const:
        logging.info(f"Features constantes/cuasiconstantes eliminadas: {len(dropped_const)}")
        logging.info(f"Ejemplos: {dropped_const[:5]}...")

    # Guardar columna de fecha para reincorporar al final
    date_data = features.pop(DATE_COL) if DATE_COL in features.columns else None

    # 3. Obtener solo features numéricas
    numeric_features = features.select_dtypes(include=[np.number])
    logging.info(f"Features numéricas para análisis: {numeric_features.shape[1]}")

    # Análisis de correlación temporal con el target
    if date_data is not None:
        temp_df = numeric_features.copy()
        temp_df[DATE_COL] = date_data
        temp_df[target_col] = target

        lag_correlations = analyze_lags_with_target(temp_df, target_col)
        top_predictors = lag_correlations.head(10)
        logging.info(f"Top 10 predictores por correlación temporal:")
        for _, row in top_predictors.iterrows():
            logging.info(
                f"  {row['column']} ({row['frequency']}): "
                f"corr={row['max_correlation']:.3f} en lag={row['optimal_lag']}"
            )

    # 4. Eliminar features correlacionadas por grupos de frecuencia
    if date_data is not None:
        temp_features = numeric_features.copy()
        temp_features[DATE_COL] = date_data
        features_corr, dropped_corr = remove_correlated_features_by_frequency(temp_features, target)
        features_corr = features_corr.drop(columns=[DATE_COL]) if DATE_COL in features_corr.columns else features_corr
    else:
        # Si no hay date_data, usar una función más simple
        features_corr, dropped_corr = remove_correlated_features_by_frequency(numeric_features, target)

    if dropped_corr:
        logging.info(f"Features correlacionadas eliminadas: {len(dropped_corr)}")
        logging.info(f"Ejemplos: {dropped_corr[:5]}...")

    # 5. Reducción por VIF adaptado para frecuencias temporales
    if date_data is not None:
        temp_features = features_corr.copy()
        temp_features[DATE_COL] = date_data
        features_vif, final_vif_df = iterative_vif_reduction_by_frequency(temp_features, DATE_COL)
        if DATE_COL in features_vif.columns:
            features_vif = features_vif.drop(columns=[DATE_COL])
    else:
        # Si no hay date_data, usar función más simple
        features_vif, final_vif_df = iterative_vif_reduction_by_frequency(features_corr)

    logging.info(f"Features finales tras reducción VIF: {features_vif.shape[1]}")

    # Mostrar top 5 VIF más altos (pero bajo el umbral)
    if not final_vif_df.empty:
        top_vif = final_vif_df.sort_values('VIF', ascending=False).head(5)
        logging.info(f"Top VIF finales:\n{top_vif}")

    # 6. Reincorporar fecha y target
    if date_data is not None:
        features_vif.insert(0, DATE_COL, date_data)

    final_df = pd.concat([features_vif, target], axis=1)

    # Generar informe detallado de variables finales
    try:
        feature_report = create_feature_summary_report(final_df, target_col, DATE_COL, archivo_informe)
        logging.info(f"Informe detallado de variables generado con éxito")

        # Análisis de estacionariedad
        stationary_vars = feature_report[feature_report['is_stationary'] == True].shape[0]
        nonstationary_vars = feature_report[feature_report['is_stationary'] == False].shape[0]
        if stationary_vars + nonstationary_vars > 0:
            stat_pct = stationary_vars / (stationary_vars + nonstationary_vars) * 100
            logging.info(
                f"Estacionariedad: {stationary_vars} variables estacionarias ({stat_pct:.1f}%), "
                f"{nonstationary_vars} no estacionarias"
            )

        # Distribución por frecuencias
        freq_dist = feature_report['frequency'].value_counts()
        logging.info("Distribución de frecuencias en variables finales:")
        for freq, count in freq_dist.items():
            logging.info(
                f"  {freq} ({'diarias' if freq=='D' else 'semanales' if freq=='W' else 'mensuales' if freq=='M' else 'trimestrales'}): {count}"
            )
    except Exception as e:
        logging.warning(f"No se pudo generar el informe detallado: {str(e)}")

    # 7. NOTA: Se removió la aplicación de LAG al target
    # El dataset final mantiene el target original sin desplazamiento temporal
    logging.info("✅ LAG al target removido - se mantiene el target original")

    # 8. Guardar resultado final (sin LAG)
    final_df.to_excel(archivo_salida, index=False)
    logging.info(f"Dataset final SIN LAG guardado en: {archivo_salida}")
    logging.info(f"Dimensiones finales: {final_df.shape[0]} filas, {final_df.shape[1]} columnas")
    logging.info(
        f"Features finales: {final_df.shape[1] - (1 if date_data is not None else 0) - 1}"
    )  # -1 solo por el target (sin target con lag)


if __name__ == "__main__":
    main()

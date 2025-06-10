import pandas as pd
import numpy as np
import re
import logging
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from scipy.stats import entropy

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_feature_frequency(df, col, date_col=DATE_COL):
    """
    Detecta la frecuencia de actualización de una variable.
    
    Args:
        df (DataFrame): DataFrame con la serie temporal
        col (str): Nombre de la columna a analizar
        date_col (str): Nombre de la columna de fecha
        
    Returns:
        str: Frecuencia detectada ('D', 'W', 'M', 'Q')
    """
    if col == date_col:
        return 'D'  # La columna de fecha es diaria por definición
        
    df_sorted = df.sort_values(date_col)
    
    # Contar cambios (diferencias no nulas)
    changes = (df_sorted[col] != df_sorted[col].shift(1)).sum()
    total = len(df_sorted)
    
    if total == 0:
        return 'D'  # Si no hay datos, asumir diario
        
    change_ratio = changes / total
    
    # Clasificar por frecuencia estimada
    if change_ratio > 0.3:
        return 'D'  # Diario/alta frecuencia
    elif change_ratio > 0.1:
        return 'W'  # Semanal
    elif change_ratio > 0.03:
        return 'M'  # Mensual
    else:
        return 'Q'  # Trimestral o menor frecuencia

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
        base = target_col[:-len(TARGET_SUFFIX)]
        pattern = rf'^(log_diff|momentum)_{re.escape(base)}'
        for col in df.columns:
            if col in [target_col, date_col]:
                continue
            if re.search(pattern, col):
                cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop, errors='ignore'), cols_to_drop

def drop_constant_features(df, threshold=CONSTANT_THRESHOLD, date_col=DATE_COL):
    """
    Elimina features con varianza muy baja (casi constantes), adaptado para
    diferentes frecuencias de actualización.
    
    Args:
        df (DataFrame): DataFrame con features
        threshold (float): Umbral para considerar una feature como constante
        date_col (str): Nombre de la columna de fecha
        
    Returns:
        tuple: (DataFrame sin columnas constantes, lista de columnas eliminadas)
    """
    cols_to_drop = []
    
    # Definir umbrales ajustados por frecuencia
    freq_thresholds = {
        'D': threshold,        # Diario - umbral estándar
        'W': threshold * 1.2,  # Semanal - más tolerante
        'M': threshold * 1.5,  # Mensual - mucho más tolerante
        'Q': threshold * 2.0   # Trimestral - extremadamente tolerante
    }
    
    # Información detallada de eliminaciones
    freq_counts = {'D': 0, 'W': 0, 'M': 0, 'Q': 0}
    columns_by_freq = {'D': [], 'W': [], 'M': [], 'Q': []}
    dropped_by_freq = {'D': [], 'W': [], 'M': [], 'Q': []}
    
    for col in df.columns:
        if col == date_col:
            continue
            
        # Detectar frecuencia de la variable
        freq = detect_feature_frequency(df, col, date_col)
        columns_by_freq[freq].append(col)
        
        # Aplicar umbral adaptado a la frecuencia
        adjusted_threshold = freq_thresholds[freq]
        
        # Criterios para variables constantes/cuasiconstantes
        is_constant = False
        
        # Para variables numéricas, usar desviación estándar
        if pd.api.types.is_numeric_dtype(df[col]):
            std_dev = df[col].std()
            mean_val = df[col].mean() if not pd.isna(df[col].mean()) else 1
            
            # Coeficiente de variación (para comparar variables con diferentes escalas)
            if mean_val != 0:
                cv = abs(std_dev / mean_val)
                is_constant = cv < (0.01 * (1 + ['D', 'W', 'M', 'Q'].index(freq)))
            else:
                is_constant = std_dev < 1e-8
        
        # Para variables categóricas, usar conteo de valores y entropía
        else:
            value_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
            is_constant = value_freq > adjusted_threshold
            
            # Análisis adicional para variables categóricas usando entropía
            if not is_constant and len(df[col].unique()) > 1:
                # Calcular entropía normalizada (0=constante, 1=distribuido uniformemente)
                counts = df[col].value_counts(dropna=False)
                norm_entropy = entropy(counts) / np.log(len(counts))
                # Ajustar umbral de entropía según frecuencia
                entropy_threshold = 0.1 * (1 + ['D', 'W', 'M', 'Q'].index(freq) * 0.5)
                is_constant = norm_entropy < entropy_threshold
        
        if is_constant:
            cols_to_drop.append(col)
            dropped_by_freq[freq].append(col)
    
    # Loguear información sobre la distribución de frecuencias
    for freq in ['D', 'W', 'M', 'Q']:
        freq_counts[freq] = len(columns_by_freq[freq])
        dropped_count = len(dropped_by_freq[freq])
        if freq_counts[freq] > 0:
            drop_ratio = dropped_count / freq_counts[freq] * 100
            logging.info(f"Variables {freq}({'diarias' if freq=='D' else 'semanales' if freq=='W' else 'mensuales' if freq=='M' else 'trimestrales'}): "
                         f"{freq_counts[freq]} totales, {dropped_count} eliminadas ({drop_ratio:.1f}%)")
    
    return df.drop(columns=cols_to_drop, errors='ignore'), cols_to_drop

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
    numeric_cols = [col for col in df_sorted.columns 
                   if pd.api.types.is_numeric_dtype(df_sorted[col]) 
                   and col not in [date_col, target_col]]
    
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
        
        results.append({
            'column': col,
            'frequency': freq,
            'max_correlation': max_corr,
            'optimal_lag': best_lag
        })
    
    # Crear DataFrame con resultados
    lag_corr_df = pd.DataFrame(results)
    lag_corr_df = lag_corr_df.sort_values('max_correlation', key=abs, ascending=False)
    
    return lag_corr_df

def test_feature_stationarity(df, date_col=DATE_COL, significance=0.05):
    """
    Evalúa estacionariedad de las variables mediante test aumentado de Dickey-Fuller.
    
    Args:
        df (DataFrame): DataFrame con series temporales
        date_col (str): Nombre de la columna de fecha
        significance (float): Nivel de significancia para el test
        
    Returns:
        DataFrame: Resultados del test de estacionariedad
    """
    df_sorted = df.sort_values(date_col).copy()
    numeric_cols = [col for col in df_sorted.columns 
                   if pd.api.types.is_numeric_dtype(df_sorted[col]) 
                   and col != date_col]
    
    results = []
    
    for col in numeric_cols:
        # Eliminar valores nulos para el test
        series = df_sorted[col].dropna()
        
        if len(series) < 20:  # Necesitamos suficientes observaciones
            results.append({
                'column': col,
                'is_stationary': None,
                'p_value': None,
                'message': 'Insuficientes datos'
            })
            continue
        
        # Test de Dickey-Fuller
        try:
            result = adfuller(series, autolag='AIC')
            p_value = result[1]
            is_stationary = p_value < significance
            
            results.append({
                'column': col,
                'is_stationary': is_stationary,
                'p_value': p_value,
                'message': 'OK'
            })
        except Exception as e:
            results.append({
                'column': col,
                'is_stationary': None,
                'p_value': None,
                'message': str(e)
            })
    
    return pd.DataFrame(results)

def remove_correlated_features_by_frequency(df_features, target, date_col=DATE_COL, threshold=CORR_THRESHOLD):
    """
    Elimina features altamente correlacionadas entre sí por grupos de frecuencia,
    manteniendo la que tenga mayor correlación con el target.
    
    Args:
        df_features (DataFrame): DataFrame con features
        target (Series): Variable objetivo
        date_col (str): Nombre de la columna de fecha
        threshold (float): Umbral de correlación para eliminar
        
    Returns:
        tuple: (DataFrame sin columnas correlacionadas, lista de columnas eliminadas)
    """
    # Agrupar variables por frecuencia
    freq_groups = {'D': [], 'W': [], 'M': [], 'Q': []}
    
    for col in df_features.columns:
        if col == date_col:
            continue
        freq = detect_feature_frequency(df_features, col, date_col)
        freq_groups[freq].append(col)
    
    # Umbrales ajustados por frecuencia
    freq_thresholds = {
        'D': threshold,
        'W': threshold * 1.05,
        'M': threshold * 1.1,
        'Q': threshold * 1.15
    }
    
    all_cols_to_drop = []
    
    # Para cada grupo de frecuencia, aplicar análisis de correlación
    for freq, cols in freq_groups.items():
        if len(cols) <= 1:
            continue
            
        df_group = df_features[cols]
        corr_matrix = df_group.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        
        # Calcular correlación con target para cada columna del grupo
        target_corr = {}
        for col in cols:
            if pd.api.types.is_numeric_dtype(target):
                # Calcular correlación con diferentes lags (0 a 5) y usar la máxima
                max_corr = 0
                for lag in range(6):
                    if lag == 0:
                        corr = abs(df_features[col].corr(target))
                    else:
                        corr = abs(df_features[col].corr(target.shift(-lag)))
                    max_corr = max(max_corr, corr)
                target_corr[col] = max_corr
            else:
                target_corr[col] = 0
        
        # Buscar variables correlacionadas y decidir cuál mantener
        cols_to_drop = set()
        group_threshold = freq_thresholds[freq]
        
        for i in range(len(cols)):
            if cols[i] in cols_to_drop:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in cols_to_drop:
                    continue
                    
                if corr_matrix.loc[cols[i], cols[j]] > group_threshold:
                    if target_corr[cols[i]] < target_corr[cols[j]]:
                        cols_to_drop.add(cols[i])
                        break
                    else:
                        cols_to_drop.add(cols[j])
        
        all_cols_to_drop.extend(cols_to_drop)
        logging.info(f"Variables {freq} correlacionadas eliminadas: {len(cols_to_drop)} de {len(cols)}")
    
    return df_features.drop(columns=all_cols_to_drop, errors='ignore'), all_cols_to_drop

def compute_vif_by_frequency(df_features, date_col=DATE_COL):
    """
    Calcula el Factor de Inflación de Varianza para cada feature,
    agrupando por frecuencia.
    
    Args:
        df_features (DataFrame): DataFrame con features numéricas
        date_col (str): Nombre de la columna de fecha
        
    Returns:
        DataFrame: DataFrame con features, sus VIF y frecuencia
    """
    # Preparar datos
    df_clean = df_features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if date_col in df_clean.columns:
        df_clean = df_clean.drop(columns=[date_col])
    
    # Eliminar columnas constantes
    non_const = [col for col in df_clean.columns if df_clean[col].std() > 0]
    df_clean = df_clean[non_const]
    
    if df_clean.shape[1] == 0:
        return pd.DataFrame(columns=['variable', 'VIF', 'frequency'])
    
    # Calcular VIF
    vif_data = []
    
    for i, column in enumerate(df_clean.columns):
        try:
            vif_value = variance_inflation_factor(df_clean.values, i)
            freq = detect_feature_frequency(df_features, column, date_col)
            vif_data.append({
                'variable': column,
                'VIF': vif_value,
                'frequency': freq
            })
        except Exception as e:
            logging.warning(f"Error al calcular VIF para '{column}': {str(e)}")
    
    return pd.DataFrame(vif_data)

def iterative_vif_reduction_by_frequency(df_features, date_col=DATE_COL, threshold=VIF_THRESHOLD):
    """
    Reduce iterativamente las features con VIF alto, respetando las frecuencias.
    
    Args:
        df_features (DataFrame): DataFrame con features
        date_col (str): Nombre de la columna de fecha
        threshold (float): Umbral VIF para eliminar features
        
    Returns:
        tuple: (DataFrame con features reducidas, DataFrame de VIF final)
    """
    df_reduced = df_features.copy()
    
    # Umbrales ajustados por frecuencia
    freq_thresholds = {
        'D': threshold,
        'W': threshold * 1.2,
        'M': threshold * 1.5,
        'Q': threshold * 2.0
    }
    
    # Límites de eliminación por frecuencia (porcentaje máximo que se puede eliminar)
    freq_limits = {
        'D': 0.7,  # Se puede eliminar hasta 70% de variables diarias
        'W': 0.5,  # Se puede eliminar hasta 50% de variables semanales
        'M': 0.3,  # Se puede eliminar hasta 30% de variables mensuales
        'Q': 0.2   # Se puede eliminar hasta 20% de variables trimestrales
    }
    
    # Contar variables por frecuencia
    freq_counts = {'D': 0, 'W': 0, 'M': 0, 'Q': 0}
    freq_dropped = {'D': 0, 'W': 0, 'M': 0, 'Q': 0}
    
    for col in df_reduced.columns:
        if col == date_col:
            continue
        freq = detect_feature_frequency(df_reduced, col, date_col)
        freq_counts[freq] += 1
    
    while True:
        vif_df = compute_vif_by_frequency(df_reduced, date_col)
        
        if vif_df.empty:
            break
            
        # Verificar si hay VIF por encima del umbral ajustado por frecuencia
        has_high_vif = False
        var_to_drop = None
        max_vif = 0
        
        for _, row in vif_df.iterrows():
            freq = row['frequency']
            adjusted_threshold = freq_thresholds[freq]
            
            # Verificar si ya hemos alcanzado el límite de eliminaciones para esta frecuencia
            current_dropped = freq_dropped[freq]
            max_to_drop = int(freq_counts[freq] * freq_limits[freq])
            
            if row['VIF'] > adjusted_threshold and current_dropped < max_to_drop:
                if row['VIF'] > max_vif:
                    max_vif = row['VIF']
                    var_to_drop = row['variable']
                    has_high_vif = True
        
        if not has_high_vif or var_to_drop is None:
            break
            
        # Eliminar la variable con el VIF más alto
        var_freq = detect_feature_frequency(df_reduced, var_to_drop, date_col)
        logging.info(f"Eliminando '{var_to_drop}' ({var_freq}) por VIF = {max_vif:.2f}")
        df_reduced.drop(columns=[var_to_drop], inplace=True)
        freq_dropped[var_freq] += 1
        
        # Parar si quedan muy pocas columnas
        if len(df_reduced.columns) <= (1 if date_col not in df_reduced.columns else 2):
            break
    
    # Loguear estadísticas
    for freq in ['D', 'W', 'M', 'Q']:
        if freq_counts[freq] > 0:
            drop_pct = freq_dropped[freq] / freq_counts[freq] * 100
            logging.info(f"Variables {freq}: {freq_counts[freq]} iniciales, "
                        f"{freq_dropped[freq]} eliminadas ({drop_pct:.1f}%), "
                        f"{freq_counts[freq] - freq_dropped[freq]} mantenidas")
    
    return df_reduced, compute_vif_by_frequency(df_reduced, date_col)

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
        
        report_data.append({
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
            'stationarity_p_value': p_value
        })
    
    # Crear DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Ordenar por frecuencia y luego por correlación
    report_df['freq_order'] = report_df['frequency'].map({'D': 0, 'W': 1, 'M': 2, 'Q': 3})
    report_df = report_df.sort_values(['freq_order', 'max_correlation'], 
                                      key=lambda x: x if x.name != 'max_correlation' else abs(x),
                                      ascending=[True, False])
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
            logging.info(f"  {row['column']} ({row['frequency']}): "
                         f"corr={row['max_correlation']:.3f} en lag={row['optimal_lag']}")

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
            logging.info(f"Estacionariedad: {stationary_vars} variables estacionarias ({stat_pct:.1f}%), "
                         f"{nonstationary_vars} no estacionarias")
        
        # Distribución por frecuencias
        freq_dist = feature_report['frequency'].value_counts()
        logging.info("Distribución de frecuencias en variables finales:")
        for freq, count in freq_dist.items():
            logging.info(f"  {freq} ({'diarias' if freq=='D' else 'semanales' if freq=='W' else 'mensuales' if freq=='M' else 'trimestrales'}): {count}")
    except Exception as e:
        logging.warning(f"No se pudo generar el informe detallado: {str(e)}")
    
    # 7. NOTA: Se removió la aplicación de LAG al target
    # El dataset final mantiene el target original sin desplazamiento temporal
    logging.info("✅ LAG al target removido - se mantiene el target original")
    
    # 8. Guardar resultado final (sin LAG)
    final_df.to_excel(archivo_salida, index=False)
    logging.info(f"Dataset final SIN LAG guardado en: {archivo_salida}")
    logging.info(f"Dimensiones finales: {final_df.shape[0]} filas, {final_df.shape[1]} columnas")
    logging.info(f"Features finales: {final_df.shape[1] - (1 if date_data is not None else 0) - 1}")  # -1 solo por el target (sin target con lag)

if __name__ == "__main__":
    main()
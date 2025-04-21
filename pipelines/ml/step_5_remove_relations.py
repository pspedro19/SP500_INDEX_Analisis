import pandas as pd
import numpy as np
import re
import logging
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Importar constantes centralizadas
from config import (
    PROJECT_ROOT, PROCESSED_DIR, DATE_COL, TARGET_SUFFIX,
    CONSTANT_THRESHOLD, CORR_THRESHOLD, VIF_THRESHOLD
)

# ------------------------------
# CONFIGURACIÓN DE LOGGING
# ------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def drop_constant_features(df, threshold=CONSTANT_THRESHOLD):
    """
    Elimina features con varianza muy baja (casi constantes).
    
    Args:
        df (DataFrame): DataFrame con features
        threshold (float): Umbral para considerar una feature como constante
        
    Returns:
        tuple: (DataFrame sin columnas constantes, lista de columnas eliminadas)
    """
    cols_to_drop = []
    for col in df.columns:
        if df[col].nunique() == 1 or df[col].value_counts(normalize=True, dropna=False).iloc[0] > threshold:
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop, errors='ignore'), cols_to_drop

def remove_correlated_features(df_features, target, threshold=CORR_THRESHOLD):
    """
    Elimina features altamente correlacionadas entre sí, manteniendo la que 
    tenga mayor correlación con el target.
    
    Args:
        df_features (DataFrame): DataFrame con features
        target (Series): Variable objetivo
        threshold (float): Umbral de correlación para eliminar
        
    Returns:
        tuple: (DataFrame sin columnas correlacionadas, lista de columnas eliminadas)
    """
    corr_matrix = df_features.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    cols_to_drop = set()
    target_corr = {col: abs(df_features[col].corr(target)) if pd.api.types.is_numeric_dtype(target) else 0
                   for col in df_features.columns}
    columns = list(df_features.columns)

    for i in range(len(columns)):
        if columns[i] in cols_to_drop:
            continue
        for j in range(i + 1, len(columns)):
            if columns[j] in cols_to_drop:
                continue
            if corr_matrix.loc[columns[i], columns[j]] > threshold:
                if target_corr[columns[i]] < target_corr[columns[j]]:
                    cols_to_drop.add(columns[i])
                    break
                else:
                    cols_to_drop.add(columns[j])
    return df_features.drop(columns=list(cols_to_drop), errors='ignore'), list(cols_to_drop)

def compute_vif(df_features):
    """
    Calcula el Factor de Inflación de Varianza para cada feature.
    
    Args:
        df_features (DataFrame): DataFrame con features numéricas
        
    Returns:
        DataFrame: DataFrame con features y sus VIF
    """
    df_clean = df_features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    non_const = [col for col in df_clean.columns if df_clean[col].std() > 0]
    df_clean = df_clean[non_const]
    return pd.DataFrame({
        'variable': df_clean.columns,
        'VIF': [variance_inflation_factor(df_clean.values, i) for i in range(df_clean.shape[1])]
    })

def iterative_vif_reduction(df_features, threshold=VIF_THRESHOLD):
    """
    Reduce iterativamente las features con VIF alto.
    
    Args:
        df_features (DataFrame): DataFrame con features
        threshold (float): Umbral VIF para eliminar features
        
    Returns:
        tuple: (DataFrame con features reducidas, DataFrame de VIF final)
    """
    df_reduced = df_features.copy()
    while True:
        vif_df = compute_vif(df_reduced)
        max_vif = vif_df['VIF'].max()
        if max_vif > threshold:
            var_to_drop = vif_df.loc[vif_df['VIF'].idxmax(), 'variable']
            logging.info(f"Eliminando '{var_to_drop}' por VIF = {max_vif:.2f}")
            df_reduced.drop(columns=[var_to_drop], inplace=True)
            if df_reduced.shape[1] <= 1:
                break
        else:
            break
    return df_reduced, compute_vif(df_reduced)

def main():
    """
    Función principal que ejecuta el proceso de reducción de dimensionalidad.
    
    Proceso:
    1. Eliminar features derivadas del target
    2. Eliminar features constantes
    3. Eliminar features correlacionadas
    4. Reducir multicolinealidad mediante VIF
    """
    logging.info("Iniciando proceso de selección y filtrado de features...")
    
    # Usar rutas definidas en config
    archivo_entrada = os.path.join(PROCESSED_DIR, "datos_economicos_1month_procesados.xlsx")
    archivo_salida = os.path.join(PROCESSED_DIR, "ULTIMO_S&P500_final.xlsx")
    
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

    # 2. Eliminar features constantes
    features, dropped_const = drop_constant_features(features)
    if dropped_const:
        logging.info(f"Features constantes eliminadas: {len(dropped_const)}")
        logging.info(f"Ejemplos: {dropped_const[:5]}...")

    # Guardar columna de fecha para reincorporar al final
    date_data = features.pop(DATE_COL) if DATE_COL in features.columns else None

    # 3. Obtener solo features numéricas
    numeric_features = features.select_dtypes(include=[np.number])
    logging.info(f"Features numéricas para análisis: {numeric_features.shape[1]}")

    # 4. Eliminar features correlacionadas
    features_corr, dropped_corr = remove_correlated_features(numeric_features, target)
    if dropped_corr:
        logging.info(f"Features correlacionadas eliminadas: {len(dropped_corr)}")
        logging.info(f"Ejemplos: {dropped_corr[:5]}...")

    # 5. Reducción por VIF
    features_vif, final_vif_df = iterative_vif_reduction(features_corr)
    logging.info(f"Features finales tras reducción VIF: {features_vif.shape[1]}")
    
    # Mostrar top 5 VIF más altos (pero bajo el umbral)
    top_vif = final_vif_df.sort_values('VIF', ascending=False).head(5)
    logging.info(f"Top VIF finales:\n{top_vif}")

    # 6. Reincorporar fecha y target
    if date_data is not None:
        features_vif.insert(0, DATE_COL, date_data)

    final_df = pd.concat([features_vif, target], axis=1)
    
    # 7. Guardar resultado
    final_df.to_excel(archivo_salida, index=False)
    logging.info(f"Dataset final guardado en: {archivo_salida}")
    logging.info(f"Dimensiones finales: {final_df.shape[0]} filas, {final_df.shape[1]} columnas")
    logging.info(f"Features finales: {final_df.shape[1] - (1 if date_data is not None else 0) - 1}")

if __name__ == "__main__":
    main()
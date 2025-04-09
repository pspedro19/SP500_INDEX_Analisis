import pandas as pd
import numpy as np
import re
import logging
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------
# CONFIGURACIÓN GLOBAL
# ------------------------------
CONSTANT_THRESHOLD = 0.98
CORR_THRESHOLD = 0.8
VIF_THRESHOLD = 10.0
DATE_COL = 'date'
TARGET_SUFFIX = '_Target'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_derived_target_features(df, target_col, date_col=DATE_COL):
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
    cols_to_drop = []
    for col in df.columns:
        if df[col].nunique() == 1 or df[col].value_counts(normalize=True, dropna=False).iloc[0] > threshold:
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop, errors='ignore'), cols_to_drop

def remove_correlated_features(df_features, target, threshold=CORR_THRESHOLD):
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
    df_clean = df_features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    non_const = [col for col in df_clean.columns if df_clean[col].std() > 0]
    df_clean = df_clean[non_const]
    return pd.DataFrame({
        'variable': df_clean.columns,
        'VIF': [variance_inflation_factor(df_clean.values, i) for i in range(df_clean.shape[1])]
    })

def iterative_vif_reduction(df_features, threshold=VIF_THRESHOLD):
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
    archivo = "Data/Macro/process/datos_economicos_1month_procesados.xlsx"
    df = pd.read_excel(archivo)

    target_col = df.columns[-1]

    df, _ = remove_derived_target_features(df, target_col)
    target = df[target_col]
    features = df.drop(columns=[target_col])

    features, _ = drop_constant_features(features)

    date_data = features.pop(DATE_COL) if DATE_COL in features.columns else None

    numeric_features = features.select_dtypes(include=[np.number])

    features_corr, _ = remove_correlated_features(numeric_features, target)
    features_vif, final_vif_df = iterative_vif_reduction(features_corr)

    if date_data is not None:
        features_vif.insert(0, DATE_COL, date_data)

    final_df = pd.concat([features_vif, target], axis=1)

    salida = "Data/Macro/Eliminar_relaciones/ULTIMO_S&P500_final.xlsx"
    final_df.to_excel(salida, index=False)
    logging.info(f"El dataset final se ha guardado en: '{salida}'.")

if __name__ == "__main__":
    main()
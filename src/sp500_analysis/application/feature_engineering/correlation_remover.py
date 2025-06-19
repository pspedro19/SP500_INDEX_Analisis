from __future__ import annotations

import logging
from typing import Iterable

try:  # optional dependencies for tests
    import pandas as pd
    import numpy as np
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tsa.stattools import adfuller
    from scipy.stats import entropy
except Exception:  # pragma: no cover - optional
    # Re-intentar imports críticos
    try:
        import pandas as pd
        import numpy as np
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tsa.stattools import adfuller
        from scipy.stats import entropy
    except ImportError:
        raise ImportError("pandas, numpy, statsmodels y scipy son requeridos para correlation_remover")


DEFAULT_DATE_COL = "date"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def detect_feature_frequency(df: pd.DataFrame, column: str, date_col: str = DEFAULT_DATE_COL) -> str:
    """
    Detecta la frecuencia de actualización de una variable.
    
    Args:
        df (DataFrame): DataFrame con la serie temporal
        column (str): Nombre de la columna a analizar
        date_col (str): Nombre de la columna de fecha
        
    Returns:
        str: Frecuencia detectada ('D', 'W', 'M', 'Q')
    """
    if column == date_col:
        return "D"  # La columna de fecha es diaria por definición
        
    df_sorted = df.sort_values(date_col)
    
    # Contar cambios (diferencias no nulas)
    changes = (df_sorted[column] != df_sorted[column].shift(1)).sum()
    total = len(df_sorted)
    
    if total == 0:
        return "D"  # Si no hay datos, asumir diario
        
    change_ratio = changes / total
    
    # Clasificar por frecuencia estimada
    if change_ratio > 0.3:
        return "D"  # Diario/alta frecuencia
    elif change_ratio > 0.1:
        return "W"  # Semanal
    elif change_ratio > 0.03:
        return "M"  # Mensual
    else:
        return "Q"  # Trimestral o menor frecuencia


def remove_derived_target_features(df: pd.DataFrame, target_col: str, date_col: str = DEFAULT_DATE_COL) -> tuple[pd.DataFrame, list[str]]:
    """
    Elimina features derivadas del target para evitar leakage.
    
    Args:
        df (DataFrame): DataFrame con features y target
        target_col (str): Nombre de la columna target
        date_col (str): Nombre de la columna de fecha
        
    Returns:
        tuple: (DataFrame sin columnas derivadas, lista de columnas eliminadas)
    """
    import re
    
    cols_to_drop = []
    TARGET_SUFFIX = "_12m_growth"  # Definir localmente para compatibilidad
    
    if target_col.endswith(TARGET_SUFFIX):
        base = target_col[:-len(TARGET_SUFFIX)]
        pattern = rf'^(log_diff|momentum)_{re.escape(base)}'
        for col in df.columns:
            if col in [target_col, date_col]:
                continue
            if re.search(pattern, col):
                cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop, errors='ignore'), cols_to_drop


def analyze_lags_with_target(df: pd.DataFrame, target_col: str, date_col: str = DEFAULT_DATE_COL, max_lag: int = 10) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Feature selector
# ---------------------------------------------------------------------------


class FeatureSelector:
    """Utility class for feature filtering operations."""

    def __init__(self, date_col: str = DEFAULT_DATE_COL, constant_threshold: float = 0.95) -> None:
        self.date_col = date_col
        self.constant_threshold = constant_threshold

    def drop_constant_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Return dataframe without near constant columns and the list removed."""
        return drop_constant_features(df, threshold=self.constant_threshold, date_col=self.date_col)

    def test_stationarity(self, df: pd.DataFrame, significance: float = 0.05) -> pd.DataFrame:
        """Run Dickey-Fuller tests and return a result DataFrame."""
        return test_feature_stationarity(df, date_col=self.date_col, significance=significance)

    def remove_derived_target_features(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list[str]]:
        """Remove features derived from target to avoid leakage."""
        return remove_derived_target_features(df, target_col, self.date_col)

    def analyze_lags_with_target(self, df: pd.DataFrame, target_col: str, max_lag: int = 10) -> pd.DataFrame:
        """Analyze correlations with lags between variables and target."""
        return analyze_lags_with_target(df, target_col, self.date_col, max_lag)


# ---------------------------------------------------------------------------
# Core functions extracted from the old script
# ---------------------------------------------------------------------------


def drop_constant_features(
    df: pd.DataFrame, threshold: float = 0.95, date_col: str = DEFAULT_DATE_COL
) -> tuple[pd.DataFrame, list[str]]:
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
    cols_to_drop: list[str] = []
    
    # Definir umbrales ajustados por frecuencia
    freq_thresholds = {
        "D": threshold,        # Diario - umbral estándar
        "W": threshold * 1.2,  # Semanal - más tolerante
        "M": threshold * 1.5,  # Mensual - mucho más tolerante
        "Q": threshold * 2.0   # Trimestral - extremadamente tolerante
    }
    
    # Información detallada de eliminaciones
    freq_counts = {"D": 0, "W": 0, "M": 0, "Q": 0}
    columns_by_freq = {"D": [], "W": [], "M": [], "Q": []}
    dropped_by_freq = {"D": [], "W": [], "M": [], "Q": []}
    
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
                is_constant = cv < (0.01 * (1 + ["D", "W", "M", "Q"].index(freq)))
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
                entropy_threshold = 0.1 * (1 + ["D", "W", "M", "Q"].index(freq) * 0.5)
                is_constant = norm_entropy < entropy_threshold
        
        if is_constant:
            cols_to_drop.append(col)
            dropped_by_freq[freq].append(col)
    
    # Loguear información sobre la distribución de frecuencias
    for freq in ["D", "W", "M", "Q"]:
        freq_counts[freq] = len(columns_by_freq[freq])
        dropped_count = len(dropped_by_freq[freq])
        if freq_counts[freq] > 0:
            drop_ratio = dropped_count / freq_counts[freq] * 100
            freq_name = {"D": "diarias", "W": "semanales", "M": "mensuales", "Q": "trimestrales"}[freq]
            logging.info(f"Variables {freq}({freq_name}): "
                         f"{freq_counts[freq]} totales, {dropped_count} eliminadas ({drop_ratio:.1f}%)")
    
    return df.drop(columns=cols_to_drop, errors="ignore"), cols_to_drop


def test_feature_stationarity(
    df: pd.DataFrame, date_col: str = DEFAULT_DATE_COL, significance: float = 0.05
) -> pd.DataFrame:
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
    
    results: list[dict[str, object]] = []
    
    for col in numeric_cols:
        # Eliminar valores nulos para el test
        series = df_sorted[col].dropna()
        
        if len(series) < 20:  # Necesitamos suficientes observaciones
            results.append({
                "column": col,
                "is_stationary": None,
                "p_value": None,
                "message": "Insuficientes datos"
            })
            continue
        
        # Test de Dickey-Fuller
        try:
            result = adfuller(series, autolag="AIC")
            p_value = result[1]
            is_stationary = p_value < significance
            
            results.append({
                "column": col,
                "is_stationary": is_stationary,
                "p_value": p_value,
                "message": "OK"
            })
        except Exception as exc:  # pragma: no cover - runtime behaviour
            results.append({
                "column": col,
                "is_stationary": None,
                "p_value": None,
                "message": str(exc)
            })
    
    return pd.DataFrame(results)


def remove_correlated_features_by_frequency(
    df_features: pd.DataFrame,
    target: pd.Series,
    *,
    date_col: str = DEFAULT_DATE_COL,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, Iterable[str]]:
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
    freq_groups: dict[str, list[str]] = {"D": [], "W": [], "M": [], "Q": []}
    
    for col in df_features.columns:
        if col == date_col:
            continue
        freq = detect_feature_frequency(df_features, col, date_col)
        freq_groups[freq].append(col)
    
    # Umbrales ajustados por frecuencia
    freq_thresholds = {
        "D": threshold,
        "W": threshold * 1.05,
        "M": threshold * 1.1,
        "Q": threshold * 1.15,
    }
    
    all_cols_to_drop: list[str] = []
    
    # Para cada grupo de frecuencia, aplicar análisis de correlación
    for freq, cols in freq_groups.items():
        if len(cols) <= 1:
            continue
            
        df_group = df_features[cols]
        corr_matrix = df_group.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        
        # Calcular correlación con target para cada columna del grupo
        target_corr: dict[str, float] = {}
        for col in cols:
            if pd.api.types.is_numeric_dtype(target):
                # Calcular correlación con diferentes lags (0 a 5) y usar la máxima
                max_corr = 0.0
                for lag in range(6):
                    if lag == 0:
                        corr = abs(df_features[col].corr(target))
                    else:
                        corr = abs(df_features[col].corr(target.shift(-lag)))
                    max_corr = max(max_corr, corr)
                target_corr[col] = max_corr
            else:
                target_corr[col] = 0.0

        # Buscar variables correlacionadas y decidir cuál mantener
        cols_to_drop: set[str] = set()
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
        
        all_cols_to_drop.extend(list(cols_to_drop))
        logging.info("Variables %s correlacionadas eliminadas: %d de %d", freq, len(cols_to_drop), len(cols))
    
    return df_features.drop(columns=all_cols_to_drop, errors="ignore"), all_cols_to_drop


def compute_vif_by_frequency(df_features: pd.DataFrame, date_col: str = DEFAULT_DATE_COL) -> pd.DataFrame:
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
        return pd.DataFrame(columns=["variable", "VIF", "frequency"])
    
    # Calcular VIF
    vif_data: list[dict[str, object]] = []
    
    for i, column in enumerate(df_clean.columns):
        try:
            vif_value = variance_inflation_factor(df_clean.values, i)
            freq = detect_feature_frequency(df_features, column, date_col)
            vif_data.append({
                "variable": column,
                "VIF": float(vif_value),
                "frequency": freq
            })
        except Exception as exc:  # pragma: no cover - runtime behaviour
            logging.warning(f"Error al calcular VIF para '{column}': {str(exc)}")
    
    return pd.DataFrame(vif_data)


def iterative_vif_reduction_by_frequency(
    df_features: pd.DataFrame,
    *,
    date_col: str = DEFAULT_DATE_COL,
    threshold: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "D": threshold,
        "W": threshold * 1.2,
        "M": threshold * 1.5,
        "Q": threshold * 2.0,
    }
    
    # Límites de eliminación por frecuencia (porcentaje máximo que se puede eliminar)
    freq_limits = {
        "D": 0.7,  # Se puede eliminar hasta 70% de variables diarias
        "W": 0.5,  # Se puede eliminar hasta 50% de variables semanales
        "M": 0.3,  # Se puede eliminar hasta 30% de variables mensuales
        "Q": 0.2   # Se puede eliminar hasta 20% de variables trimestrales
    }
    
    # Contar variables por frecuencia
    freq_counts = {"D": 0, "W": 0, "M": 0, "Q": 0}
    freq_dropped = {"D": 0, "W": 0, "M": 0, "Q": 0}
    
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
            freq = row["frequency"]
            adjusted_threshold = freq_thresholds[freq]
            
            # Verificar si ya hemos alcanzado el límite de eliminaciones para esta frecuencia
            current_dropped = freq_dropped[freq]
            max_to_drop = int(freq_counts[freq] * freq_limits[freq])
            
            if row["VIF"] > adjusted_threshold and current_dropped < max_to_drop:
                if row["VIF"] > max_vif:
                    max_vif = row["VIF"]
                    var_to_drop = row["variable"]
                    has_high_vif = True
        
        if not has_high_vif or var_to_drop is None:
            break
            
        # Eliminar la variable con el VIF más alto
        var_freq = detect_feature_frequency(df_reduced, var_to_drop, date_col)
        logging.info(f"Eliminando '{var_to_drop}' ({var_freq}) por VIF = {max_vif:.2f}")
        df_reduced = df_reduced.drop(columns=[var_to_drop])
        freq_dropped[var_freq] += 1
        
        # Parar si quedan muy pocas columnas
        if len(df_reduced.columns) <= (1 if date_col not in df_reduced.columns else 2):
            break
    
    # Loguear estadísticas
    for freq in ["D", "W", "M", "Q"]:
        if freq_counts[freq] > 0:
            drop_pct = freq_dropped[freq] / freq_counts[freq] * 100
            logging.info(f"Variables {freq}: {freq_counts[freq]} iniciales, "
                        f"{freq_dropped[freq]} eliminadas ({drop_pct:.1f}%), "
                        f"{freq_counts[freq] - freq_dropped[freq]} mantenidas")
    
    return df_reduced, compute_vif_by_frequency(df_reduced, date_col)

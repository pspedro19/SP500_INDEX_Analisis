from __future__ import annotations

"""Feature selection utilities based on permutation importance."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - optional dependencies
    import pandas as pd
    import numpy as np
    from catboost import CatBoostRegressor
    from feature_engine.selection import SelectByShuffling
    from sklearn.model_selection import TimeSeriesSplit
    from statsmodels.tsa.stattools import adfuller
    from scipy.stats import entropy
except Exception:  # pragma: no cover - optional
    # Re-intentar imports críticos
    try:
        import pandas as pd
        import numpy as np
        from catboost import CatBoostRegressor
        from feature_engine.selection import SelectByShuffling
        from sklearn.model_selection import TimeSeriesSplit
        from statsmodels.tsa.stattools import adfuller
        from scipy.stats import entropy
    except ImportError:
        raise ImportError("pandas, numpy, catboost, feature_engine, sklearn, statsmodels y scipy son requeridos para fpi_selection")

__all__ = [
    "get_most_recent_file",
    "plot_cv_splits",
    "plot_performance_drift",
    "select_features_fpi",
    "create_feature_summary_report",
    "detect_feature_frequency",
    "analyze_lags_with_target",
    "test_feature_stationarity",
]

DEFAULT_DATE_COL = "date"


def detect_feature_frequency(df: pd.DataFrame, col: str, date_col: str = DEFAULT_DATE_COL) -> str:
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


def test_feature_stationarity(df: pd.DataFrame, date_col: str = DEFAULT_DATE_COL, significance: float = 0.05) -> pd.DataFrame:
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


def create_feature_summary_report(df: pd.DataFrame, target_col: str, date_col: str = DEFAULT_DATE_COL, output_file: str | None = None) -> pd.DataFrame:
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


def get_most_recent_file(directory: str, extension: str = ".xlsx") -> str | None:
    """Return the most recent file with ``extension`` in ``directory``."""
    files = list(Path(directory).glob(f"*{extension}"))
    if not files:
        return None
    return str(max(files, key=lambda p: p.stat().st_mtime))


def plot_cv_splits(
    X: "pd.DataFrame",
    tscv: TimeSeriesSplit,
    output_path: str | Path,
    *,
    cv_splits: int,
    gap: int,
) -> None:
    """Save a visualisation of the ``tscv`` splits used."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 5))
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        ax.scatter(train_idx, [i + 0.5] * len(train_idx), c="blue", marker="_", s=40, label="Train" if i == 0 else "")
        ax.scatter(val_idx, [i + 0.5] * len(val_idx), c="red", marker="_", s=40, label="Validation" if i == 0 else "")
        ax.text(
            X.shape[0] + 5,
            i + 0.5,
            f"Split {i+1}: {len(train_idx)} train, {len(val_idx)} val",
            va="center",
            ha="left",
        )

    ax.legend(loc="upper right")
    ax.set_xlabel("Índice de muestra")
    ax.set_yticks(range(1, cv_splits + 1))
    ax.set_yticklabels([f"Split {i+1}" for i in range(cv_splits)])
    ax.set_title(f"Validación Cruzada Temporal (CV_SPLITS={cv_splits}, GAP={gap})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info("Gráfico de CV splits guardado en: %s", output_path)


def plot_performance_drift(
    features: Iterable[str],
    drifts: Iterable[float] | dict[str, float],
    selected: Iterable[str],
    threshold: float,
    output_path: str | Path,
) -> None:
    """Save a bar plot of the drift scores for ``features``."""
    import matplotlib.pyplot as plt

    if isinstance(drifts, dict):
        drifts_list = [drifts.get(f, 0) for f in features]
    else:
        drifts_list = list(drifts)

    df = pd.DataFrame({"feature": list(features), "drift": drifts_list})
    df["selected"] = df["feature"].isin(list(selected))
    df = df.sort_values("drift", ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(8, len(features) / 5)))
    colors = ["green" if sel else "red" for sel in df["selected"]]
    bars = ax.barh(df["feature"], df["drift"], color=colors)
    ax.axvline(x=threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.4f})")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", va="center", ha="left")

    ax.legend()
    ax.set_xlabel("Performance Drift")
    ax.set_ylabel("Feature")
    ax.set_title("Performance Drift por Feature (FPI)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info("Gráfico de performance drift guardado en: %s", output_path)


def select_features_fpi(
    X: "pd.DataFrame",
    y: "pd.Series",
    *,
    cv_splits: int,
    gap: int,
    threshold: float,
    catboost_params: dict | None = None,
    scorer=None,
    plots_dir: str | Path | None = None,
    timestamp: str | None = None,
) -> tuple[list[str], list[float]]:
    """Run permutation importance based feature selection using CatBoost."""
    if pd is None or np is None:  # pragma: no cover - optional deps
        raise ImportError("pandas and numpy are required for select_features_fpi")

    start_time = time.time()
    logging.info("=" * 50)
    logging.info("[FPI] INICIANDO ANÁLISIS DE FEATURE PERMUTATION IMPORTANCE")
    logging.info("=" * 50)
    logging.info("[FPI] Dimensiones de datos - X: %s, y: %s", X.shape, y.shape)
    logging.info(
        "[FPI] Parámetros - CV splits: %s, gap: %s, threshold: %s",
        cv_splits,
        gap,
        threshold,
    )

    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=gap)

    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        cv_plot = plots_dir / f"cv_splits_{ts}.png"
        plot_cv_splits(X, tscv, cv_plot, cv_splits=cv_splits, gap=gap)
    else:
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    params = catboost_params or {"iterations": 100, "verbose": False}
    regressor = CatBoostRegressor(**params)
    selector = SelectByShuffling(estimator=regressor, scoring=scorer, cv=tscv, threshold=threshold)

    selector.fit(X, y)

    selected_features = list(selector.get_feature_names_out())
    performance_drifts = selector.performance_drifts_

    if plots_dir is not None:
        drift_csv = plots_dir / f"fpi_drifts_{ts}.csv"
        drift_df = pd.DataFrame({
            "feature": list(X.columns), 
            "performance_drift": list(performance_drifts)
        })
        drift_df.to_csv(drift_csv, index=False)
        drift_plot = plots_dir / f"performance_drift_{ts}.png"
        plot_performance_drift(
            list(X.columns),
            performance_drifts,
            selected_features,
            threshold,
            drift_plot,
        )

    total_time = time.time() - start_time
    logging.info("[FPI] Proceso FPI completado en %.2f segundos", total_time)
    logging.info("[FPI] Número de features seleccionadas: %d", len(selected_features))

    return selected_features, performance_drifts

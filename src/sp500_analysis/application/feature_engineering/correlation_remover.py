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
    pd = None
    np = None


DEFAULT_DATE_COL = "date"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def detect_feature_frequency(df: pd.DataFrame, column: str, date_col: str = DEFAULT_DATE_COL) -> str:
    """Infer the update frequency for *column* in *df*."""
    if column == date_col:
        return "D"

    df_sorted = df.sort_values(date_col)
    changes = (df_sorted[column] != df_sorted[column].shift(1)).sum()
    total = len(df_sorted)
    if total == 0:
        return "D"

    ratio = changes / total
    if ratio > 0.3:
        return "D"
    if ratio > 0.1:
        return "W"
    if ratio > 0.03:
        return "M"
    return "Q"


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


# ---------------------------------------------------------------------------
# Core functions extracted from the old script
# ---------------------------------------------------------------------------


def drop_constant_features(
    df: pd.DataFrame, threshold: float = 0.95, date_col: str = DEFAULT_DATE_COL
) -> tuple[pd.DataFrame, list[str]]:
    cols_to_drop: list[str] = []
    freq_thresholds = {
        "D": threshold,
        "W": threshold * 1.2,
        "M": threshold * 1.5,
        "Q": threshold * 2.0,
    }
    for col in df.columns:
        if col == date_col:
            continue
        freq = detect_feature_frequency(df, col, date_col)
        adjusted_threshold = freq_thresholds[freq]
        is_constant = False
        if pd.api.types.is_numeric_dtype(df[col]):
            std_dev = df[col].std()
            mean_val = df[col].mean() if not pd.isna(df[col].mean()) else 1
            if mean_val != 0:
                cv = abs(std_dev / mean_val)
                is_constant = cv < (0.01 * (1 + ["D", "W", "M", "Q"].index(freq)))
            else:
                is_constant = std_dev < 1e-8
        else:
            value_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
            is_constant = value_freq > adjusted_threshold
            if not is_constant and len(df[col].unique()) > 1:
                counts = df[col].value_counts(dropna=False)
                norm_entropy = entropy(counts) / np.log(len(counts))
                entropy_threshold = 0.1 * (1 + ["D", "W", "M", "Q"].index(freq) * 0.5)
                is_constant = norm_entropy < entropy_threshold
        if is_constant:
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop, errors="ignore"), cols_to_drop


def test_feature_stationarity(
    df: pd.DataFrame, date_col: str = DEFAULT_DATE_COL, significance: float = 0.05
) -> pd.DataFrame:
    df_sorted = df.sort_values(date_col).copy()
    numeric_cols = [c for c in df_sorted.columns if pd.api.types.is_numeric_dtype(df_sorted[c]) and c != date_col]
    results: list[dict[str, object]] = []
    for col in numeric_cols:
        series = df_sorted[col].dropna()
        if len(series) < 20:
            results.append({"column": col, "is_stationary": None, "p_value": None, "message": "Insuficientes datos"})
            continue
        try:
            res = adfuller(series, autolag="AIC")
            p_value = res[1]
            results.append(
                {"column": col, "is_stationary": p_value < significance, "p_value": p_value, "message": "OK"}
            )
        except Exception as exc:  # pragma: no cover - runtime behaviour
            results.append({"column": col, "is_stationary": None, "p_value": None, "message": str(exc)})
    return pd.DataFrame(results)


def remove_correlated_features_by_frequency(
    df_features: pd.DataFrame,
    target: pd.Series,
    *,
    date_col: str = DEFAULT_DATE_COL,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, Iterable[str]]:
    freq_groups: dict[str, list[str]] = {"D": [], "W": [], "M": [], "Q": []}
    for col in df_features.columns:
        if col == date_col:
            continue
        freq = detect_feature_frequency(df_features, col, date_col)
        freq_groups[freq].append(col)

    freq_thresholds = {
        "D": threshold,
        "W": threshold * 1.05,
        "M": threshold * 1.1,
        "Q": threshold * 1.15,
    }
    all_cols_to_drop: list[str] = []
    for freq, cols in freq_groups.items():
        if len(cols) <= 1:
            continue
        df_group = df_features[cols]
        corr_matrix = df_group.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        target_corr: dict[str, float] = {}
        for col in cols:
            if pd.api.types.is_numeric_dtype(target):
                max_corr = 0.0
                for lag in range(6):
                    corr = df_features[col].corr(target.shift(-lag)) if lag else df_features[col].corr(target)
                    max_corr = max(max_corr, abs(corr))
                target_corr[col] = max_corr
            else:
                target_corr[col] = 0.0

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
                    cols_to_drop.add(cols[j])
        all_cols_to_drop.extend(list(cols_to_drop))
        logging.info("Variables %s correlacionadas eliminadas: %d de %d", freq, len(cols_to_drop), len(cols))
    return df_features.drop(columns=all_cols_to_drop, errors="ignore"), all_cols_to_drop


def compute_vif_by_frequency(df_features: pd.DataFrame, date_col: str = DEFAULT_DATE_COL) -> pd.DataFrame:
    df_clean = df_features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    if date_col in df_clean.columns:
        df_clean = df_clean.drop(columns=[date_col])
    non_const = [c for c in df_clean.columns if df_clean[c].std() > 0]
    df_clean = df_clean[non_const]
    if df_clean.shape[1] == 0:
        return pd.DataFrame(columns=["variable", "VIF", "frequency"])
    vif_data: list[dict[str, object]] = []
    for i, column in enumerate(df_clean.columns):
        try:
            vif_value = variance_inflation_factor(df_clean.values, i)
            freq = detect_feature_frequency(df_features, column, date_col)
            vif_data.append({"variable": column, "VIF": float(vif_value), "frequency": freq})
        except Exception as exc:  # pragma: no cover - runtime behaviour
            logging.warning("Error al calcular VIF para %s: %s", column, exc)
    return pd.DataFrame(vif_data)


def iterative_vif_reduction_by_frequency(
    df_features: pd.DataFrame,
    *,
    date_col: str = DEFAULT_DATE_COL,
    threshold: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_reduced = df_features.copy()
    freq_thresholds = {
        "D": threshold,
        "W": threshold * 1.2,
        "M": threshold * 1.5,
        "Q": threshold * 2.0,
    }
    freq_limits = {"D": 0.7, "W": 0.5, "M": 0.3, "Q": 0.2}
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
        has_high_vif = False
        var_to_drop: str | None = None
        max_vif = 0.0
        for _, row in vif_df.iterrows():
            freq = row["frequency"]
            adjusted_threshold = freq_thresholds[freq]
            current_dropped = freq_dropped[freq]
            max_to_drop = int(freq_counts[freq] * freq_limits[freq])
            if row["VIF"] > adjusted_threshold and current_dropped < max_to_drop:
                if row["VIF"] > max_vif:
                    max_vif = row["VIF"]
                    var_to_drop = row["variable"]
                    has_high_vif = True
        if not has_high_vif or var_to_drop is None:
            break
        var_freq = detect_feature_frequency(df_reduced, var_to_drop, date_col)
        logging.info("Eliminando %s (%s) por VIF=%.2f", var_to_drop, var_freq, max_vif)
        df_reduced.drop(columns=[var_to_drop], inplace=True)
        freq_dropped[var_freq] += 1
        if len(df_reduced.columns) <= (1 if date_col not in df_reduced.columns else 2):
            break
    for freq in ["D", "W", "M", "Q"]:
        if freq_counts[freq] > 0:
            drop_pct = freq_dropped[freq] / freq_counts[freq] * 100
            logging.info(
                "Variables %s: %d iniciales, %d eliminadas (%.1f%%), %d mantenidas",
                freq,
                freq_counts[freq],
                freq_dropped[freq],
                drop_pct,
                freq_counts[freq] - freq_dropped[freq],
            )
    return df_reduced, compute_vif_by_frequency(df_reduced, date_col)

from __future__ import annotations

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional
    pd = None
try:
    import numpy as np
except Exception:  # pragma: no cover - optional
    np = None
try:
    from ta.trend import ema_indicator, sma_indicator
except Exception:  # pragma: no cover - optional
    ema_indicator = lambda s, span: s.ewm(span=span, adjust=False).mean() if pd is not None else None
    sma_indicator = lambda s, window: s.rolling(window).mean() if pd is not None else None

from .lag import FORECAST_HORIZON_1MONTH, FORECAST_HORIZON_3MONTHS


def add_log_and_diff_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df[f"log_{target_column}"] = np.log(df[target_column] + 1)
    df[f"log_diff_{target_column}"] = df[f"log_{target_column}"] - df[f"log_{target_column}"].shift(1)
    return df


def add_volatility_features(df: pd.DataFrame, target_column: str, window: int = FORECAST_HORIZON_1MONTH) -> pd.DataFrame:
    df[f"rolling_std_{target_column}"] = df[target_column].rolling(window).std()
    df[f"rolling_var_{target_column}"] = df[target_column].rolling(window).var()
    ma = df[target_column].rolling(window).mean()
    std = df[target_column].rolling(window).std()
    df[f"bollinger_upper_{target_column}"] = ma + 2 * std
    df[f"bollinger_lower_{target_column}"] = ma - 2 * std
    return df


def add_rsi(df: pd.DataFrame, target_column: str, window: int = 14) -> pd.DataFrame:
    delta = df[target_column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    df[f"RSI_{target_column}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, target_column: str, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df[target_column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[target_column].ewm(span=slow, adjust=False).mean()
    df[f"MACD_{target_column}"] = ema_fast - ema_slow
    df[f"MACD_signal_{target_column}"] = df[f"MACD_{target_column}"].ewm(span=signal, adjust=False).mean()
    return df


def add_momentum(df: pd.DataFrame, target_column: str, window: int = 10) -> pd.DataFrame:
    df[f"momentum_{target_column}"] = df[target_column] - df[target_column].shift(window)
    return df


def add_intermediate_changes(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df[f"3M_change_{target_column}"] = (df[target_column] / df[target_column].shift(FORECAST_HORIZON_3MONTHS)) - 1
    df[f"6M_change_{target_column}"] = (df[target_column] / df[target_column].shift(FORECAST_HORIZON_3MONTHS * 2)) - 1
    return df


def add_ytd_performance(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df["year"] = pd.to_datetime(df["date"]).dt.year
    first_day = df.groupby("year")[target_column].transform("first")
    df[f"YTD_{target_column}"] = (df[target_column] / first_day) - 1
    df.drop(columns=["year"], inplace=True)
    return df


def add_zscore(df: pd.DataFrame, target_column: str, window: int = 60) -> pd.DataFrame:
    rolling_mean = df[target_column].rolling(window).mean()
    rolling_std = df[target_column].rolling(window).std()
    rolling_std_safe = rolling_std.replace(0, np.nan)
    df[f"zscore_{target_column}"] = (df[target_column] - rolling_mean) / rolling_std_safe
    return df


def add_minmax_scaling(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    min_val = df[target_column].min()
    max_val = df[target_column].max()
    df[f"minmax_{target_column}"] = (df[target_column] - min_val) / (max_val - min_val)
    return df


def transform_bond(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df[f"MoM_{target_column}"] = (df[target_column] / df[target_column].shift(FORECAST_HORIZON_1MONTH)) - 1
    df[f"YoY_{target_column}"] = (df[target_column] / df[target_column].shift(252)) - 1
    df[f"MA_60_{target_column}"] = sma_indicator(df[target_column], 60)
    df[f"EMA_60_{target_column}"] = ema_indicator(df[target_column], 60)
    df = add_log_and_diff_features(df, target_column)
    df = add_volatility_features(df, target_column)
    df = add_intermediate_changes(df, target_column)
    df = add_ytd_performance(df, target_column)
    df = add_zscore(df, target_column)
    df = add_minmax_scaling(df, target_column)
    return df


def transform_exchange_rate(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df[f"EMA_5_{target_column}"] = ema_indicator(df[target_column], 5)
    df[f"EMA_10_{target_column}"] = ema_indicator(df[target_column], 10)
    df[f"EMA_20_{target_column}"] = ema_indicator(df[target_column], FORECAST_HORIZON_1MONTH)
    df[f"MA_5_{target_column}"] = sma_indicator(df[target_column], 5)
    df[f"MA_10_{target_column}"] = sma_indicator(df[target_column], 10)
    df[f"MA_20_{target_column}"] = sma_indicator(df[target_column], FORECAST_HORIZON_1MONTH)
    df = add_log_and_diff_features(df, target_column)
    df = add_volatility_features(df, target_column)
    df = add_rsi(df, target_column)
    df = add_macd(df, target_column)
    df = add_momentum(df, target_column)
    df = add_intermediate_changes(df, target_column)
    df = add_ytd_performance(df, target_column)
    df = add_zscore(df, target_column)
    df = add_minmax_scaling(df, target_column)
    return df



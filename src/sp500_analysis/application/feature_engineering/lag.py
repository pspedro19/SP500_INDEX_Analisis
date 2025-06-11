from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:
    pd = None

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:
    np = None

# Forecast horizons
FORECAST_HORIZON_1MONTH = 20  # business days
FORECAST_HORIZON_3MONTHS = 60


def configure_target_variable(
    df: pd.DataFrame,
    target_column: str,
    *,
    use_lag: bool = True,
    lag_days: int = FORECAST_HORIZON_1MONTH,
    lag_type: str = "future",
):
    if pd is None:
        raise ImportError("pandas is required")
    if np is None:
        raise ImportError("numpy is required")

    if not use_lag:
        df[f"{target_column}_Target"] = df[target_column]
        df[f"{target_column}_Return_Target"] = df[target_column].pct_change().fillna(0)
        return df

    if lag_type not in {"future", "past", "current"}:
        raise ValueError("lag_type must be 'future', 'past' or 'current'")

    shift_value = -lag_days if lag_type == "future" else lag_days if lag_type == "past" else 0

    df[f"{target_column}_Target"] = df[target_column].shift(shift_value)
    if shift_value != 0:
        df[f"{target_column}_Return_Target"] = (df[f"{target_column}_Target"] / df[target_column]) - 1
    else:
        df[f"{target_column}_Return_Target"] = 0
    return df

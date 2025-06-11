from __future__ import annotations

"""Utilities for filtering economic data to the last n days."""

from typing import TYPE_CHECKING

try:  # pragma: no cover - optional pandas dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas may be unavailable
    pd = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import pandas as pd

__all__ = ["filter_last_n_days"]


def filter_last_n_days(
    df: "pd.DataFrame",
    reference_df: "pd.DataFrame",
    *,
    days: int = 20,
    date_col: str = "date",
    target_suffix: str = "_Target",
) -> "pd.DataFrame":
    """Return ``df`` filtered to the last *days* rows based on *date_col*.

    Only columns present in ``reference_df`` are kept and any column ending with
    ``target_suffix`` is removed. Rows are returned oldest to newest.
    """
    if pd is None:  # pragma: no cover - pandas optional
        raise ImportError("pandas is required for filter_last_n_days")

    columnas_comunes = [
        col
        for col in df.columns
        if col in reference_df.columns and not col.endswith(target_suffix)
    ]
    filtrado = df[columnas_comunes]

    if date_col in filtrado.columns:
        if not pd.api.types.is_datetime64_any_dtype(filtrado[date_col]):
            filtrado[date_col] = pd.to_datetime(filtrado[date_col], errors="coerce")
        filtrado = filtrado.sort_values(by=date_col, ascending=False).head(days)
        filtrado = filtrado.sort_values(by=date_col, ascending=True).reset_index(drop=True)
    else:
        filtrado = filtrado.tail(days)
        filtrado = filtrado.iloc[::-1].reset_index(drop=True)

    return filtrado

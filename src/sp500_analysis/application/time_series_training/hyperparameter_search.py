from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def search_arima_order(
    series: pd.Series,
    p_values: Iterable[int] = (0, 1),
    d_values: Iterable[int] = (0, 1),
    q_values: Iterable[int] = (0, 1),
) -> Tuple[int, int, int]:
    """Return the (p, d, q) combination with the lowest AIC."""
    best_order = (0, 0, 0)
    best_aic = np.inf
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(series, order=(p, d, q)).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_order = (p, d, q)
                except Exception:
                    # Skip invalid parameter sets
                    continue
    return best_order

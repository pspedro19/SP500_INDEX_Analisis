from __future__ import annotations

from typing import Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def fit_model(
    train_series: pd.Series,
    test_series: pd.Series,
    order: Tuple[int, int, int],
) -> Tuple[ARIMA, pd.Series]:
    """Fit an ARIMA model and return fitted model and forecasts for the test set."""
    model = ARIMA(train_series, order=order).fit()
    preds = model.forecast(len(test_series))
    preds.index = test_series.index
    return model, preds

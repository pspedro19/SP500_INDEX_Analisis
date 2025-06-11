from __future__ import annotations

import logging
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

from .frequency import infer_frequencies, convert_dataframe, impute_time_series_ffill, resample_to_business_day
from .lag import configure_target_variable, FORECAST_HORIZON_1MONTH
from .indicators import transform_bond, transform_exchange_rate


class FeatureEngineeringService:
    """Service that performs feature engineering for the economic dataset."""

    def __init__(self, project_root: Path | str | None = None) -> None:
        self.project_root = Path(project_root or Path.cwd())

    def run(
        self,
        input_file: Path | str,
        output_file: Path | str,
        target_column: str,
        *,
        use_lag: bool = True,
        lag_days: int = FORECAST_HORIZON_1MONTH,
        lag_type: str = "future",
    ) -> Path:
        if pd is None:
            raise ImportError("pandas is required for feature engineering")

        df = pd.read_excel(input_file)
        df = convert_dataframe(df, excluded_column=None)
        df = impute_time_series_ffill(df)
        df = resample_to_business_day(df, input_frequency="D")

        freqs = infer_frequencies(df)

        transformed = []
        for col in df.columns:
            if col in {"date", "id"}:
                continue
            if freqs.get(col, "D") != "D":
                transformed.append(df[["date", "id", col]].copy())
                continue
            temp = df[["date", "id", col]].copy()
            if "bond" in col.lower():
                temp = transform_bond(temp, col)
            else:
                temp = transform_exchange_rate(temp, col)
            transformed.append(temp)

        final_df = transformed[0]
        for tdf in transformed[1:]:
            final_df = final_df.merge(tdf, on=["date", "id"], how="outer")

        final_df = configure_target_variable(
            final_df,
            target_column,
            use_lag=use_lag,
            lag_days=lag_days,
            lag_type=lag_type,
        )

        final_df.to_excel(output_file, index=False)
        logging.info("Features saved to %s", output_file)
        return Path(output_file)

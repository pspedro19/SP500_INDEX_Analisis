from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sp500_analysis.shared.visualization.plotters import plot_real_vs_pred


@dataclass
class Backtester:
    """Utility class to evaluate model predictions."""

    results_dir: Path
    metrics_dir: Path
    charts_dir: Path
    subperiods_dir: Path
    date_col: str = "date"

    def _get_most_recent_file(self, pattern: str = "*powerbi*.csv") -> Path | None:
        files = list(self.results_dir.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _smape(real: Iterable[float], pred: Iterable[float]) -> float:
        real = np.array(real)
        pred = np.array(pred)
        return 100 * np.mean(2 * np.abs(pred - real) / (np.abs(real) + np.abs(pred)))

    @classmethod
    def _all_metrics(cls, real: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        mae = mean_absolute_error(real, pred)
        mse = mean_squared_error(real, pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((real - pred) / real)) * 100
        smape = cls._smape(real, pred)
        r2 = r2_score(real, pred)
        r2_adj = 1 - ((1 - r2) * (len(real) - 1)) / (len(real) - 2)
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "SMAPE": smape,
            "R2": r2,
            "R2_adjusted": r2_adj,
        }

    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep=";")
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df = df.dropna(subset=[self.date_col])
        df = df.sort_values(self.date_col)
        for col in ["Valor_Real", "Valor_Predicho", "RMSE"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
        return df

    def run(self) -> None:
        """Execute the backtest using the latest predictions file."""
        file_path = self._get_most_recent_file()
        if not file_path:
            logging.error("No prediction file found in %s", self.results_dir)
            return

        df = self._load_dataframe(file_path)
        results: list[dict[str, float]] = []

        for (modelo, mercado), group in df.groupby(["Modelo", "Tipo_Mercado"]):
            group = group.dropna(subset=["Valor_Real", "Valor_Predicho"])
            real = group["Valor_Real"].values
            pred = group["Valor_Predicho"].values
            if len(real) == 0:
                continue

            metrics = self._all_metrics(real, pred)
            results.append({"Tipo_de_Mercado": mercado, "Modelo": modelo, **metrics})

            chart_name = f"{mercado.replace(' ', '_')}_{modelo.replace(' ', '_')}_comparison.png"
            chart_path = self.charts_dir / chart_name
            plot_real_vs_pred(
                group,
                title=f"Comparaci\u00f3n {mercado} - {modelo}",
                metrics=metrics,
                model_name=modelo,
                output_path=str(chart_path),
            )

        if results:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            df_res = pd.DataFrame(results)
            df_res.to_csv(self.metrics_dir / "resultados_totales.csv", index=False)
            df_res.to_json(self.metrics_dir / "resultados_totales.json", orient="records", indent=4)
            logging.info("Metrics exported to %s", self.metrics_dir)
        else:
            logging.warning("No results generated")


def run_backtest(
    *,
    results_dir: str | Path,
    metrics_dir: str | Path,
    charts_dir: str | Path,
    subperiods_dir: str | Path,
    date_col: str = "date",
) -> None:
    """Convenience wrapper used by CLI and pipeline scripts."""
    tester = Backtester(
        Path(results_dir),
        Path(metrics_dir),
        Path(charts_dir),
        Path(subperiods_dir),
        date_col,
    )
    tester.run()

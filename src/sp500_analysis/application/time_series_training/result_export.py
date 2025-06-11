from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_forecast(preds: pd.Series, output_file: str | Path) -> Path:
    """Save the forecast series to CSV and return the path."""
    df = pd.DataFrame({"date": preds.index, "forecast": preds.values})
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path

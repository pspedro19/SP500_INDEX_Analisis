from __future__ import annotations

import os
from typing import Optional
import pandas as pd

from sp500_analysis.config.settings import settings


class DataRepository:
    """Simple repository to load training data from disk."""

    def __init__(self, training_dir: str | os.PathLike = settings.training_dir) -> None:
        self.training_dir = str(training_dir)

    def load_latest(self) -> Optional[pd.DataFrame]:
        files = [f for f in os.listdir(self.training_dir) if f.endswith(".xlsx")]
        if not files:
            return None
        paths = [os.path.join(self.training_dir, f) for f in files]
        latest = max(paths, key=os.path.getmtime)
        return pd.read_excel(latest)

"""Shared helpers for visualization modules."""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt

# Global style configuration for consistency
plt.style.use("seaborn-v0_8-whitegrid")

COLORS: Dict[str, str] = {
    "real": "#1f77b4",
    "predicted": "#ff7f0e",
    "training": "#2ca02c",
    "validation": "#d62728",
    "test": "#9467bd",
    "forecast": "#8c564b",
    "catboost": "#e377c2",
    "lightgbm": "#7f7f7f",
    "xgboost": "#bcbd22",
    "mlp": "#17becf",
    "svm": "#1f77b4",
    "ensemble": "#ff7f0e",
}


def create_directory(directory: str) -> None:
    """Create the given directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

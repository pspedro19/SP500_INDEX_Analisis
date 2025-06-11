"""Visualization helpers for importance and correlation charts."""

from __future__ import annotations

from typing import Sequence
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from .common import COLORS, create_directory


def plot_feature_importance(
    importances: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
    *,
    title: str | None = None,
    top_n: int = 20,
    model_name: str | None = None,
    output_path: str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> Figure:
    """Visualize feature importances."""
    if len(importances) != len(feature_names):
        raise ValueError("Las longitudes de importances y feature_names deben coincidir")

    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(top_indices)), np.asarray(importances)[top_indices], align="center")
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([feature_names[i] for i in top_indices])

    if title:
        plt.title(title)
    else:
        titulo = "Importancia de Features"
        if model_name:
            titulo += f" - {model_name}"
        plt.title(titulo)

    plt.xlabel("Importancia")
    plt.tight_layout()

    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    title: str | None = None,
    target_col: str | None = None,
    threshold: float = 0.7,
    figsize: tuple[int, int] = (14, 12),
    output_path: str | None = None,
) -> Figure:
    """Plot a correlation matrix heatmap."""
    corr_matrix = df.corr(method=method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )

    if target_col and target_col in corr_matrix.columns:
        target_corrs = corr_matrix[target_col].abs().sort_values(ascending=False)
        high_corrs = target_corrs[target_corrs > threshold].index.tolist()
        high_corrs_text = "\n".join(
            [f"{col}: {corr_matrix.loc[col, target_col]:.3f}" for col in high_corrs if col != target_col]
        )
        if high_corrs_text:
            plt.figtext(
                0.01,
                0.01,
                f"Correlaciones altas con {target_col}:\n{high_corrs_text}",
                fontsize=10,
                ha="left",
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
            )

    plt.title(title or f"Matriz de Correlación ({method})")
    plt.tight_layout()

    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig

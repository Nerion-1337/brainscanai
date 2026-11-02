"""Visualisations réutilisables (échantillon d'images, embeddings 2D, perf)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

sns.set_theme(style="whitegrid", context="notebook")


def save_fig(fig: plt.Figure, name: str, out_dir: str | Path | None = None, dpi: int = 150) -> Path | None:
    """Sauvegarde une figure en PNG si ``out_dir`` est fourni."""
    if out_dir is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_sample_grid(paths, titles=None, ncols: int = 5, figsize_per: float = 2.6) -> plt.Figure:
    """Grille d'images d'exemple."""
    n = len(paths)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_per, nrows * figsize_per))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, p in enumerate(paths):
        axes[i].imshow(Image.open(p).convert("L"), cmap="gray")
        if titles is not None:
            axes[i].set_title(str(titles[i]), fontsize=9)
    fig.tight_layout()
    return fig


def plot_embedding_2d(coords: np.ndarray, labels, title: str = "", ax=None, palette="tab10",
                      alpha: float = 0.7, s: int = 18) -> plt.Axes:
    """Nuage de points 2D coloré par ``labels`` (clusters ou classes)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    labels = np.asarray(labels).astype(str)
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette=palette,
                    alpha=alpha, s=s, ax=ax, edgecolor=None)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(title="", fontsize=8, loc="best")
    return ax


def plot_confusion_matrix(cm: np.ndarray, classes, title: str = "Matrice de confusion", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    return ax


def plot_roc_curves(curves: dict, ax=None):
    """``curves`` = {nom: (fpr, tpr, auc)}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs (rappel)")
    ax.set_title("Courbes ROC")
    ax.legend(fontsize=8)
    return ax


def plot_metric_comparison(df: pd.DataFrame, metrics=("recall_cancer", "f1", "roc_auc", "accuracy"),
                           title: str = "Comparaison des approches"):
    """Barres groupées moyenne ± écart-type. ``df`` : colonnes ``approach`` + ``<metric>_mean/_std``."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(df)
    x = np.arange(len(metrics))
    for i, (_, row) in enumerate(df.iterrows()):
        means = [row[f"{m}_mean"] for m in metrics]
        stds = [row.get(f"{m}_std", 0) for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=row["approach"])
    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

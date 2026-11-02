"""Métriques de classification orientées risque clinique.

Classe positive = cancer (1). L'erreur la plus grave est le FAUX NÉGATIF (tumeur manquée) :
métrique principale = rappel (sensibilité) sur « cancer », sous contrainte de précision
raisonnable ; F1/F2, AUC-ROC et AUC-PR en complément (accuracy seule est trompeuse).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        precision_cancer=precision_score(y_true, y_pred, zero_division=0),
        recall_cancer=recall_score(y_true, y_pred, zero_division=0),
        specificity=tn / (tn + fp) if (tn + fp) else np.nan,
        f1=f1_score(y_true, y_pred, zero_division=0),
        f2=fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob) if both else np.nan,
        pr_auc=average_precision_score(y_true, y_prob) if both else np.nan,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=threshold,
    )


def roc_points(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr, roc_auc_score(y_true, y_prob)


def choose_threshold_for_recall(y_true, y_prob, min_recall: float = 0.95) -> float:
    """Plus haut seuil garantissant ``recall >= min_recall`` (à calibrer sur validation, pas sur test)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    best = 0.0
    for t in np.unique(np.r_[y_prob, 0.5]):
        pred = (y_prob >= t).astype(int)
        pos = (y_true == 1).sum()
        if pos and ((pred == 1) & (y_true == 1)).sum() / pos >= min_recall:
            best = max(best, float(t))
    return best if best > 0 else 0.5


def bootstrap_ci(y_true, y_prob, metric: str = "recall_cancer", n: int = 1000, seed: int = 0,
                 threshold: float = 0.5) -> tuple[float, float]:
    """IC 95 % par bootstrap : indispensable avec un petit jeu de test."""
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[i])) < 2:
            continue
        vals.append(compute_metrics(y_true[i], y_prob[i], threshold)[metric])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def summarize_runs(runs: list[dict], approach: str) -> dict:
    """Moyenne ± écart-type sur les répétitions, au format attendu par ``plot_metric_comparison``."""
    df = pd.DataFrame(runs)
    out = {"approach": approach}
    for c in ("accuracy", "balanced_accuracy", "precision_cancer", "recall_cancer", "specificity",
              "f1", "f2", "roc_auc", "pr_auc"):
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std(ddof=0))
    return out

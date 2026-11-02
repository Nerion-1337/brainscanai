"""Évaluation des clusters : scores internes, externes (ARI...) et mapping cluster -> classe."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)


def internal_scores(X: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette, Davies-Bouldin, Calinski-Harabasz (bruit DBSCAN exclu)."""
    m = labels >= 0
    if len(np.unique(labels[m])) < 2 or m.sum() < 3:
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return dict(silhouette=silhouette_score(X[m], labels[m]),
                davies_bouldin=davies_bouldin_score(X[m], labels[m]),
                calinski_harabasz=calinski_harabasz_score(X[m], labels[m]))


def external_scores(y_true: np.ndarray, labels: np.ndarray) -> dict:
    """ARI, NMI, homogénéité/complétude/V-measure sur les images labellisées."""
    h, c, v = homogeneity_completeness_v_measure(y_true, labels)
    return dict(ARI=adjusted_rand_score(y_true, labels),
                NMI=normalized_mutual_info_score(y_true, labels),
                homogeneity=h, completeness=c, v_measure=v)


def map_clusters_to_classes(cluster_labels: np.ndarray, y_true: np.ndarray, n_classes: int = 2) -> dict:
    """Associe chaque cluster à une classe (appariement hongrois sur la matrice de contingence).

    À APPELER UNIQUEMENT avec des labels du jeu ``strong_train`` (jamais le test).
    Les clusters non appariés prennent la classe majoritaire ; -1 (bruit) reste -1.
    """
    clusters = [c for c in np.unique(cluster_labels) if c >= 0]
    cont = np.zeros((len(clusters), n_classes), dtype=int)
    for i, c in enumerate(clusters):
        for k in range(n_classes):
            cont[i, k] = np.sum((cluster_labels == c) & (y_true == k))
    rows, cols = linear_sum_assignment(-cont)
    mapping = {clusters[r]: int(c) for r, c in zip(rows, cols)}
    for i, c in enumerate(clusters):
        if c not in mapping:
            mapping[c] = int(cont[i].argmax())
    mapping[-1] = -1
    return mapping


def purity(y_true: np.ndarray, labels: np.ndarray) -> float:
    total = 0
    for c in np.unique(labels):
        total += np.bincount(y_true[labels == c]).max()
    return total / len(y_true)


def evaluate_all(results: dict, X: np.ndarray, y_all: np.ndarray) -> pd.DataFrame:
    """Tableau comparatif des méthodes. ``y_all`` : -1 pour les images non labellisées.

    Les scores externes utilisent uniquement les images labellisées (le clustering lui-même
    n'a vu aucun label : il n'y a donc pas de fuite).
    """
    m = y_all >= 0
    rows = []
    for name, r in results.items():
        row = dict(method=name, n_clusters=len(set(r.labels[r.labels >= 0])),
                   noise_pct=float((r.labels == -1).mean() * 100))
        row.update(internal_scores(X, r.labels))
        row.update(external_scores(y_all[m], r.labels[m]))
        row["purity"] = purity(y_all[m], r.labels[m])
        rows.append(row)
    return pd.DataFrame(rows).set_index("method").round(4)

"""Algorithmes de clustering comparés + score de confiance par échantillon."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


@dataclass
class ClusterResult:
    name: str
    labels: np.ndarray            # -1 = bruit (DBSCAN)
    confidence: np.ndarray        # dans [0, 1]
    params: dict = field(default_factory=dict)


def knn_agreement(X: np.ndarray, labels: np.ndarray, k: int = 10) -> np.ndarray:
    """Confiance générique : part des k plus proches voisins appartenant au même cluster."""
    k = min(k, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]
    return (labels[idx] == labels[:, None]).mean(axis=1)


def run_kmeans(X, k: int = 2, seed: int = 42) -> ClusterResult:
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    d = np.sort(km.transform(X), axis=1)
    conf = (d[:, 1] - d[:, 0]) / (d[:, 1] + d[:, 0] + 1e-12)  # marge relative entre 2 centroïdes
    return ClusterResult("kmeans", km.labels_, conf, {"k": k})


def run_gmm(X, k: int = 2, seed: int = 42) -> ClusterResult:
    gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, n_init=3).fit(X)
    proba = gmm.predict_proba(X)
    return ClusterResult("gmm", proba.argmax(1), proba.max(1), {"k": k, "cov": "diag"})


def run_agglomerative(X, k: int = 2) -> ClusterResult:
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    return ClusterResult("agglomerative", labels, knn_agreement(X, labels), {"k": k, "linkage": "ward"})


def estimate_dbscan_eps(X, min_samples: int = 5, quantile: float = 0.90) -> float:
    """Heuristique du « coude » : quantile des distances au k-ième voisin."""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    d = nn.kneighbors(X)[0][:, -1]
    return float(np.quantile(d, quantile))


def run_dbscan(X, min_samples: int = 5, eps: float | None = None) -> ClusterResult:
    eps = eps or estimate_dbscan_eps(X, min_samples)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    conf = knn_agreement(X, labels)
    conf[labels == -1] = 0.0
    return ClusterResult("dbscan", labels, conf, {"eps": round(eps, 3), "min_samples": min_samples})


def run_all(X: np.ndarray, k: int = 2, seed: int = 42, dbscan_min_samples: int = 5) -> dict[str, ClusterResult]:
    """Lance les 4 méthodes comparées dans le notebook."""
    res = [run_kmeans(X, k, seed), run_gmm(X, k, seed), run_agglomerative(X, k),
           run_dbscan(X, dbscan_min_samples)]
    return {r.name: r for r in res}

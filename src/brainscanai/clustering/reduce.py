"""Standardisation et réduction de dimension (PCA pour le clustering, t-SNE pour la visualisation)."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def standardize(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def pca_reduce(X: np.ndarray, n_components: float | int = 0.95, seed: int = 42):
    """PCA ; ``n_components`` flottant = part de variance expliquée conservée."""
    pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)
    return pca.fit_transform(X), pca


def tsne_embed(X: np.ndarray, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    """t-SNE 2D (visualisation uniquement : ne pas clusteriser dessus)."""
    perplexity = min(perplexity, max(2.0, (len(X) - 1) / 3))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                random_state=seed).fit_transform(X)

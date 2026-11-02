"""Méthodes semi-supervisées légères sur embeddings (sans fine-tuning du CNN).

Toutes retournent P(cancer) pour ``X_test``. Le test n'est JAMAIS passé à ``fit`` :
on prédit ensuite via ``predict_proba`` (évaluation inductive, sans fuite).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier


def _scale(X_fit: np.ndarray, *others: np.ndarray):
    sc = StandardScaler().fit(X_fit)
    return (sc.transform(X_fit), *[sc.transform(o) for o in others])


def supervised_logreg(X_l, y_l, X_test, C: float = 1.0, seed: int = 42) -> np.ndarray:
    """Référence : régression logistique sur les seuls labels forts."""
    Xl, Xt = _scale(X_l, X_test)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed).fit(Xl, y_l)
    return clf.predict_proba(Xt)[:, 1]


def self_training(X_l, y_l, X_u, X_test, threshold: float = 0.90, C: float = 1.0, seed: int = 42):
    """Pseudo-labelling itératif : on ajoute les prédictions de confiance >= ``threshold``."""
    Xa, Xt = _scale(np.vstack([X_l, X_u]), X_test)
    y = np.r_[y_l, -np.ones(len(X_u), dtype=int)]
    base = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
    clf = SelfTrainingClassifier(base, threshold=threshold, max_iter=20).fit(Xa, y)
    return clf.predict_proba(Xt)[:, 1]


def _graph(cls, X_l, y_l, X_u, X_test, kernel="knn", n_neighbors=7, alpha=0.2):
    Xa, Xt = _scale(np.vstack([X_l, X_u]), X_test)
    y = np.r_[y_l, -np.ones(len(X_u), dtype=int)]
    kw = dict(kernel=kernel, n_neighbors=n_neighbors, max_iter=1000)
    if cls is LabelSpreading:
        kw["alpha"] = alpha
    model = cls(**kw).fit(Xa, y)
    p = model.predict_proba(Xt)
    return np.nan_to_num(p[:, 1], nan=0.5)


def label_spreading(X_l, y_l, X_u, X_test, **kw):
    return _graph(LabelSpreading, X_l, y_l, X_u, X_test, **kw)


def label_propagation(X_l, y_l, X_u, X_test, **kw):
    return _graph(LabelPropagation, X_l, y_l, X_u, X_test, kernel=kw.get("kernel", "knn"),
                  n_neighbors=kw.get("n_neighbors", 7))


def weak_plus_strong_logreg(X_l, y_l, X_w, y_w, X_test, w_weak: float = 0.3, C: float = 1.0, seed: int = 42):
    """Régression logistique sur labels forts + faibles (poids réduit pour les faibles)."""
    Xa, Xt = _scale(np.vstack([X_l, X_w]), X_test)
    y = np.r_[y_l, y_w]
    sw = np.r_[np.ones(len(y_l)), np.full(len(y_w), w_weak)]
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(Xa, y, sample_weight=sw)
    return clf.predict_proba(Xt)[:, 1]

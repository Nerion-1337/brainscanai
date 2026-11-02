"""Labellisation faible : clusters -> classes, conservation des pseudo-labels les plus confiants."""
from __future__ import annotations

import numpy as np
import pandas as pd

from brainscanai.clustering.cluster import ClusterResult
from brainscanai.clustering.evaluate import map_clusters_to_classes
from brainscanai.data.split import ROLE_STRONG, ROLE_WEAK


def build_weak_labels(df: pd.DataFrame, result: ClusterResult, keep_fraction: float = 0.8,
                      n_classes: int = 2) -> pd.DataFrame:
    """Retourne le jeu « faiblement » labellisé (rôle ``weak_pool`` uniquement).

    ``df`` est aligné ligne à ligne avec ``result`` et contient ``role`` et ``label``.
    Le mapping cluster->classe n'est appris que sur ``strong_train``.
    """
    tr = (df["role"] == ROLE_STRONG).to_numpy()
    mapping = map_clusters_to_classes(result.labels[tr], df.loc[tr, "label"].to_numpy(), n_classes)
    weak_mask = (df["role"] == ROLE_WEAK).to_numpy()

    weak = df.loc[weak_mask, ["image_id", "path"]].copy()
    weak["cluster"] = result.labels[weak_mask]
    weak["weak_label"] = [mapping.get(c, -1) for c in weak["cluster"]]
    weak["confidence"] = result.confidence[weak_mask]
    weak = weak[weak["weak_label"] >= 0]
    if len(weak):
        # seuil de confiance par classe, pour conserver un équilibre entre classes
        thr = weak.groupby("weak_label")["confidence"].transform(lambda s: s.quantile(1 - keep_fraction))
        weak = weak[weak["confidence"] >= thr]
    weak = weak.reset_index(drop=True)
    weak.attrs["mapping"] = mapping
    return weak


def weak_label_quality(weak: pd.DataFrame, truth: dict[int, int]) -> dict:
    """Audit possible UNIQUEMENT sur données de démo/validation : compare aux vrais labels connus."""
    known = weak[weak["image_id"].isin(truth)]
    if known.empty:
        return {}
    y = np.array([truth[i] for i in known["image_id"]])
    return dict(n=len(known), accuracy=float((known["weak_label"].to_numpy() == y).mean()))

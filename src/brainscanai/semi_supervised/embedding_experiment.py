"""Comparaison supervisé / semi-supervisé sur embeddings (rapide, sans GPU), répétée sur plusieurs seeds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from brainscanai.clustering.cluster import run_all
from brainscanai.clustering.reduce import pca_reduce, standardize
from brainscanai.data.split import ROLE_STRONG, ROLE_TEST, ROLE_WEAK, assign_roles
from brainscanai.evaluation.metrics import compute_metrics, summarize_runs
from brainscanai.semi_supervised import sklearn_methods as sm
from brainscanai.semi_supervised.weak_labels import build_weak_labels


def run_embedding_experiment(df: pd.DataFrame, X: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """``df`` (manifeste nettoyé) et ``X`` (embeddings) sont alignés ligne à ligne.

    Retourne (tableau résumé moyenne±std par approche, probabilités du dernier run pour les ROC).
    """
    cc, sc = cfg["clustering"], cfg["semi_supervised"]
    # Réduction non supervisée (aucun label utilisé) : appliquée à tout le monde.
    Z, _ = pca_reduce(standardize(X)[0], cc["pca_components"], cfg["seed"])
    runs: dict[str, list] = {}
    last_probs: dict = {}
    for r in range(sc["n_repeats"]):
        seed = cfg["seed"] + r
        d = assign_roles(df, cfg["data"]["test_size"], seed)
        tr = (d.role == ROLE_STRONG).to_numpy()
        te = (d.role == ROLE_TEST).to_numpy()
        wk = (d.role == ROLE_WEAK).to_numpy()
        y_tr, y_te = d.loc[tr, "label"].to_numpy(), d.loc[te, "label"].to_numpy()

        res = run_all(Z, cc["n_clusters"], seed, cc["dbscan_min_samples"])[cc["selected_method"]]
        weak = build_weak_labels(d, res, cc["keep_fraction"], cc["n_clusters"])
        pos = d.reset_index(drop=True).reset_index().set_index("image_id")["index"]
        Xw = Z[pos.loc[weak["image_id"]].to_numpy()]

        probs = {
            "Supervisé (labels forts seuls)": sm.supervised_logreg(Z[tr], y_tr, Z[te], seed=seed),
            "Faible seul (clusters)": sm.supervised_logreg(Xw, weak["weak_label"].to_numpy(), Z[te], seed=seed),
            "Faible + fort (pondéré)": sm.weak_plus_strong_logreg(Z[tr], y_tr, Xw, weak["weak_label"].to_numpy(),
                                                                   Z[te], seed=seed),
            "Self-training": sm.self_training(Z[tr], y_tr, Z[wk], Z[te], sc["sklearn"]["self_training_threshold"],
                                              seed=seed),
            "Label Spreading": sm.label_spreading(Z[tr], y_tr, Z[wk], Z[te], **sc["sklearn"]["label_spreading"]),
            "Label Propagation": sm.label_propagation(Z[tr], y_tr, Z[wk], Z[te],
                                                       n_neighbors=sc["sklearn"]["label_spreading"]["n_neighbors"]),
        }
        for name, p in probs.items():
            runs.setdefault(name, []).append(compute_metrics(y_te, p))
        last_probs = {"y_test": y_te, **probs}
    summary = pd.DataFrame([summarize_runs(v, k) for k, v in runs.items()])
    return summary, last_probs

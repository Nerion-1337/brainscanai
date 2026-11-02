"""Protocole de la mission : CNN faible -> fine-tuning fort, comparé au CNN supervisé seul.

Pour chaque répétition (seed) :
  1. split labellisées -> strong_train / test ; non labellisées -> weak_pool
  2. clustering (non supervisé) -> labels faibles (mapping appris sur strong_train UNIQUEMENT)
  3. A) CNN entraîné sur labels faibles            -> évalué sur test
     B) A) puis fine-tuné sur strong_train         -> évalué sur test  (semi-supervisé)
     C) CNN entraîné sur strong_train seul         -> évalué sur test  (supervisé, référence)
Le test n'est jamais vu pendant aucun entraînement ni pour le mapping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from brainscanai.clustering.cluster import run_all
from brainscanai.clustering.reduce import pca_reduce, standardize
from brainscanai.data.split import ROLE_STRONG, ROLE_TEST, assign_roles
from brainscanai.evaluation.metrics import compute_metrics, summarize_runs
from brainscanai.semi_supervised.models import build_cnn
from brainscanai.semi_supervised.train import TrainConfig, predict_proba, train_model
from brainscanai.semi_supervised.weak_labels import build_weak_labels
from brainscanai.utils import get_device, get_logger, set_seed

log = get_logger()


def _tc(stage: dict, cnn_cfg: dict, cfg: dict) -> TrainConfig:
    return TrainConfig(epochs=stage["epochs"], lr=stage["lr"], weight_decay=cnn_cfg["weight_decay"],
                       batch_size=cnn_cfg["batch_size"], patience=cnn_cfg["patience"],
                       image_size=cfg["data"]["image_size"], num_workers=cfg["features"]["num_workers"])


def _val_split(paths, labels, frac, seed):
    p_tr, p_va, y_tr, y_va = train_test_split(paths, labels, test_size=frac, stratify=labels, random_state=seed)
    return list(p_tr), list(y_tr), list(p_va), list(y_va)


def run_cnn_experiment(df: pd.DataFrame, X: np.ndarray, cfg: dict, n_repeats: int | None = None):
    """``df`` / ``X`` : manifeste nettoyé et embeddings alignés. Retourne (résumé, runs détaillés)."""
    cnn, cc = cfg["semi_supervised"]["cnn"], cfg["clustering"]
    device = get_device(cfg["features"]["device"])
    Z, _ = pca_reduce(standardize(X)[0], cc["pca_components"], cfg["seed"])
    runs: dict[str, list] = {"CNN faible seul (A)": [], "CNN faible -> fort (B, semi-sup.)": [],
                             "CNN fort seul (C, supervisé)": []}
    probs_last = {}
    for r in range(n_repeats or cfg["semi_supervised"]["n_repeats"]):
        seed = cfg["seed"] + r
        set_seed(seed)
        d = assign_roles(df, cfg["data"]["test_size"], seed).reset_index(drop=True)
        tr, te = d[d.role == ROLE_STRONG], d[d.role == ROLE_TEST]
        res = run_all(Z, cc["n_clusters"], seed, cc["dbscan_min_samples"])[cc["selected_method"]]
        weak = build_weak_labels(d, res, cc["keep_fraction"], cc["n_clusters"])
        log.info("[seed %d] strong=%d test=%d weak=%d", seed, len(tr), len(te), len(weak))

        # ---- A : labels faibles (validation = 10 % des labels faibles, jamais de label fort)
        wp, wy, wvp, wvy = _val_split(weak["path"].tolist(), weak["weak_label"].tolist(), 0.1, seed)
        model = build_cnn(cnn["arch"], cnn["pretrained"])
        train_model(model, wp, wy, _tc(cnn["weak_stage"], cnn, cfg), device, wvp, wvy)
        pa = predict_proba(model, te["path"].tolist(), _tc(cnn["weak_stage"], cnn, cfg), device)
        runs["CNN faible seul (A)"].append(compute_metrics(te["label"], pa))

        # ---- B : fine-tuning sur labels forts (même modèle, poursuite de l'entraînement)
        sp, sy, svp, svy = _val_split(tr["path"].tolist(), tr["label"].tolist(), cnn["val_fraction"], seed)
        tcs = _tc(cnn["strong_stage"], cnn, cfg)
        train_model(model, sp, sy, tcs, device, svp, svy)
        pb = predict_proba(model, te["path"].tolist(), tcs, device)
        runs["CNN faible -> fort (B, semi-sup.)"].append(compute_metrics(te["label"], pb))

        # ---- C : supervisé de référence, mêmes données fortes / même validation
        base = build_cnn(cnn["arch"], cnn["pretrained"])
        tcb = _tc(cnn["baseline"], cnn, cfg)
        train_model(base, sp, sy, tcb, device, svp, svy)
        pc = predict_proba(base, te["path"].tolist(), tcb, device)
        runs["CNN fort seul (C, supervisé)"].append(compute_metrics(te["label"], pc))
        probs_last = {"y_test": te["label"].to_numpy(), "A": pa, "B": pb, "C": pc}
    summary = pd.DataFrame([summarize_runs(v, k) for k, v in runs.items()])
    return summary, probs_last

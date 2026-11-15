"""Protocole de la mission : CNN faible -> fort, comparé au CNN supervisé seul (+ pseudo-labelling par le modèle).

Pour chaque répétition (seed) :
  1. split labellisées -> strong_train / test ; non labellisées -> weak_pool
  2. clustering (non supervisé) -> labels faibles (mapping appris sur strong_train UNIQUEMENT)
  A) CNN entraîné sur labels faibles (clusters)                    -> évalué sur test
  B) A) puis fine-tuné sur strong_train (mêmes hyperparamètres que C) -> évalué sur test  [semi-supervisé]
  C) CNN entraîné sur strong_train seul                            -> évalué sur test  [supervisé, référence]
  D) pseudo-labelling PAR LE MODÈLE : C prédit le weak_pool, on garde les prédictions très
     confiantes, puis un nouveau CNN est entraîné sur strong_train + pseudo-labels  [semi-supervisé]
Le test n'est jamais vu pendant aucun entraînement, ni pour le mapping, ni pour la sélection des pseudo-labels.
Comparaison équitable : B, C et D reçoivent le même nombre d'epochs/lr pour l'étape « forte » (strong_stage),
D utilisant ``pseudo_stage`` car son jeu d'entraînement est plus grand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from brainscanai.clustering.cluster import run_all
from brainscanai.clustering.reduce import prepare_clustering_space
from brainscanai.data.split import ROLE_STRONG, ROLE_TEST, ROLE_WEAK, assign_roles
from brainscanai.evaluation.metrics import compute_metrics, summarize_runs
from brainscanai.semi_supervised.models import build_cnn
from brainscanai.semi_supervised.train import TrainConfig, predict_proba, train_model
from brainscanai.semi_supervised.weak_labels import build_weak_labels
from brainscanai.utils import get_device, get_logger, set_seed

log = get_logger()

ARMS_ALL = ("A", "B", "C", "D")
ARM_NAMES = {
    "A": "CNN faible seul (A)",
    "B": "CNN faible -> fort (B, semi-sup.)",
    "C": "CNN fort seul (C, supervisé)",
    "D": "CNN pseudo-labels du modèle (D, semi-sup.)",
}


def _tc(stage: dict, cnn_cfg: dict, cfg: dict) -> TrainConfig:
    return TrainConfig(epochs=stage["epochs"], lr=stage["lr"], weight_decay=cnn_cfg["weight_decay"],
                       batch_size=cnn_cfg["batch_size"], patience=cnn_cfg["patience"],
                       image_size=cfg["data"]["image_size"], num_workers=cfg["features"]["num_workers"])


def _val_split(paths, labels, frac, seed):
    p_tr, p_va, y_tr, y_va = train_test_split(paths, labels, test_size=frac, stratify=labels, random_state=seed)
    return list(p_tr), list(y_tr), list(p_va), list(y_va)


def select_pseudo_labels(paths, proba_cancer: np.ndarray, threshold: float, max_images: int):
    """Garde, par classe, les prédictions les plus confiantes (>= threshold), au plus ``max_images/2`` par classe."""
    conf = np.maximum(proba_cancer, 1 - proba_cancer)
    pred = (proba_cancer >= 0.5).astype(int)
    per_class = max(max_images // 2, 1)
    keep = []
    for c in (0, 1):
        idx = np.where((pred == c) & (conf >= threshold))[0]
        keep.extend(idx[np.argsort(-conf[idx])][:per_class].tolist())
    keep = np.array(sorted(keep), dtype=int)
    return [paths[i] for i in keep], pred[keep].tolist()


def run_cnn_experiment(df: pd.DataFrame, X: np.ndarray, cfg: dict, n_repeats: int | None = None,
                       arms: tuple[str, ...] = ARMS_ALL):
    """``df`` / ``X`` : manifeste nettoyé et embeddings alignés.

    Retourne (résumé moyenne±std par approche, probabilités de la dernière répétition, diagnostics par répétition).
    """
    cnn, cc = cfg["semi_supervised"]["cnn"], cfg["clustering"]
    device = get_device(cfg["features"]["device"])
    Z, _ = prepare_clustering_space(X, cc["pca_components"], cc.get("l2_normalize", True), cfg["seed"])
    runs = {a: [] for a in arms}
    diags, probs_last = [], {}
    for r in range(n_repeats or cfg["semi_supervised"]["n_repeats"]):
        seed = cfg["seed"] + r
        set_seed(seed)
        d = assign_roles(df, cfg["data"]["test_size"], seed).reset_index(drop=True)
        tr, te, pool = d[d.role == ROLE_STRONG], d[d.role == ROLE_TEST], d[d.role == ROLE_WEAK]
        y_te = te["label"].to_numpy()
        te_paths = te["path"].tolist()
        res = run_all(Z, cc["n_clusters"], seed, cc["dbscan_min_samples"])[cc["selected_method"]]
        weak = build_weak_labels(d, res, cc["keep_fraction"], cc["n_clusters"])
        mapping = weak.attrs["mapping"]
        # Qualité des labels faibles : précision du mapping cluster->classe sur les images de test
        cl_pred = np.array([mapping.get(c, -1) for c in res.labels[te.index.to_numpy()]])
        diag = dict(seed=seed, n_strong=len(tr), n_test=len(te), n_weak_kept=len(weak),
                    weak_label_acc_on_test=float((cl_pred == y_te).mean()))
        log.info("[seed %d] strong=%d test=%d weak=%d | précision des labels faibles (proxy test)=%.2f",
                 seed, len(tr), len(te), len(weak), diag["weak_label_acc_on_test"])

        tcs = _tc(cnn["strong_stage"], cnn, cfg)
        sp, sy, svp, svy = _val_split(tr["path"].tolist(), tr["label"].tolist(), cnn["val_fraction"], seed)
        probs = {}

        # ---- A puis B : labels faibles, puis fine-tuning sur labels forts
        if "A" in arms or "B" in arms:
            wp, wy, wvp, wvy = _val_split(weak["path"].tolist(), weak["weak_label"].tolist(), 0.1, seed)
            model = build_cnn(cnn["arch"], cnn["pretrained"])
            tcw = _tc(cnn["weak_stage"], cnn, cfg)
            train_model(model, wp, wy, tcw, device, wvp, wvy)
            if "A" in arms:
                probs["A"] = predict_proba(model, te_paths, tcw, device)
            if "B" in arms:
                train_model(model, sp, sy, tcs, device, svp, svy)
                probs["B"] = predict_proba(model, te_paths, tcs, device)

        # ---- C : supervisé de référence (mêmes hyperparamètres « forts » que B) ; requis aussi par D
        if "C" in arms or "D" in arms:
            base = build_cnn(cnn["arch"], cnn["pretrained"])
            train_model(base, sp, sy, tcs, device, svp, svy)
            probs["C"] = predict_proba(base, te_paths, tcs, device)

        # ---- D : pseudo-labelling par le modèle (le test n'est pas utilisé)
        if "D" in arms:
            p_pool = predict_proba(base, pool["path"].tolist(), tcs, device)
            pp, py = select_pseudo_labels(pool["path"].tolist(), p_pool, cnn["pseudo_threshold"],
                                          cnn["pseudo_max_images"])
            diag["n_pseudo_kept"] = len(pp)
            if pp:
                tcd = _tc(cnn["pseudo_stage"], cnn, cfg)
                model_d = build_cnn(cnn["arch"], cnn["pretrained"])
                train_model(model_d, sp + pp, sy + py, tcd, device, svp, svy)
                probs["D"] = predict_proba(model_d, te_paths, tcd, device)
            else:
                log.warning("[seed %d] aucun pseudo-label >= %.2f : D = C", seed, cnn["pseudo_threshold"])
                probs["D"] = probs["C"]

        for a in arms:
            if a in probs:
                runs[a].append(compute_metrics(y_te, probs[a]))
        diags.append(diag)
        probs_last = {"y_test": y_te, **probs}
    summary = pd.DataFrame([summarize_runs(v, ARM_NAMES[a]) for a, v in runs.items() if v])
    return summary, probs_last, pd.DataFrame(diags)

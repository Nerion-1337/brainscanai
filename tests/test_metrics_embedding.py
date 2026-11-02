import numpy as np
import pandas as pd

from brainscanai.config import load_config
from brainscanai.evaluation.metrics import (
    bootstrap_ci,
    choose_threshold_for_recall,
    compute_metrics,
    summarize_runs,
)
from brainscanai.semi_supervised.embedding_experiment import run_embedding_experiment


def test_metrics_perfect_and_worst():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["recall_cancer"] == 1 and m["f1"] == 1 and m["roc_auc"] == 1 and m["fn"] == 0
    m = compute_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))
    assert m["recall_cancer"] == 0 and m["fn"] == 2


def test_threshold_for_recall_and_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 200), 0, 1)
    t = choose_threshold_for_recall(y, p, 0.95)
    assert compute_metrics(y, p, t)["recall_cancer"] >= 0.95
    lo, hi = bootstrap_ci(y, p, n=200)
    assert lo <= hi


def test_summarize_runs_keys():
    r = [compute_metrics([0, 1, 1, 0], [0.2, 0.9, 0.6, 0.4])] * 2
    s = summarize_runs(r, "x")
    assert s["recall_cancer_mean"] == 1 and s["recall_cancer_std"] == 0


def test_embedding_experiment_runs_and_semi_competitive():
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=400, centers=2, n_features=30, cluster_std=4.0, random_state=1)
    df = pd.DataFrame(dict(image_id=range(400), path="x", source="unlabeled", label=-1))
    lab = np.r_[np.arange(0, 20), np.arange(200, 220)]
    df.loc[lab, ["source", "label"]] = ["labeled", 0]
    df.loc[lab, "label"] = y[lab]
    cfg = load_config()
    cfg["semi_supervised"]["n_repeats"] = 2
    summary, probs = run_embedding_experiment(df, X, cfg)
    assert len(summary) == 6 and "roc_auc_mean" in summary
    assert summary["roc_auc_mean"].max() > 0.8
    assert len(probs["y_test"]) == 12

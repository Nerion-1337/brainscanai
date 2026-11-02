"""Étape 4 : comparaison supervisé / semi-supervisé (embeddings sklearn + CNN)."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

from brainscanai.config import load_config, resolve_path  # noqa: E402
from brainscanai.evaluation.plots import plot_metric_comparison, save_fig  # noqa: E402
from brainscanai.features.io import load_features  # noqa: E402
from brainscanai.semi_supervised.embedding_experiment import run_embedding_experiment  # noqa: E402
from brainscanai.utils import get_logger, set_seed  # noqa: E402

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-cnn", action="store_true", help="n'exécuter que les méthodes sur embeddings")
    ap.add_argument("--repeats", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.repeats:
        cfg["semi_supervised"]["n_repeats"] = a.repeats
    set_seed(cfg["seed"])
    manifest = pd.read_csv(resolve_path(cfg, "processed") / "manifest_clean.csv")
    X, df = load_features(resolve_path(cfg, "features") / "embeddings.npz", manifest)
    tdir, fdir = resolve_path(cfg, "tables", create=True), resolve_path(cfg, "figures", create=True)

    summary, _ = run_embedding_experiment(df, X, cfg)
    summary.to_csv(tdir / "semi_supervised_embeddings.csv", index=False)
    save_fig(plot_metric_comparison(summary, title="Méthodes sur embeddings"), "compare_embeddings", fdir)
    log.info("\n%s", summary[["approach", "recall_cancer_mean", "f1_mean", "roc_auc_mean"]].round(3))

    if not a.skip_cnn:
        from brainscanai.semi_supervised.cnn_experiment import (
            run_cnn_experiment,  # import torch tardif
        )

        cnn_summary, _ = run_cnn_experiment(df, X, cfg)
        cnn_summary.to_csv(tdir / "semi_supervised_cnn.csv", index=False)
        save_fig(plot_metric_comparison(cnn_summary, title="CNN : supervisé vs semi-supervisé"),
                 "compare_cnn", fdir)
        log.info("\n%s", cnn_summary[["approach", "recall_cancer_mean", "f1_mean", "roc_auc_mean"]].round(3))


if __name__ == "__main__":
    main()

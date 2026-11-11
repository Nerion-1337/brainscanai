"""Ablation de `keep_fraction` (qualité vs quantité des labels faibles) + bras C/D, sur les MÊMES splits.

Coût : C et D ne dépendent pas de keep_fraction -> lancés une seule fois ; A et B sont relancés pour chaque valeur.
Exemple : docker compose run --rm semi python scripts/run_ablation.py --keep 0.3 0.5 0.8 --repeats 3
"""
from __future__ import annotations

import argparse
import copy

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

from brainscanai.config import load_config, resolve_path  # noqa: E402
from brainscanai.evaluation.plots import plot_metric_comparison, save_fig  # noqa: E402
from brainscanai.features.io import load_features  # noqa: E402
from brainscanai.semi_supervised.cnn_experiment import run_cnn_experiment  # noqa: E402
from brainscanai.utils import get_logger  # noqa: E402

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--keep", type=float, nargs="+", default=[0.3, 0.5, 0.8])
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    cfg = load_config(a.config)
    manifest = pd.read_csv(resolve_path(cfg, "processed") / "manifest_clean.csv")
    X, df = load_features(resolve_path(cfg, "features") / "embeddings.npz", manifest)

    parts, diags = [], []
    log.info("Bras C (supervisé) et D (pseudo-labels du modèle)")
    s, _, dg = run_cnn_experiment(df, X, cfg, n_repeats=a.repeats, arms=("C", "D"))
    parts.append(s)
    for k in a.keep:
        c2 = copy.deepcopy(cfg)
        c2["clustering"]["keep_fraction"] = k
        log.info("Bras A et B avec keep_fraction=%.2f", k)
        s, _, dg = run_cnn_experiment(df, X, c2, n_repeats=a.repeats, arms=("A", "B"))
        s["approach"] = s["approach"] + f" [keep={k}]"
        parts.append(s)
        diags.append(dg.assign(keep_fraction=k))
    table = pd.concat(parts, ignore_index=True)
    tdir, fdir = resolve_path(cfg, "tables", create=True), resolve_path(cfg, "figures", create=True)
    table.to_csv(tdir / "cnn_ablation.csv", index=False)
    pd.concat(diags).to_csv(tdir / "cnn_ablation_diagnostics.csv", index=False)
    save_fig(plot_metric_comparison(table, title="Ablation keep_fraction (CNN)"), "cnn_ablation", fdir)
    log.info("\n%s", table[["approach", "recall_cancer_mean", "recall_cancer_std", "f1_mean", "roc_auc_mean"]].round(3))


if __name__ == "__main__":
    main()

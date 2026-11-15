"""Étape 3 : standardisation -> PCA -> clustering (4 méthodes) -> ARI & figures."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from brainscanai.clustering.cluster import run_all  # noqa: E402
from brainscanai.clustering.evaluate import evaluate_all  # noqa: E402
from brainscanai.clustering.reduce import prepare_clustering_space, tsne_embed  # noqa: E402
from brainscanai.config import load_config, resolve_path  # noqa: E402
from brainscanai.evaluation.plots import plot_embedding_2d, save_fig  # noqa: E402
from brainscanai.features.io import load_features  # noqa: E402
from brainscanai.utils import get_logger, set_seed  # noqa: E402

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    cc = cfg["clustering"]

    manifest = pd.read_csv(resolve_path(cfg, "processed") / "manifest_clean.csv")
    X, m = load_features(resolve_path(cfg, "features") / "embeddings.npz", manifest)
    Z, pca = prepare_clustering_space(X, cc["pca_components"], cc.get("l2_normalize", True), cfg["seed"])
    log.info("PCA : %d -> %d dimensions (%.1f%% variance)", X.shape[1], Z.shape[1],
             100 * pca.explained_variance_ratio_.sum())

    results = run_all(Z, cc["n_clusters"], cfg["seed"], cc["dbscan_min_samples"])
    table = evaluate_all(results, Z, m["label"].to_numpy())
    tdir = resolve_path(cfg, "tables", create=True)
    table.to_csv(tdir / "clustering_comparison.csv")
    log.info("\n%s", table[["silhouette", "ARI", "NMI", "purity", "noise_pct"]])

    coords = tsne_embed(Z, cc["tsne_perplexity"], cfg["seed"])
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(4.2 * (len(results) + 1), 4))
    plot_embedding_2d(coords, m["label_name"], "Vérité terrain (labellisées)", axes[0])
    for ax, (name, r) in zip(axes[1:], results.items()):
        plot_embedding_2d(coords, r.labels, name, ax)
    save_fig(fig, "clustering_tsne", resolve_path(cfg, "figures", create=True))

    out = m[["image_id"]].copy()
    for name, r in results.items():
        out[f"{name}_cluster"] = r.labels
        out[f"{name}_conf"] = r.confidence
    out.to_csv(resolve_path(cfg, "processed") / "cluster_assignments.csv", index=False)


if __name__ == "__main__":
    main()

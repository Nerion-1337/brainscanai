"""Étape 1 : manifeste + contrôle qualité + nettoyage -> data/processed/."""
from __future__ import annotations

import argparse

from brainscanai.config import load_config, resolve_path
from brainscanai.data.loader import build_manifest
from brainscanai.data.quality import (
    clean_manifest,
    compute_image_stats,
    domain_shift_report,
    flag_duplicates,
    flag_outliers,
)
from brainscanai.utils import get_logger

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = resolve_path(cfg, "processed", create=True)

    manifest = build_manifest(cfg)
    log.info("%d images (%d labellisées)", len(manifest), (manifest.source == "labeled").sum())
    stats = compute_image_stats(manifest)
    q = cfg["data"]["quality"]
    stats = flag_duplicates(flag_outliers(stats, z=q["outlier_robust_z"]).assign(
        md5=lambda d: d["md5"].fillna("x_" + d["image_id"].astype(str))))
    cleaned, removed = clean_manifest(manifest, stats, q["drop_duplicates"], q["drop_unlabeled_outliers"])

    stats.to_csv(out / "image_stats.csv", index=False)
    cleaned.to_csv(out / "manifest_clean.csv", index=False)
    removed.to_csv(out / "removed_images.csv", index=False)
    domain_shift_report(stats).to_csv(out / "domain_shift.csv", index=False)
    log.info("Conservées : %d | retirées : %d -> %s", len(cleaned), len(removed), out)


if __name__ == "__main__":
    main()

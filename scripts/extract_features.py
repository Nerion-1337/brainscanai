"""Étape 2 : embeddings d'un backbone pré-entraîné -> data/features/embeddings.npz."""
from __future__ import annotations

import argparse

import pandas as pd

from brainscanai.config import load_config, resolve_path
from brainscanai.features.extractor import FeatureExtractor
from brainscanai.features.io import save_features
from brainscanai.utils import get_logger, set_seed

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="poids aléatoires (test hors-ligne)")
    ap.add_argument("--limit", type=int, default=None, help="n'extraire que N images (debug)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    fc = cfg["features"]

    manifest = pd.read_csv(resolve_path(cfg, "processed") / "manifest_clean.csv")
    if a.limit:
        manifest = manifest.sample(a.limit, random_state=cfg["seed"]).reset_index(drop=True)
    ext = FeatureExtractor(fc["backbone"], pretrained=fc["pretrained"] and not a.no_pretrained,
                           layer=fc["layer"], device=fc["device"])
    feats = ext.extract(manifest["path"], cfg["data"]["image_size"], fc["batch_size"], fc["num_workers"])
    out = save_features(manifest, feats, resolve_path(cfg, "features", create=True))
    log.info("Features %s sauvegardées -> %s", feats.shape, out)


if __name__ == "__main__":
    main()

"""Lecture / écriture des embeddings (sans dépendance à PyTorch)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_features(manifest: pd.DataFrame, feats: np.ndarray, out_dir: str | Path,
                  name: str = "embeddings") -> Path:
    """Sauve ``<name>.npz`` (features + image_id alignés) dans ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert len(manifest) == len(feats), "manifeste et features non alignés"
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, features=feats, image_id=manifest["image_id"].to_numpy())
    return path


def load_features(path: str | Path, manifest: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Recharge les features et réaligne le manifeste (au cas où des images ont été filtrées)."""
    z = np.load(path)
    ids = z["image_id"]
    m = manifest.set_index("image_id").loc[ids].reset_index()
    return z["features"], m

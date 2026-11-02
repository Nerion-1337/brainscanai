"""Génère un mini jeu d'IRM synthétiques (512x512 JPEG) pour tester le pipeline sans le vrai dataset.

Structure produite = celle attendue par le projet :
    data/raw/labeled/{normal,cancer}/  (par défaut 50 + 50)  et  data/raw/unlabeled/ (par défaut 300)
Les images « cancer » contiennent une masse brillante ; ce n'est PAS de l'imagerie médicale réelle.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]


def synth_brain(rng: np.random.Generator, tumor: bool, size: int = 512) -> Image.Image:
    yy, xx = np.mgrid[:size, :size]
    cy, cx = size / 2 + rng.normal(0, 8, 2)
    ry, rx = size * rng.uniform(0.34, 0.40), size * rng.uniform(0.28, 0.34)
    brain = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) < 1
    img = np.zeros((size, size), np.float32)
    img[brain] = 90 + 25 * np.sin(xx[brain] / 17.0) * np.cos(yy[brain] / 23.0)
    if tumor:
        ty, tx = cy + rng.normal(0, 55), cx + rng.normal(0, 55)
        r = rng.uniform(22, 48)
        mass = ((yy - ty) ** 2 + (xx - tx) ** 2) < r**2
        img[mass & brain] = rng.uniform(190, 235)
    img += rng.normal(0, 6, img.shape)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-labeled", type=int, default=50, help="par classe")
    ap.add_argument("--n-unlabeled", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for cls, tumor in (("normal", False), ("cancer", True)):
        d = ROOT / "data/raw/labeled" / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.n_labeled):
            synth_brain(rng, tumor).save(d / f"{cls}_{i:04d}.jpg", quality=92)
    d = ROOT / "data/raw/unlabeled"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(a.n_unlabeled):
        synth_brain(rng, bool(rng.random() < 0.5)).save(d / f"unl_{i:05d}.jpg", quality=92)
    print(f"Données de démo générées dans {ROOT / 'data/raw'}")


if __name__ == "__main__":
    main()

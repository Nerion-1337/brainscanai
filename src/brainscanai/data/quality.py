"""Contrôle qualité des images : format, corruption, doublons, outliers, biais de source."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import ks_2samp


def laplacian_variance(gray: np.ndarray) -> float:
    """Netteté : variance du Laplacien (faible = image floue)."""
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def compute_image_stats(manifest: pd.DataFrame) -> pd.DataFrame:
    """Statistiques par image (largeur, hauteur, mode, intensité, netteté, hash, corruption)."""
    records = []
    for row in manifest.itertuples():
        rec = dict(image_id=row.image_id, corrupt=False)
        try:
            with open(row.path, "rb") as f:
                rec["md5"] = hashlib.md5(f.read()).hexdigest()
            with Image.open(row.path) as im:
                rec.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
                gray = np.asarray(im.convert("L"), dtype=np.float32)
            rec.update(mean=gray.mean(), std=gray.std(), min=gray.min(), max=gray.max(),
                       sharpness=laplacian_variance(gray),
                       black_ratio=float((gray < 5).mean()))
        except Exception as exc:  # fichier illisible
            rec.update(corrupt=True, error=str(exc))
        records.append(rec)
    return manifest[["image_id", "source", "label"]].merge(pd.DataFrame(records), on="image_id")


def flag_outliers(stats: pd.DataFrame, columns=("mean", "std", "sharpness", "black_ratio"),
                  z: float = 4.0) -> pd.DataFrame:
    """Marque les outliers via un z-score robuste (médiane / MAD)."""
    stats = stats.copy()
    stats["outlier_reason"] = ""
    for col in columns:
        x = stats[col].astype(float)
        med = x.median()
        mad = (x - med).abs().median() * 1.4826
        if mad == 0 or np.isnan(mad):
            continue
        is_out = ((x - med).abs() / mad) > z
        stats.loc[is_out, "outlier_reason"] += f"{col};"
    stats["is_outlier"] = stats["outlier_reason"] != ""
    return stats


def flag_duplicates(stats: pd.DataFrame) -> pd.DataFrame:
    """Marque les doublons exacts ; on conserve en priorité l'image labellisée."""
    stats = stats.copy()
    order = stats.assign(_prio=(stats["source"] != "labeled").astype(int)).sort_values(["_prio", "image_id"])
    keep_ids = order.drop_duplicates("md5", keep="first")["image_id"]
    stats["is_duplicate"] = ~stats["image_id"].isin(keep_ids)
    return stats


def domain_shift_report(stats: pd.DataFrame, columns=("mean", "std", "sharpness")) -> pd.DataFrame:
    """Test de Kolmogorov-Smirnov labellisé vs non labellisé (homogénéité des sources)."""
    lab = stats[stats["source"] == "labeled"]
    unl = stats[stats["source"] == "unlabeled"]
    rows = []
    for col in columns:
        if len(lab) > 1 and len(unl) > 1:
            res = ks_2samp(lab[col].dropna(), unl[col].dropna())
            rows.append(dict(feature=col, ks_stat=res.statistic, p_value=res.pvalue,
                             shift_suspected=bool(res.pvalue < 0.01)))
    return pd.DataFrame(rows)


def clean_manifest(manifest: pd.DataFrame, stats: pd.DataFrame, drop_duplicates: bool = True,
                   drop_unlabeled_outliers: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (manifeste nettoyé, journal des images retirées).

    Règles : on retire toujours les fichiers corrompus ; doublons (copie non prioritaire) ;
    outliers uniquement parmi les non labellisés (les labellisées sont seulement signalées).
    """
    s = stats.copy()
    if "is_outlier" not in s:
        s = flag_outliers(s)
    s["md5"] = s["md5"].fillna("nohash_" + s["image_id"].astype(str))  # fichiers illisibles
    s = flag_duplicates(s)

    reasons = pd.Series("", index=s.index)
    reasons[s["corrupt"]] += "corrupt;"
    if drop_duplicates:
        reasons[s["is_duplicate"]] += "duplicate;"
    if drop_unlabeled_outliers:
        reasons[s["is_outlier"] & (s["source"] == "unlabeled")] += "outlier;"
    removed = s.loc[reasons != "", ["image_id", "source"]].assign(reason=reasons[reasons != ""])
    cleaned = manifest[~manifest["image_id"].isin(removed["image_id"])].reset_index(drop=True)
    return cleaned, removed.reset_index(drop=True)

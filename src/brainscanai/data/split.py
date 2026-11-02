"""Attribution des rôles : strong_train / test / weak_pool.

Règle d'or : les images labellisées sont scindées en ``strong_train`` et ``test``
(stratifié). Les images non labellisées forment le ``weak_pool`` (labels faibles
générés plus tard). Les trois ensembles ne se mélangent JAMAIS.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

ROLE_STRONG = "strong_train"
ROLE_TEST = "test"
ROLE_WEAK = "weak_pool"


def assign_roles(manifest: pd.DataFrame, test_size: float = 0.3, seed: int = 42) -> pd.DataFrame:
    df = manifest.copy()
    df["role"] = ROLE_WEAK
    labeled = df[df["source"] == "labeled"]
    if len(labeled) < 4:
        raise ValueError("Pas assez d'images labellisées pour créer un split train/test.")
    tr_idx, te_idx = train_test_split(
        labeled.index, test_size=test_size, stratify=labeled["label"], random_state=seed
    )
    df.loc[tr_idx, "role"] = ROLE_STRONG
    df.loc[te_idx, "role"] = ROLE_TEST
    assert_no_overlap(df)
    return df


def assert_no_overlap(df: pd.DataFrame) -> None:
    """Garde-fou anti-fuite : un même fichier ne doit jamais appartenir à deux rôles."""
    if df["image_id"].duplicated().any():
        raise AssertionError("image_id dupliqué : chevauchement entre ensembles")
    if (df.loc[df["role"] == ROLE_WEAK, "source"] == "labeled").any():
        raise AssertionError("Une image labellisée se trouve dans le weak_pool")
    if (df.loc[df["role"] != ROLE_WEAK, "source"] == "unlabeled").any():
        raise AssertionError("Une image non labellisée se trouve dans un ensemble 'fort'")

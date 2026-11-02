import numpy as np
import pandas as pd
import pytest
from PIL import Image

from brainscanai.data.quality import (
    clean_manifest,
    compute_image_stats,
    flag_duplicates,
    flag_outliers,
    laplacian_variance,
)
from brainscanai.data.split import ROLE_TEST, ROLE_WEAK, assert_no_overlap, assign_roles


def _make_images(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        arr = rng.normal(100, 15, (64, 64)).clip(0, 255).astype(np.uint8)
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr).save(p)
        rows.append(dict(image_id=i, path=str(p), filename=p.name,
                         source="labeled" if i < 8 else "unlabeled",
                         label=i % 2 if i < 8 else -1))
    # doublon exact d'une image labellisée, placé côté non labellisé
    dup = tmp_path / "dup.png"
    dup.write_bytes((tmp_path / "img_0.png").read_bytes())
    rows.append(dict(image_id=12, path=str(dup), filename="dup.png", source="unlabeled", label=-1))
    # image noire = outlier
    black = tmp_path / "black.png"
    Image.fromarray(np.zeros((64, 64), np.uint8)).save(black)
    rows.append(dict(image_id=13, path=str(black), filename="black.png", source="unlabeled", label=-1))
    return pd.DataFrame(rows)


def test_laplacian_variance_sharp_vs_flat():
    flat = np.full((32, 32), 100.0)
    noisy = np.random.default_rng(1).normal(100, 30, (32, 32))
    assert laplacian_variance(flat) == 0.0
    assert laplacian_variance(noisy) > 100


def test_quality_pipeline(tmp_path):
    manifest = _make_images(tmp_path)
    stats = compute_image_stats(manifest)
    assert not stats["corrupt"].any()
    stats = flag_duplicates(flag_outliers(stats, z=3.5))
    # la copie non labellisée est le doublon à retirer, l'originale labellisée est gardée
    assert bool(stats.loc[stats.image_id == 12, "is_duplicate"].iloc[0])
    assert not bool(stats.loc[stats.image_id == 0, "is_duplicate"].iloc[0])
    cleaned, removed = clean_manifest(manifest, stats)
    assert 12 in removed["image_id"].values
    assert (cleaned["source"] == "labeled").sum() == 8  # aucune image labellisée supprimée


def test_corrupt_file_is_removed(tmp_path):
    manifest = _make_images(tmp_path)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    manifest = pd.concat([manifest, pd.DataFrame([dict(
        image_id=99, path=str(bad), filename="bad.png", source="unlabeled", label=-1)])], ignore_index=True)
    stats = compute_image_stats(manifest)
    assert stats.loc[stats.image_id == 99, "corrupt"].iloc[0]
    cleaned, removed = clean_manifest(manifest, stats)
    assert 99 in removed["image_id"].values


def test_split_roles_and_no_leak():
    manifest = pd.DataFrame(dict(
        image_id=range(60),
        source=["labeled"] * 20 + ["unlabeled"] * 40,
        label=[0, 1] * 10 + [-1] * 40,
    ))
    df = assign_roles(manifest, test_size=0.3, seed=1)
    assert (df.role == ROLE_TEST).sum() == 6
    assert (df.role == ROLE_WEAK).sum() == 40
    assert df.loc[df.role == ROLE_TEST, "label"].nunique() == 2  # stratifié
    assert_no_overlap(df)
    df.loc[0, "role"] = ROLE_WEAK  # une image labellisée dans le weak pool = fuite
    with pytest.raises(AssertionError):
        assert_no_overlap(df)

"""Construction du manifeste d'images (une ligne par image)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from brainscanai.config import resolve_path


def _list_images(folder: Path, extensions: set[str]) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def build_manifest(cfg: dict) -> pd.DataFrame:
    """Parcourt ``data/raw`` et retourne un DataFrame.

    Colonnes : image_id, path, filename, source (labeled|unlabeled),
    label (0/1, -1 si inconnu), label_name.
    Arborescence attendue :
        data/raw/labeled/<classe>/*.jpg   et   data/raw/unlabeled/*.jpg
    """
    classes = cfg["data"]["classes"]
    exts = {e.lower() for e in cfg["data"]["extensions"]}
    labeled_dir = resolve_path(cfg, "raw_labeled")
    unlabeled_dir = resolve_path(cfg, "raw_unlabeled")

    rows = []
    for idx, cls in enumerate(classes):
        cls_dir = labeled_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Dossier de classe introuvable : {cls_dir}")
        for p in _list_images(cls_dir, exts):
            rows.append(dict(path=str(p), filename=p.name, source="labeled", label=idx, label_name=cls))
    if unlabeled_dir.exists():
        for p in _list_images(unlabeled_dir, exts):
            rows.append(dict(path=str(p), filename=p.name, source="unlabeled", label=-1,
                             label_name="unlabeled"))

    if not rows:
        raise FileNotFoundError(
            "Aucune image trouvée. Placez le jeu de données dans data/raw/ "
            "(ou lancez `make demo-data` pour générer des images synthétiques)."
        )
    df = pd.DataFrame(rows)
    df.insert(0, "image_id", range(len(df)))
    return df

"""Chargement de la configuration et résolution des chemins."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Charge ``configs/config.yaml`` (ou un fichier personnalisé)."""
    path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict[str, Any], key: str, create: bool = False) -> Path:
    """Retourne le chemin absolu associé à ``cfg['paths'][key]``."""
    p = Path(cfg["paths"][key])
    p = p if p.is_absolute() else PROJECT_ROOT / p
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p

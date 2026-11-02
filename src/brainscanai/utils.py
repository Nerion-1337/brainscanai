"""Fonctions utilitaires transverses (graine, device, logs)."""
from __future__ import annotations

import logging
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fixe les graines pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch est optionnel pour les étapes non-deep
        pass


def get_device(preference: str = "auto") -> str:
    """Retourne ``cuda``, ``mps`` ou ``cpu`` selon la disponibilité."""
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_logger(name: str = "brainscanai") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

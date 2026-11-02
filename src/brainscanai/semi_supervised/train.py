"""Boucles d'entraînement / prédiction PyTorch."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from brainscanai.data.dataset import MRIDataset, get_transforms


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    patience: int = 4          # arrêt anticipé sur la loss de validation
    image_size: int = 224
    num_workers: int = 0


def make_loader(paths, labels, cfg: TrainConfig, train: bool) -> DataLoader:
    ds = MRIDataset(paths, labels, get_transforms(cfg.image_size, train=train))
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=cfg.num_workers)


def _class_weights(labels, device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=2).astype(float)
    w = counts.sum() / (2 * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_model(model: nn.Module, train_paths, train_labels, cfg: TrainConfig, device: str,
                val_paths=None, val_labels=None) -> list[dict]:
    """Entraîne ``model`` en place ; restaure les meilleurs poids si validation fournie."""
    model.to(device)
    tr_loader = make_loader(train_paths, train_labels, cfg, train=True)
    va_loader = make_loader(val_paths, val_labels, cfg, train=False) if val_paths is not None else None
    crit = nn.CrossEntropyLoss(weight=_class_weights(train_labels, device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        rec = {"epoch": epoch + 1, "train_loss": tot / max(n, 1)}
        if va_loader is not None:
            model.eval()
            vt, vn = 0.0, 0
            with torch.no_grad():
                for x, y in va_loader:
                    x, y = x.to(device), y.to(device)
                    vt += crit(model(x), y).item() * len(y)
                    vn += len(y)
            rec["val_loss"] = vt / max(vn, 1)
            if rec["val_loss"] < best:
                best, best_state, bad = rec["val_loss"], copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
        history.append(rec)
        if va_loader is not None and bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.inference_mode()
def predict_proba(model: nn.Module, paths, cfg: TrainConfig, device: str) -> np.ndarray:
    """P(cancer) pour chaque image."""
    model.to(device).eval()
    loader = make_loader(paths, None, cfg, train=False)
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)

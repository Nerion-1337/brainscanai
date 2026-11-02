"""Extraction d'embeddings visuels avec un backbone pré-entraîné gelé."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from brainscanai.data.dataset import MRIDataset, get_transforms
from brainscanai.utils import get_device


def _build_body(backbone: str, pretrained: bool, layer: str) -> nn.Module:
    """Retourne le réseau tronqué (sans tête de classification) + pooling global."""
    weights = "DEFAULT" if pretrained else None
    model = models.get_model(backbone, weights=weights)
    if hasattr(model, "features"):  # DenseNet, VGG, MobileNet, EfficientNet...
        body = model.features
    else:  # ResNet & co : on coupe avant avgpool/fc, ou après ``layer`` (ex. layer3)
        names = [n for n, _ in model.named_children() if n not in ("avgpool", "fc")]
        if layer not in ("avgpool", *names):
            raise ValueError(f"Couche inconnue '{layer}'. Disponibles : {names + ['avgpool']}")
        keep = names if layer == "avgpool" else names[: names.index(layer) + 1]
        body = nn.Sequential(*[getattr(model, n) for n in keep])
    return nn.Sequential(body, nn.AdaptiveAvgPool2d(1), nn.Flatten())


class FeatureExtractor:
    """Backbone gelé en mode ``eval`` : image (3xHxW) -> vecteur de dimension fixe."""

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 layer: str = "avgpool", device: str = "auto"):
        self.backbone, self.layer = backbone, layer
        self.device = get_device(device)
        self.model = _build_body(backbone, pretrained, layer).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False  # couches gelées

    @torch.inference_mode()
    def extract(self, paths, image_size: int = 224, batch_size: int = 32, num_workers: int = 0,
                show_progress: bool = True) -> np.ndarray:
        ds = MRIDataset(paths, transform=get_transforms(image_size, train=False))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        feats = []
        for x, _ in tqdm(loader, disable=not show_progress, desc=f"{self.backbone}/{self.layer}"):
            feats.append(self.model(x.to(self.device)).cpu().numpy())
        return np.concatenate(feats).astype(np.float32)

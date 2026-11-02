"""Construction du CNN de classification (fine-tuning d'un ResNet pré-entraîné)."""
from __future__ import annotations

from torch import nn
from torchvision import models


def build_cnn(arch: str = "resnet18", pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    model = models.get_model(arch, weights="DEFAULT" if pretrained else None)
    if hasattr(model, "fc"):  # ResNet
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    elif hasattr(model, "classifier"):  # DenseNet / MobileNet / EfficientNet
        last = model.classifier[-1] if isinstance(model.classifier, nn.Sequential) else model.classifier
        in_f = last.in_features
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(in_f, num_classes)
        else:
            model.classifier = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Architecture non supportée : {arch}")
    return model

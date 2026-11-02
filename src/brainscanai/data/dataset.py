"""Dataset PyTorch et transformations (prétraitement adapté aux CNN pré-entraînés)."""
from __future__ import annotations

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ToRGB:
    """Les IRM sont en niveaux de gris : on réplique sur 3 canaux pour les backbones ImageNet."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.convert("RGB")


def get_transforms(image_size: int = 224, train: bool = False) -> T.Compose:
    """Resize + normalisation ImageNet ; augmentations légères et réalistes si ``train``.

    Pas de flip vertical ni de rotation forte : l'orientation anatomique compte en IRM.
    """
    ops = [ToRGB(), T.Resize((image_size, image_size))]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomRotation(8), T.ColorJitter(brightness=0.1, contrast=0.1)]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class MRIDataset(Dataset):
    """Retourne ``(tensor, label)`` ; label = -1 si inconnu."""

    def __init__(self, paths, labels=None, transform=None):
        self.paths = list(paths)
        self.labels = list(labels) if labels is not None else [-1] * len(self.paths)
        self.transform = transform or get_transforms()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            x = self.transform(im.copy())
        return x, torch.tensor(self.labels[idx], dtype=torch.long)

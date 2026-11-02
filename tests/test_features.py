import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from brainscanai.features.extractor import FeatureExtractor
from brainscanai.features.io import load_features, save_features  # noqa: E402


def test_extract_shape_and_roundtrip(tmp_path):
    from PIL import Image

    paths = []
    for i in range(4):
        p = tmp_path / f"{i}.png"
        Image.fromarray((np.random.rand(64, 64) * 255).astype("uint8")).save(p)
        paths.append(str(p))
    manifest = pd.DataFrame(dict(image_id=range(4), path=paths))
    ext = FeatureExtractor("resnet18", pretrained=False, device="cpu")
    feats = ext.extract(paths, image_size=64, batch_size=2, show_progress=False)
    assert feats.shape == (4, 512)
    f = save_features(manifest, feats, tmp_path)
    back, m = load_features(f, manifest)
    assert np.allclose(back, feats) and len(m) == 4

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from brainscanai.semi_supervised.cnn_experiment import select_pseudo_labels  # noqa: E402


def test_select_pseudo_labels_threshold_balance_and_cap():
    paths = [f"p{i}" for i in range(10)]
    proba = np.array([0.99, 0.98, 0.97, 0.60, 0.5, 0.40, 0.03, 0.02, 0.01, 0.55])
    p, y = select_pseudo_labels(paths, proba, threshold=0.9, max_images=4)
    assert len(p) == 4 and sorted(set(y)) == [0, 1]       # plafonné, classes équilibrées
    assert all(pp in ("p0", "p1", "p6", "p7", "p8", "p2") for pp in p)
    assert "p3" not in p and "p9" not in p                 # sous le seuil de confiance
    p2, y2 = select_pseudo_labels(paths, proba, threshold=0.999, max_images=10)
    assert p2 == [] and y2 == []

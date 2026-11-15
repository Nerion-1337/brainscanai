import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from brainscanai.clustering.cluster import run_all
from brainscanai.clustering.evaluate import evaluate_all, external_scores, map_clusters_to_classes
from brainscanai.clustering.reduce import pca_reduce, standardize
from brainscanai.data.split import assign_roles
from brainscanai.semi_supervised.weak_labels import build_weak_labels


def _blobs():
    X, y = make_blobs(n_samples=300, centers=2, n_features=20, cluster_std=1.5, random_state=0)
    return X, y


def test_all_methods_run_and_ari_is_high_on_separable_blobs():
    X, y = _blobs()
    Xs, _ = standardize(X)
    Z, _ = pca_reduce(Xs, 0.95)
    res = run_all(Z, k=2, seed=0)
    assert set(res) == {"kmeans", "gmm", "agglomerative", "dbscan"}
    table = evaluate_all(res, Z, y)
    assert table.loc["kmeans", "ARI"] > 0.9
    for r in res.values():
        assert r.confidence.shape == (300,) and r.confidence.min() >= 0 and r.confidence.max() <= 1


def test_mapping_handles_permuted_clusters():
    y = np.array([0] * 20 + [1] * 20)
    clusters = np.array([1] * 20 + [0] * 20)  # clusters inversés
    mp = map_clusters_to_classes(clusters, y)
    assert mp[1] == 0 and mp[0] == 1
    assert external_scores(y, clusters)["ARI"] == 1.0


def test_weak_labels_use_only_weak_pool_and_strong_train_mapping():
    X, y = _blobs()
    Z, _ = pca_reduce(standardize(X)[0], 0.95)
    df = pd.DataFrame(dict(image_id=range(300), path="x", label=-1, source="unlabeled"))
    lab = np.r_[np.arange(0, 20), np.arange(150, 170)]
    df.loc[lab, "label"] = y[lab]
    df.loc[lab, "source"] = "labeled"
    df = assign_roles(df, test_size=0.3, seed=0)
    res = run_all(Z, 2, 0)["kmeans"]
    weak = build_weak_labels(df, res, keep_fraction=0.8)
    assert set(weak["image_id"]).isdisjoint(df.loc[df.role != "weak_pool", "image_id"])
    acc = (weak["weak_label"].to_numpy() == y[weak["image_id"].to_numpy()]).mean()
    assert acc > 0.95 and 0.7 < len(weak) / 260 < 0.9


def test_prepare_clustering_space_l2_and_cap():
    from brainscanai.clustering.reduce import prepare_clustering_space

    X, _ = _blobs()                                   # 20 features
    Z, _ = prepare_clustering_space(X, n_components=50, l2_normalize=True)   # 50 > 20 : plafonné
    assert Z.shape == (300, 20)
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0)
    Z2, _ = prepare_clustering_space(X, n_components=5, l2_normalize=False)
    assert Z2.shape == (300, 5) and not np.allclose(np.linalg.norm(Z2, axis=1), 1.0)

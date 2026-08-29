import numpy as np

from id_features.metrics import _neighbor_support_purity, measure_linear_accessibility


def test_linear_accessibility_recovers_a_directly_encoded_feature() -> None:
    rng = np.random.default_rng(0)
    train_labels = rng.integers(0, 2, size=(200, 2)).astype(bool)
    test_labels = rng.integers(0, 2, size=(100, 2)).astype(bool)
    # Each label is literally one representation coordinate.
    train_h = train_labels.astype(float)
    test_h = test_labels.astype(float)
    result = measure_linear_accessibility(train_h, train_labels, test_h, test_labels)
    assert result.mean_auroc > 0.99


def test_neighbor_support_purity_uses_every_neighbor_through_rank() -> None:
    # Column zero is self; the remaining columns are nearest neighbors in order.
    neighbors = np.array([[0, 1, 2], [1, 0, 2], [2, 3, 0], [3, 2, 1]])
    support_ids = np.array([0, 0, 1, 1])
    mean_same, all_same = _neighbor_support_purity(neighbors, support_ids, np.array([2]))
    assert mean_same[0] == 0.5
    assert all_same[0] == 0.0

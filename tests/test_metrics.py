import numpy as np

from id_features.metrics import measure_linear_accessibility


def test_linear_accessibility_recovers_a_directly_encoded_feature() -> None:
    rng = np.random.default_rng(0)
    train_labels = rng.integers(0, 2, size=(200, 2)).astype(bool)
    test_labels = rng.integers(0, 2, size=(100, 2)).astype(bool)
    # Each label is literally one representation coordinate.
    train_h = train_labels.astype(float)
    test_h = test_labels.astype(float)
    result = measure_linear_accessibility(train_h, train_labels, test_h, test_labels)
    assert result.mean_auroc > 0.99

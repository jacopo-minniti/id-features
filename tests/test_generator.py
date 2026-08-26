import numpy as np

from id_features.generator import make_feature_matrix, sample_sparse_representations


def test_feature_matrix_has_unit_columns() -> None:
    matrix = make_feature_matrix(8, 20, 0.4, np.random.default_rng(3))
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=0), 1.0)


def test_samples_have_exactly_k_known_active_features() -> None:
    matrix = make_feature_matrix(6, 12, 0.0, np.random.default_rng(1))
    representations, labels = sample_sparse_representations(matrix, 3, 40, np.random.default_rng(2))
    assert representations.shape == (40, 6)
    assert labels.shape == (40, 12)
    np.testing.assert_array_equal(labels.sum(axis=1), np.full(40, 3))

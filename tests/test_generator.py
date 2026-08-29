import numpy as np

from id_features.generator import (
    make_feature_matrix,
    make_support_bank,
    sample_sparse_representations,
    sample_support_pool_representations,
)


def test_feature_matrix_has_unit_columns() -> None:
    matrix = make_feature_matrix(8, 20, 0.4, np.random.default_rng(3))
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=0), 1.0)


def test_samples_have_exactly_k_known_active_features() -> None:
    matrix = make_feature_matrix(6, 12, 0.0, np.random.default_rng(1))
    representations, labels = sample_sparse_representations(matrix, 3, 40, np.random.default_rng(2))
    assert representations.shape == (40, 6)
    assert labels.shape == (40, 12)
    np.testing.assert_array_equal(labels.sum(axis=1), np.full(40, 3))


def test_support_pool_is_distinct_balanced_and_reproducibly_nested() -> None:
    matrix = make_feature_matrix(8, 20, 0.0, np.random.default_rng(1))
    bank = make_support_bank(20, 3, 4, np.random.default_rng(2))
    assert bank.shape == (4, 3)
    assert len({tuple(row) for row in bank}) == 4

    def rngs() -> list[np.random.Generator]:
        return [np.random.default_rng(index + 10) for index in range(4)]

    small_h, small_ids = sample_support_pool_representations(matrix, bank, 40, rngs())
    large_h, large_ids = sample_support_pool_representations(matrix, bank, 80, rngs())
    np.testing.assert_array_equal(np.bincount(small_ids), np.full(4, 10))
    np.testing.assert_array_equal(np.bincount(large_ids), np.full(4, 20))
    for support_index in range(4):
        np.testing.assert_allclose(
            small_h[small_ids == support_index], large_h[large_ids == support_index][:10]
        )

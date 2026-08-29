import numpy as np

from id_features.manifold import (
    combined_representation,
    feature_branch,
    local_geometry_from_singular_values,
    make_residual_sparse_map,
    nearest_neighbor_support_diagnostics,
    sample_unit_sphere,
    sparse_features,
    threshold_for_expected_activity,
)


def test_residual_and_sparse_outputs_are_orthogonal_and_combined_map_retains_u() -> None:
    rng = np.random.default_rng(3)
    mapping = make_residual_sparse_map(16, 32, 4, rng)
    points = sample_unit_sphere(4, 200, rng)
    threshold = threshold_for_expected_activity(4, 32, 4)
    codes, _ = sparse_features(points, mapping.feature_directions, threshold)
    features, _ = feature_branch(codes, mapping.feature_dictionary)
    combined = combined_representation(points, mapping, features, 5.0)

    np.testing.assert_allclose(
        mapping.residual_matrix.T @ mapping.residual_matrix, np.eye(5), atol=1e-12
    )
    np.testing.assert_allclose(
        mapping.residual_matrix.T @ mapping.feature_dictionary, 0.0, atol=1e-12
    )
    np.testing.assert_allclose(combined @ mapping.residual_matrix, points, atol=1e-12)


def test_threshold_controls_expected_activity_on_the_sphere() -> None:
    rng = np.random.default_rng(7)
    points = sample_unit_sphere(4, 30_000, rng)
    directions = rng.normal(size=(64, 5))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    threshold = threshold_for_expected_activity(4, 64, 8)
    _, active = sparse_features(points, directions, threshold)
    assert abs(float(active.sum(axis=1).mean()) - 8.0) < 0.15


def test_residual_geometry_is_full_rank_even_if_sparse_branch_is_singular() -> None:
    sparse_singular_values = np.array([[3.0, 1.0, 0.0], [2.0, 0.5, 0.0]])
    sparse = local_geometry_from_singular_values(
        sparse_singular_values, 1.0, residual=False
    )
    combined = local_geometry_from_singular_values(
        sparse_singular_values, 4.0, residual=True
    )
    assert sparse.full_rank_fraction == 0.0
    assert combined.full_rank_fraction == 1.0
    assert np.isfinite(combined.mean_condition_number)


def test_nearest_neighbor_support_diagnostics_handle_empty_supports() -> None:
    active = np.array(
        [[False, False], [False, False], [True, False], [True, True]], dtype=bool
    )
    neighbors = np.array([[0, 1], [1, 0], [2, 3], [3, 2]])
    jaccard, exact = nearest_neighbor_support_diagnostics(active, neighbors)
    assert jaccard == 0.75
    assert exact == 0.5

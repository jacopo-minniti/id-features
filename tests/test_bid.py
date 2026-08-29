import numpy as np

from id_features.bid import (
    binary_diagnostics,
    population_center_and_scale,
    sign_binarize,
    support_mask_spins,
    two_bit_quantize,
)


def test_paper_two_bit_quantization_uses_natural_binary_codes() -> None:
    values = np.array([[-2.0, -0.5, 0.5, 2.0]])
    spins = two_bit_quantize(values, scale=1.0)
    expected_bits = np.array([[0, 0, 0, 1, 1, 0, 1, 1]])
    np.testing.assert_array_equal(spins, 2 * expected_bits - 1)


def test_population_center_and_scale_matches_balanced_pool_moments() -> None:
    matrix = np.eye(3)
    supports = np.array([[0, 1], [1, 2]])
    center, scale = population_center_and_scale(matrix, supports)
    np.testing.assert_allclose(center, np.array([0.5, 1.0, 0.5]))
    # Within-support energy is k/12 and between-support mean energy is 1/2.
    np.testing.assert_allclose(scale**2, ((2.0 / 12.0) + 0.5) / 3.0)


def test_binary_views_return_spins_and_expose_constant_support_bits() -> None:
    signed = sign_binarize(np.array([[-1.0, 0.0], [2.0, -3.0]]))
    np.testing.assert_array_equal(signed, np.array([[-1, 1], [1, -1]]))

    supports = np.array([[0, 2], [1, 2]])
    masks = support_mask_spins(np.array([0, 1]), supports, feature_count=4)
    diagnostics = binary_diagnostics(masks)
    assert diagnostics.bit_count == 4
    assert diagnostics.constant_bit_fraction == 0.5
    assert diagnostics.unique_pattern_count == 2

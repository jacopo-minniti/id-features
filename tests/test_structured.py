from __future__ import annotations

import numpy as np

from id_features.structured import (
    feature_frequency_diagnostics,
    feature_geometry_diagnostics,
    make_feature_groups,
    make_modular_feature_matrix,
    sample_structured_representations,
)
from id_features.structured_experiment import (
    StructuredExperimentConfig,
    conditions_for_suite,
)


def _matrix(coherence: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    groups = make_feature_groups(32, 4)
    matrix = make_modular_feature_matrix(
        12, 32, groups, coherence, np.random.default_rng(3)
    )
    return matrix, groups


def _samples(
    matrix: np.ndarray,
    groups: np.ndarray,
    n_samples: int,
    *,
    bias: float = 0.0,
    alpha: float = 0.0,
    rho: float = 0.0,
):
    return sample_structured_representations(
        matrix,
        groups,
        4,
        n_samples,
        support_module_bias=bias,
        zipf_alpha=alpha,
        amplitude_correlation=rho,
        amplitude_log_scale=0.35,
        noise_scale=0.0,
        seed_parts=(7, 11),
    )


def test_structured_samples_have_exact_k_and_are_nested_across_n() -> None:
    matrix, groups = _matrix()
    small = _samples(matrix, groups, 128, bias=2.0, alpha=1.0, rho=0.5)
    large = _samples(matrix, groups, 256, bias=2.0, alpha=1.0, rho=0.5)

    assert small.supports.shape == (128, 4)
    assert np.all(np.diff(small.supports, axis=1) > 0)
    np.testing.assert_array_equal(small.supports, large.supports[:128])
    np.testing.assert_allclose(small.representations, large.representations[:128])
    np.testing.assert_array_equal(small.context_ids, large.context_ids[:128])


def test_modularity_and_zipf_controls_change_only_the_named_diagnostics() -> None:
    matrix, groups = _matrix()
    baseline = _samples(matrix, groups, 4_000)
    modular = _samples(matrix, groups, 4_000, bias=4.0)
    zipf = _samples(matrix, groups, 4_000, alpha=1.5)

    assert modular.active_module_counts.mean() < baseline.active_module_counts.mean()
    _, baseline_cv = feature_frequency_diagnostics(baseline.supports, matrix.shape[1])
    _, zipf_cv = feature_frequency_diagnostics(zipf.supports, matrix.shape[1])
    assert zipf_cv > baseline_cv


def test_exact_shared_amplitudes_reduce_source_rank_to_active_modules() -> None:
    matrix, groups = _matrix()
    independent = _samples(matrix, groups, 512, bias=4.0, rho=0.0)
    shared = _samples(matrix, groups, 512, bias=4.0, rho=1.0)

    np.testing.assert_allclose(independent.amplitude_source_ranks, 4.0)
    np.testing.assert_allclose(independent.log_amplitude_covariance_pr, 4.0)
    np.testing.assert_array_equal(shared.amplitude_source_ranks, shared.active_module_counts)
    assert np.all(shared.log_amplitude_covariance_pr <= 4.0)


def test_coherence_increases_within_module_alignment() -> None:
    isotropic_matrix, groups = _matrix(0.0)
    coherent_matrix, _ = _matrix(0.8)
    isotropic_within, _ = feature_geometry_diagnostics(isotropic_matrix, groups)
    coherent_within, coherent_between = feature_geometry_diagnostics(coherent_matrix, groups)

    assert coherent_within > isotropic_within + 0.5
    assert coherent_within > coherent_between + 0.5


def test_part2_suites_have_predeclared_condition_counts() -> None:
    assert len(conditions_for_suite("support")) == 4
    assert len(conditions_for_suite("amplitude")) == 4
    assert len(conditions_for_suite("geometry")) == 5
    assert len(conditions_for_suite("frequency")) == 4
    assert len(conditions_for_suite("noise")) == 4
    assert len(conditions_for_suite("combined")) == 7
    config = StructuredExperimentConfig(
        suite="combined",
        representation_dim=8,
        feature_count=16,
        module_count=4,
        active_features=(2, 4),
        sample_counts=(128,),
        repeats=1,
        gride_range_max=8,
    )
    assert config.active_features == (2, 4)

"""Structured sparse-superposition generator for the Part 2 experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class StructuredSamples:
    """Representations and known latent diagnostics for one exact-k sample."""

    representations: FloatArray
    supports: IntArray
    context_ids: IntArray
    active_module_counts: FloatArray
    log_amplitude_covariance_pr: FloatArray
    amplitude_source_ranks: FloatArray
    signal_rms: float


def make_feature_groups(feature_count: int, module_count: int) -> IntArray:
    """Assign equally many interleaved features to every module."""

    if module_count < 1 or feature_count % module_count != 0:
        raise ValueError("module_count must be positive and divide feature_count")
    return np.arange(feature_count, dtype=np.int_) % module_count


def make_modular_feature_matrix(
    representation_dim: int,
    feature_count: int,
    groups: IntArray,
    coherence: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Create unit feature directions with controllable within-module coherence."""

    if groups.shape != (feature_count,):
        raise ValueError("groups must contain one module index per feature")
    if not 0.0 <= coherence < 1.0:
        raise ValueError("coherence must lie in [0, 1)")
    module_count = int(groups.max()) + 1
    independent = rng.normal(size=(representation_dim, feature_count))
    module_directions = rng.normal(size=(representation_dim, module_count))
    vectors = (
        np.sqrt(1.0 - coherence) * independent
        + np.sqrt(coherence) * module_directions[:, groups]
    )
    return vectors / np.linalg.norm(vectors, axis=0, keepdims=True)


def _within_module_ranks(groups: IntArray) -> IntArray:
    """Give every module the same 1,2,... frequency-rank profile."""

    ranks = np.empty_like(groups)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        ranks[indices] = np.arange(1, len(indices) + 1)
    return ranks


def _component_rng(seed_parts: Sequence[int], component: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([*seed_parts, component]))


def sample_structured_representations(
    matrix: FloatArray,
    groups: IntArray,
    active_features: int,
    n_samples: int,
    *,
    support_module_bias: float,
    zipf_alpha: float,
    amplitude_correlation: float,
    amplitude_log_scale: float,
    noise_scale: float,
    seed_parts: Sequence[int],
) -> StructuredSamples:
    """Generate exact-k examples with modular supports and correlated amplitudes.

    Supports use a coupled Gumbel-top-k draw.  A sample first chooses a context
    module; ``support_module_bias`` favors features in that module, while
    ``zipf_alpha`` creates the same heavy-tailed rank profile inside every
    module.  Log amplitudes combine a module-shared Gaussian and a
    feature-specific Gaussian while preserving the same marginal distribution.
    Separate deterministic random streams make increasing-N conditions nested
    and pair all structural interventions within a repeat.
    """

    representation_dim, feature_count = matrix.shape
    del representation_dim
    if groups.shape != (feature_count,):
        raise ValueError("groups must contain one module index per feature")
    if not 1 <= active_features < feature_count:
        raise ValueError("active_features must lie in [1, feature_count)")
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    if support_module_bias < 0.0 or zipf_alpha < 0.0:
        raise ValueError("support structure parameters must be nonnegative")
    if not 0.0 <= amplitude_correlation <= 1.0:
        raise ValueError("amplitude_correlation must lie in [0, 1]")
    if amplitude_log_scale <= 0.0 or noise_scale < 0.0:
        raise ValueError("amplitude_log_scale must be positive and noise_scale nonnegative")

    module_count = int(groups.max()) + 1
    context_rng = _component_rng(seed_parts, 1)
    support_rng = _component_rng(seed_parts, 2)
    shared_rng = _component_rng(seed_parts, 3)
    innovation_rng = _component_rng(seed_parts, 4)
    noise_rng = _component_rng(seed_parts, 5)

    context_ids = context_rng.integers(module_count, size=n_samples, dtype=np.int_)
    ranks = _within_module_ranks(groups)
    log_frequency_weights = -zipf_alpha * np.log(ranks)
    support_scores = support_rng.gumbel(size=(n_samples, feature_count))
    support_scores += log_frequency_weights[None, :]
    if support_module_bias > 0.0:
        support_scores += support_module_bias * (groups[None, :] == context_ids[:, None])
    supports = np.argpartition(
        support_scores, kth=feature_count - active_features, axis=1
    )[:, -active_features:]
    supports.sort(axis=1)

    active_groups = groups[supports]
    shared_latents = shared_rng.normal(size=(n_samples, module_count))
    innovations = innovation_rng.normal(size=(n_samples, active_features))
    shared_values = np.take_along_axis(shared_latents, active_groups, axis=1)
    standardized_log_amplitudes = (
        np.sqrt(amplitude_correlation) * shared_values
        + np.sqrt(1.0 - amplitude_correlation) * innovations
    )
    amplitudes = np.exp(
        amplitude_log_scale * standardized_log_amplitudes
        - 0.5 * amplitude_log_scale**2
    )

    selected_vectors = matrix[:, supports]
    representations = np.einsum("dnk,nk->nd", selected_vectors, amplitudes, optimize=True)
    centered_signal = representations - representations.mean(axis=0, keepdims=True)
    signal_rms = float(np.sqrt(np.mean(centered_signal**2)))
    if noise_scale > 0.0:
        representations = representations + noise_scale * signal_rms * noise_rng.normal(
            size=representations.shape
        )

    module_counts = np.zeros((n_samples, module_count), dtype=np.int_)
    sample_indices = np.repeat(np.arange(n_samples), active_features)
    np.add.at(module_counts, (sample_indices, active_groups.ravel()), 1)
    active_module_counts = (module_counts > 0).sum(axis=1).astype(np.float64)
    same_module_ordered_pairs = np.sum(module_counts * (module_counts - 1), axis=1)
    pr_denominator = active_features + amplitude_correlation**2 * same_module_ordered_pairs
    log_amplitude_covariance_pr = active_features**2 / pr_denominator
    amplitude_source_ranks = np.where(
        np.isclose(amplitude_correlation, 1.0), active_module_counts, active_features
    ).astype(np.float64)

    return StructuredSamples(
        representations=np.asarray(representations, dtype=np.float64),
        supports=supports,
        context_ids=context_ids,
        active_module_counts=active_module_counts,
        log_amplitude_covariance_pr=np.asarray(log_amplitude_covariance_pr),
        amplitude_source_ranks=amplitude_source_ranks,
        signal_rms=signal_rms,
    )


def feature_geometry_diagnostics(
    matrix: FloatArray, groups: IntArray
) -> tuple[float, float]:
    """Return mean within-module and between-module feature dot products."""

    gram = matrix.T @ matrix
    same = groups[:, None] == groups[None, :]
    diagonal = np.eye(len(groups), dtype=bool)
    within = gram[same & ~diagonal]
    between = gram[~same]
    return float(within.mean()), float(between.mean())


def feature_frequency_diagnostics(
    supports: IntArray, feature_count: int
) -> tuple[float, float]:
    """Return normalized activation entropy and coefficient of variation."""

    counts = np.bincount(supports.ravel(), minlength=feature_count).astype(np.float64)
    probabilities = counts / counts.sum()
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    normalized_entropy = entropy / np.log(feature_count)
    frequency_cv = float(counts.std(ddof=0) / counts.mean())
    return normalized_entropy, frequency_cv

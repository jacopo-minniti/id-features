"""The no-learning sparse linear representation used by both sweeps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _unit_columns(vectors: FloatArray) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=0, keepdims=True)
    return vectors / norms


def make_feature_matrix(
    representation_dim: int,
    feature_count: int,
    correlation: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Create unit-norm feature directions with a tunable common component.

    At correlation zero this is an isotropic random dictionary.  Higher values
    add the same random direction to every feature before re-normalization.
    The reported empirical dot products, rather than ``correlation`` itself,
    are the geometry diagnostic because normalization changes its exact value.
    """

    independent = rng.normal(size=(representation_dim, feature_count))
    common = rng.normal(size=(representation_dim, 1))
    vectors = np.sqrt(1.0 - correlation) * independent + np.sqrt(correlation) * common
    return _unit_columns(vectors)


def sample_sparse_representations(
    matrix: FloatArray,
    active_features: int,
    n_samples: int,
    rng: np.random.Generator,
    amplitude_low: float = 0.5,
    amplitude_high: float = 1.5,
) -> tuple[FloatArray, BoolArray]:
    """Sample exactly-``k`` positive latent features and their representations."""

    _, feature_count = matrix.shape
    if not 1 <= active_features < feature_count:
        raise ValueError("active_features must be in [1, feature_count)")
    if not amplitude_low > 0 or not amplitude_high > amplitude_low:
        raise ValueError("amplitudes must be strictly positive and ordered")

    # argpartition gives a uniformly random size-k support without a Python loop.
    support_scores = rng.random((n_samples, feature_count))
    supports = np.argpartition(support_scores, kth=active_features - 1, axis=1)[
        :, :active_features
    ]
    labels = np.zeros((n_samples, feature_count), dtype=bool)
    labels[np.arange(n_samples)[:, None], supports] = True

    latents = np.zeros((n_samples, feature_count), dtype=np.float64)
    latents[labels] = rng.uniform(amplitude_low, amplitude_high, size=n_samples * active_features)
    return latents @ matrix.T, labels


def mean_off_diagonal_dot_product(matrix: FloatArray) -> float:
    """Average pairwise feature alignment, the realized interference control."""

    gram = matrix.T @ matrix
    n_features = gram.shape[0]
    return float((gram.sum() - n_features) / (n_features * (n_features - 1)))

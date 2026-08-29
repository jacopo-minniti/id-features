"""The no-learning sparse linear representation used by both sweeps."""

from __future__ import annotations

from math import comb
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]


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


def make_support_bank(
    feature_count: int,
    active_features: int,
    pool_size: int,
    rng: np.random.Generator,
) -> IntArray:
    """Draw a uniform bank of distinct exact-k supports.

    Taking prefixes of one largest bank gives nested support pools across B.
    """

    if not 1 <= active_features < feature_count:
        raise ValueError("active_features must be in [1, feature_count)")
    if not 1 <= pool_size <= comb(feature_count, active_features):
        raise ValueError("pool_size must fit the number of distinct exact-k supports")

    supports: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(supports) < pool_size:
        support = tuple(sorted(rng.choice(feature_count, size=active_features, replace=False)))
        if support not in seen:
            seen.add(support)
            supports.append(support)
    return np.asarray(supports, dtype=np.int_)


def sample_support_pool_representations(
    matrix: FloatArray,
    supports: IntArray,
    n_samples: int,
    support_rngs: Sequence[np.random.Generator],
    amplitude_low: float = 0.5,
    amplitude_high: float = 1.5,
) -> tuple[FloatArray, IntArray]:
    """Sample equally from a fixed support bank and return each support identity.

    A separate resettable RNG per support lets larger-N conditions extend the
    exact samples used at smaller N, while prefixes of ``supports`` nest B.
    """

    supports = np.asarray(supports, dtype=np.int_)
    if supports.ndim != 2 or len(supports) < 1:
        raise ValueError("supports must be a non-empty two-dimensional array")
    if len(support_rngs) != len(supports):
        raise ValueError("one RNG is required for every support")
    if n_samples % len(supports) != 0:
        raise ValueError("n_samples must be divisible by the support pool size")
    if supports.min() < 0 or supports.max() >= matrix.shape[1]:
        raise ValueError("support index lies outside the feature matrix")
    if not amplitude_low > 0 or not amplitude_high > amplitude_low:
        raise ValueError("amplitudes must be strictly positive and ordered")

    per_support = n_samples // len(supports)
    representations: list[FloatArray] = []
    support_ids: list[IntArray] = []
    for support_index, (support, rng) in enumerate(zip(supports, support_rngs, strict=True)):
        amplitudes = rng.uniform(
            amplitude_low, amplitude_high, size=(per_support, supports.shape[1])
        )
        representations.append(amplitudes @ matrix[:, support].T)
        support_ids.append(np.full(per_support, support_index, dtype=np.int_))
    return np.concatenate(representations), np.concatenate(support_ids)


def mean_off_diagonal_dot_product(matrix: FloatArray) -> float:
    """Average pairwise feature alignment, the realized interference control."""

    gram = matrix.T @ matrix
    n_features = gram.shape[0]
    return float((gram.sum() - n_features) / (n_features * (n_features - 1)))

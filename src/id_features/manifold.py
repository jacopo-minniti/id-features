"""Boundaryless sparse-feature manifolds with known population dimension."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaincinv


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ResidualSparseMap:
    """A residual embedding and a sparse ReLU feature map in orthogonal subspaces."""

    residual_matrix: FloatArray
    feature_directions: FloatArray
    feature_dictionary: FloatArray


@dataclass(frozen=True)
class LocalGeometry:
    """Exact local differential diagnostics for one representation branch."""

    full_rank_fraction: float
    mean_condition_number: float
    p95_condition_number: float
    log_volume_std: float


def sample_unit_sphere(
    intrinsic_dim: int, n_samples: int, rng: np.random.Generator
) -> FloatArray:
    """Draw uniformly from the unit sphere S^d in R^(d+1)."""

    if intrinsic_dim < 1:
        raise ValueError("intrinsic_dim must be positive")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    points = rng.normal(size=(n_samples, intrinsic_dim + 1))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def make_residual_sparse_map(
    representation_dim: int,
    feature_count: int,
    intrinsic_dim: int,
    rng: np.random.Generator,
) -> ResidualSparseMap:
    """Construct ``Q``, ``A``, and ``W`` for ``h(u)=Qu+gamma W ReLU(Au-tau)``.

    ``Q`` has orthonormal columns and every column of ``W`` lies in the
    orthogonal complement of ``Q``.  Consequently the combined map is
    injective for every finite ``gamma``: its residual component always
    retains ``u`` exactly.
    """

    latent_ambient_dim = intrinsic_dim + 1
    if representation_dim <= latent_ambient_dim:
        raise ValueError("representation_dim must leave a non-empty feature subspace")
    if feature_count < 1:
        raise ValueError("feature_count must be positive")

    orthogonal, _ = np.linalg.qr(rng.normal(size=(representation_dim, representation_dim)))
    residual = orthogonal[:, :latent_ambient_dim]
    complement = orthogonal[:, latent_ambient_dim:]

    directions = rng.normal(size=(feature_count, latent_ambient_dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    complement_coordinates = rng.normal(size=(complement.shape[1], feature_count))
    complement_coordinates /= np.linalg.norm(complement_coordinates, axis=0, keepdims=True)
    dictionary = complement @ complement_coordinates
    return ResidualSparseMap(residual, directions, dictionary)


def threshold_for_expected_activity(
    intrinsic_dim: int, feature_count: int, expected_active: int
) -> float:
    """Return ``tau`` such that each sphere point activates ``expected_active`` features on average.

    For a fixed unit vector ``a`` and uniform ``u`` on S^d,
    ``(a dot u)^2`` follows ``Beta(1/2, d/2)``.  The requested experiments use
    an activation probability below one half, hence a non-negative threshold.
    """

    if intrinsic_dim < 1:
        raise ValueError("intrinsic_dim must be positive")
    if not 1 <= expected_active < feature_count / 2:
        raise ValueError("expected_active must lie in [1, feature_count / 2)")
    activation_probability = expected_active / feature_count
    squared_threshold = betaincinv(0.5, intrinsic_dim / 2.0, 1.0 - 2.0 * activation_probability)
    return float(np.sqrt(squared_threshold))


def sparse_features(
    points: FloatArray, feature_directions: FloatArray, threshold: float
) -> tuple[FloatArray, BoolArray]:
    """Evaluate the continuous sparse code ``ReLU(Au-tau)`` and its support."""

    preactivations = points @ feature_directions.T - threshold
    active = preactivations > 0.0
    return np.maximum(preactivations, 0.0), active


def feature_branch(
    sparse_codes: FloatArray, feature_dictionary: FloatArray
) -> tuple[FloatArray, float]:
    """Map sparse codes to representation space and normalize centered RMS variation."""

    raw = sparse_codes @ feature_dictionary.T
    centered = raw - raw.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError("feature branch has zero or non-finite variation")
    # The mean is only a translation, so retaining it does not affect distances.
    return raw / rms, rms


def combined_representation(
    points: FloatArray,
    mapping: ResidualSparseMap,
    normalized_feature_branch: FloatArray,
    feature_strength: float,
) -> FloatArray:
    """Evaluate the known-dimension residual-plus-sparse representation."""

    return points @ mapping.residual_matrix.T + feature_strength * normalized_feature_branch


def _tangent_basis(point: FloatArray) -> FloatArray:
    """Return an orthonormal basis of the sphere tangent space at ``point``."""

    ambient_dim = len(point)
    omitted = int(np.argmax(np.abs(point)))
    coordinate_indices = [index for index in range(ambient_dim) if index != omitted]
    raw = np.eye(ambient_dim)[:, coordinate_indices]
    raw -= point[:, None] * point[coordinate_indices][None, :]
    basis, _ = np.linalg.qr(raw, mode="reduced")
    return basis


def sparse_tangent_singular_values(
    points: FloatArray,
    active_sets: BoolArray,
    mapping: ResidualSparseMap,
    feature_rms: float,
) -> FloatArray:
    """Compute singular values of the normalized sparse branch on sphere tangents."""

    if len(points) != len(active_sets):
        raise ValueError("points and active_sets must have equal length")
    intrinsic_dim = points.shape[1] - 1
    singular_values = np.empty((len(points), intrinsic_dim), dtype=np.float64)
    for sample_index, (point, active) in enumerate(zip(points, active_sets, strict=True)):
        tangent = _tangent_basis(point)
        if not np.any(active):
            singular_values[sample_index] = 0.0
            continue
        derivative = (
            mapping.feature_dictionary[:, active]
            @ mapping.feature_directions[active]
            @ tangent
            / feature_rms
        )
        values = np.linalg.svd(derivative, compute_uv=False)
        singular_values[sample_index] = values[:intrinsic_dim]
    return singular_values


def local_geometry_from_singular_values(
    sparse_singular_values: FloatArray, feature_strength: float, *, residual: bool
) -> LocalGeometry:
    """Summarize rank, anisotropy, and volume variation of a local Jacobian.

    Orthogonality of the residual and sparse output subspaces makes the
    combined singular values exactly ``sqrt(1 + gamma^2 sigma_i^2)``.
    """

    values = np.asarray(sparse_singular_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("sparse_singular_values must be a non-empty matrix")
    if feature_strength < 0.0:
        raise ValueError("feature_strength must be non-negative")
    if residual:
        local_values = np.sqrt(1.0 + (feature_strength * values) ** 2)
    else:
        local_values = feature_strength * values

    largest = local_values[:, 0]
    smallest = local_values[:, -1]
    tolerance = np.finfo(float).eps * max(local_values.shape) * np.maximum(largest, 1.0)
    full_rank = smallest > tolerance
    condition_numbers = np.full(len(values), np.inf)
    condition_numbers[full_rank] = largest[full_rank] / smallest[full_rank]
    finite_condition_numbers = condition_numbers[np.isfinite(condition_numbers)]
    mean_condition = (
        float(finite_condition_numbers.mean()) if len(finite_condition_numbers) else float("inf")
    )
    p95_condition = (
        float(np.quantile(finite_condition_numbers, 0.95))
        if len(finite_condition_numbers)
        else float("inf")
    )
    with np.errstate(divide="ignore"):
        log_volumes = np.log(local_values).sum(axis=1)
    finite_log_volumes = log_volumes[np.isfinite(log_volumes)]
    log_volume_std = (
        float(finite_log_volumes.std(ddof=0)) if len(finite_log_volumes) else float("inf")
    )
    return LocalGeometry(
        full_rank_fraction=float(full_rank.mean()),
        mean_condition_number=mean_condition,
        p95_condition_number=p95_condition,
        log_volume_std=log_volume_std,
    )


def nearest_neighbor_support_diagnostics(
    active_sets: BoolArray, neighbor_indices: NDArray[np.int_]
) -> tuple[float, float]:
    """Return mean 1-NN support Jaccard and exact-support-match fraction."""

    nearest = neighbor_indices[:, 1]
    neighbor_active = active_sets[nearest]
    intersection = np.logical_and(active_sets, neighbor_active).sum(axis=1)
    union = np.logical_or(active_sets, neighbor_active).sum(axis=1)
    jaccard = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union > 0,
    )
    exact_match = np.all(active_sets == neighbor_active, axis=1)
    return float(jaccard.mean()), float(exact_match.mean())

"""Binary encodings and DADApy BID measurement helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
SpinArray = NDArray[np.int8]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class BinaryDiagnostics:
    bit_count: int
    mean_absolute_imbalance: float
    maximum_absolute_imbalance: float
    constant_bit_fraction: float
    unique_pattern_count: int


def population_center_and_scale(
    matrix: FloatArray, supports: IntArray, amplitude_variance: float = 1.0 / 12.0
) -> tuple[FloatArray, float]:
    """Return the exact balanced-pool mean and global RMS around that mean.

    Amplitudes have population mean one. The scalar scale follows the paper's
    layer-wide standard deviation after centering, while using population
    moments keeps this controlled audit free of calibration-sample noise.
    """

    supports = np.asarray(supports, dtype=np.int_)
    if supports.ndim != 2 or supports.shape[0] < 1:
        raise ValueError("supports must be a non-empty two-dimensional array")
    if supports.min() < 0 or supports.max() >= matrix.shape[1]:
        raise ValueError("support index lies outside the feature matrix")
    selected = matrix[:, supports]
    support_means = selected.sum(axis=2).T
    center = support_means.mean(axis=0)
    within_energy = amplitude_variance * np.mean(np.sum(selected**2, axis=(0, 2)))
    between_energy = np.mean(np.sum((support_means - center) ** 2, axis=1))
    scale = float(np.sqrt((within_energy + between_energy) / matrix.shape[0]))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("population scale must be finite and positive")
    return center, scale


def sign_binarize(values: FloatArray) -> SpinArray:
    """Encode real values as spins in {-1,+1}, with zero assigned to +1."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    return np.where(values >= 0.0, 1, -1).astype(np.int8)


def two_bit_quantize(values: FloatArray, scale: float) -> SpinArray:
    """Apply the paper's 00/01/10/11 bins at -sigma, 0, and +sigma."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    categories = np.digitize(values, (-scale, 0.0, scale), right=False)
    high_bit = (categories >> 1) & 1
    low_bit = categories & 1
    bits = np.stack((high_bit, low_bit), axis=2).reshape(len(values), -1)
    return (2 * bits - 1).astype(np.int8)


def support_mask_spins(
    support_ids: IntArray, supports: IntArray, feature_count: int
) -> SpinArray:
    """Encode the exact active support as one spin per latent feature."""

    support_ids = np.asarray(support_ids, dtype=np.int_)
    supports = np.asarray(supports, dtype=np.int_)
    if support_ids.ndim != 1 or supports.ndim != 2:
        raise ValueError("support_ids and supports have incompatible shapes")
    if support_ids.min() < 0 or support_ids.max() >= len(supports):
        raise ValueError("support ID lies outside the support bank")
    mask = np.zeros((len(support_ids), feature_count), dtype=np.int8)
    sample_indices = np.arange(len(support_ids))[:, None]
    mask[sample_indices, supports[support_ids]] = 1
    return 2 * mask - 1


def binary_diagnostics(spins: SpinArray) -> BinaryDiagnostics:
    """Summarize marginal polarization and binary-code resolution."""

    spins = np.asarray(spins)
    if spins.ndim != 2 or len(spins) < 2:
        raise ValueError("spins must contain at least two binary samples")
    if not np.all(np.isin(spins, (-1, 1))):
        raise ValueError("spins must contain only -1 and +1")
    probabilities = np.mean(spins == 1, axis=0)
    imbalances = np.abs(probabilities - 0.5)
    constant = (probabilities == 0.0) | (probabilities == 1.0)
    return BinaryDiagnostics(
        bit_count=spins.shape[1],
        mean_absolute_imbalance=float(imbalances.mean()),
        maximum_absolute_imbalance=float(imbalances.max()),
        constant_bit_fraction=float(constant.mean()),
        unique_pattern_count=int(np.unique(spins, axis=0).shape[0]),
    )


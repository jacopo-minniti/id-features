"""Explicit, controlled configurations for the first three hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Literal


ExperimentKind = Literal["load-capacity", "load-ratio", "geometry-control", "support-pool"]


@dataclass(frozen=True)
class ExperimentConfig:
    """All settings needed to reproduce one controlled sweep.

    ``n_train`` and ``n_test`` are generated independently but share the same
    feature matrix inside a replicate.  This makes the linear probe genuinely
    held out while retaining known ground-truth labels.
    """

    experiment: ExperimentKind
    representation_dim: int = 32
    feature_count: int = 128
    active_features: tuple[int, ...] = (1, 2, 4, 8, 16)
    capacity_values: tuple[int, ...] = (64, 128, 256)
    representation_dims: tuple[int, ...] = (16, 24, 32, 48, 64)
    correlations: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9)
    support_pool_sizes: tuple[int, ...] = (1, 4, 16, 64)
    id_sample_values: tuple[int, ...] = (1_024, 2_048, 4_096)
    fixed_k: int = 8
    n_id_samples: int = 4_000
    n_train: int = 4_000
    n_test: int = 4_000
    repeats: int = 5
    gride_range_max: int = 64
    gride_n_jobs: int = 1
    id_match_relative_tolerance: float = 0.10
    auroc_effect_threshold: float = 0.02
    seed: int = 0

    def __post_init__(self) -> None:
        if self.representation_dim < 2:
            raise ValueError("representation_dim must be at least 2")
        if self.feature_count <= self.representation_dim:
            raise ValueError("feature_count must exceed representation_dim (superposition regime)")
        if self.n_id_samples < 4 or self.n_train < 2 or self.n_test < 2:
            raise ValueError("sample counts are too small")
        if self.repeats < 1:
            raise ValueError("repeats must be positive")
        if self.gride_range_max < 2:
            raise ValueError("gride_range_max must be at least 2")
        if self.gride_range_max >= self.n_id_samples:
            raise ValueError("gride_range_max must be smaller than n_id_samples")
        if self.gride_n_jobs < 1:
            raise ValueError("gride_n_jobs must be positive")
        if any(k < 1 or k >= self.feature_count for k in self.active_features):
            raise ValueError("every active feature count must be in [1, feature_count)")
        if self.experiment == "load-capacity":
            if any(m <= self.representation_dim for m in self.capacity_values):
                raise ValueError("every capacity value must exceed representation_dim")
            if any(k >= min(self.capacity_values) for k in self.active_features):
                raise ValueError("every active feature count must fit every capacity value")
        if self.experiment == "load-ratio" and any(
            d < 2 or d >= self.feature_count for d in self.representation_dims
        ):
            raise ValueError("every representation dimension must lie in [2, feature_count)")
        if self.fixed_k < 1 or self.fixed_k >= self.feature_count:
            raise ValueError("fixed_k must be in [1, feature_count)")
        if any(not 0.0 <= rho < 1.0 for rho in self.correlations):
            raise ValueError("correlations must lie in [0, 1)")
        if self.experiment == "geometry-control" and 0.0 not in self.correlations:
            raise ValueError("geometry-control requires rho=0 as its isotropic baseline")
        if self.experiment == "support-pool":
            if any(k > self.representation_dim for k in self.active_features):
                raise ValueError("support-pool requires k <= representation_dim")
            if any(b < 1 for b in self.support_pool_sizes):
                raise ValueError("support pool sizes must be positive")
            if any(
                b > comb(self.feature_count, k)
                for b in self.support_pool_sizes
                for k in self.active_features
            ):
                raise ValueError("support pool size exceeds the number of distinct supports")
            if any(n <= self.gride_range_max for n in self.id_sample_values):
                raise ValueError("every support-pool N must exceed gride_range_max")
            if any(n % b != 0 for n in self.id_sample_values for b in self.support_pool_sizes):
                raise ValueError("every support-pool N must be divisible by every support pool size")
        if not 0.0 < self.id_match_relative_tolerance < 1.0:
            raise ValueError("id_match_relative_tolerance must lie in (0, 1)")
        if not 0.0 < self.auroc_effect_threshold < 1.0:
            raise ValueError("auroc_effect_threshold must lie in (0, 1)")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ManifoldGeometryConfig:
    """Controls for the boundaryless, known-ID sparse-feature geometry audit."""

    representation_dim: int = 64
    feature_count: int = 256
    intrinsic_dims: tuple[int, ...] = (4, 8, 16)
    sample_counts: tuple[int, ...] = (4_096, 16_384)
    activity_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    feature_strengths: tuple[float, ...] = (0.25, 1.0, 4.0)
    repeats: int = 3
    gride_range_max: int = 64
    gride_n_jobs: int = 1
    jacobian_samples: int = 128
    id_relative_tolerance: float = 0.10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.representation_dim < 3:
            raise ValueError("representation_dim must be at least 3")
        if self.feature_count < 3:
            raise ValueError("feature_count must be at least 3")
        if not self.intrinsic_dims or any(d < 1 for d in self.intrinsic_dims):
            raise ValueError("intrinsic_dims must contain positive values")
        if any(d + 1 >= self.representation_dim for d in self.intrinsic_dims):
            raise ValueError("every intrinsic dimension must leave a feature subspace")
        if not self.sample_counts or any(n <= self.gride_range_max for n in self.sample_counts):
            raise ValueError("every sample count must exceed gride_range_max")
        if tuple(sorted(set(self.sample_counts))) != self.sample_counts:
            raise ValueError("sample_counts must be unique and increasing")
        if not self.activity_multipliers or any(value <= 0.0 for value in self.activity_multipliers):
            raise ValueError("activity_multipliers must be positive")
        expected_activities = {
            max(1, int(round(d * multiplier)))
            for d in self.intrinsic_dims
            for multiplier in self.activity_multipliers
        }
        if any(active >= self.feature_count / 2 for active in expected_activities):
            raise ValueError("every expected activity must be below feature_count / 2")
        if not self.feature_strengths or any(value <= 0.0 for value in self.feature_strengths):
            raise ValueError("feature_strengths must be positive")
        if self.repeats < 1:
            raise ValueError("repeats must be positive")
        if self.gride_range_max < 2:
            raise ValueError("gride_range_max must be at least 2")
        if self.gride_n_jobs < 1:
            raise ValueError("gride_n_jobs must be positive")
        if self.jacobian_samples < 1:
            raise ValueError("jacobian_samples must be positive")
        if not 0.0 < self.id_relative_tolerance < 1.0:
            raise ValueError("id_relative_tolerance must lie in (0, 1)")

    def expected_activities(self, intrinsic_dim: int) -> tuple[int, ...]:
        """Map relative activity controls to unique integer target counts."""

        return tuple(
            dict.fromkeys(max(1, int(round(intrinsic_dim * value))) for value in self.activity_multipliers)
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BidAuditConfig:
    """Controls for comparing binary ID with known continuous ID and GRIDE."""

    representation_dim: int = 64
    feature_count: int = 256
    active_features: tuple[int, ...] = (2, 4, 8, 16)
    support_pool_sizes: tuple[int, ...] = (1, 64)
    sample_count: int = 2_560
    repeats: int = 3
    alpha_max_values: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 1.0)
    primary_alpha_max: float = 0.2
    bid_steps: int = 100_000
    bid_delta: float = 5e-3
    gride_range_max: int = 64
    gride_n_jobs: int = 1
    id_relative_tolerance: float = 0.10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.representation_dim < 2:
            raise ValueError("representation_dim must be at least 2")
        if self.feature_count <= self.representation_dim:
            raise ValueError("feature_count must exceed representation_dim")
        if not self.active_features or any(
            k < 1 or k > self.representation_dim or k >= self.feature_count
            for k in self.active_features
        ):
            raise ValueError("active_features must lie in [1, representation_dim]")
        if not self.support_pool_sizes or any(b < 1 for b in self.support_pool_sizes):
            raise ValueError("support_pool_sizes must be positive")
        if 1 not in self.support_pool_sizes:
            raise ValueError("the BID audit requires a fixed-support B=1 control")
        if any(
            b > comb(self.feature_count, k)
            for b in self.support_pool_sizes
            for k in self.active_features
        ):
            raise ValueError("support pool size exceeds the number of distinct supports")
        if self.sample_count <= self.gride_range_max:
            raise ValueError("sample_count must exceed gride_range_max")
        if any(self.sample_count % b != 0 for b in self.support_pool_sizes):
            raise ValueError("sample_count must be divisible by every support pool size")
        if self.repeats < 1:
            raise ValueError("repeats must be positive")
        if not self.alpha_max_values or any(
            not 0.0 < alpha <= 1.0 for alpha in self.alpha_max_values
        ):
            raise ValueError("alpha_max_values must lie in (0, 1]")
        if tuple(sorted(set(self.alpha_max_values))) != self.alpha_max_values:
            raise ValueError("alpha_max_values must be unique and increasing")
        if self.primary_alpha_max not in self.alpha_max_values:
            raise ValueError("primary_alpha_max must be one of alpha_max_values")
        if self.bid_steps < 1_000:
            raise ValueError("bid_steps must be at least 1000")
        if self.bid_delta <= 0.0:
            raise ValueError("bid_delta must be positive")
        if self.gride_range_max < 2:
            raise ValueError("gride_range_max must be at least 2")
        if self.gride_n_jobs < 1:
            raise ValueError("gride_n_jobs must be positive")
        if not 0.0 < self.id_relative_tolerance < 1.0:
            raise ValueError("id_relative_tolerance must lie in (0, 1)")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

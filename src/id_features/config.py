"""Explicit, controlled configurations for the first three hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ExperimentKind = Literal["load-capacity", "load-ratio", "geometry-control"]


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
        if not 0.0 < self.id_match_relative_tolerance < 1.0:
            raise ValueError("id_match_relative_tolerance must lie in (0, 1)")
        if not 0.0 < self.auroc_effect_threshold < 1.0:
            raise ValueError("auroc_effect_threshold must lie in (0, 1)")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

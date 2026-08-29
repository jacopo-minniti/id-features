"""Controlled Part 2 experiments for structured sparse representations."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from .metrics import measure_gride
from .structured import (
    feature_frequency_diagnostics,
    feature_geometry_diagnostics,
    make_feature_groups,
    make_modular_feature_matrix,
    sample_structured_representations,
)


StructuredSuite = Literal["support", "amplitude", "geometry", "frequency", "noise", "combined"]


@dataclass(frozen=True)
class StructuredExperimentConfig:
    """Complete configuration for one controlled Part 2 suite."""

    suite: StructuredSuite
    representation_dim: int = 32
    feature_count: int = 256
    module_count: int = 8
    active_features: tuple[int, ...] = (2, 4, 8)
    sample_counts: tuple[int, ...] = (4_096, 16_384)
    repeats: int = 5
    gride_range_max: int = 64
    gride_n_jobs: int = 1
    amplitude_log_scale: float = 0.35
    geometry_diagnostic_samples: int = 256
    seed: int = 0

    def __post_init__(self) -> None:
        if self.suite not in {"support", "amplitude", "geometry", "frequency", "noise", "combined"}:
            raise ValueError("unknown structured experiment suite")
        if self.representation_dim < 2 or self.feature_count <= self.representation_dim:
            raise ValueError("Part 2 requires an overcomplete m > D representation")
        if self.module_count < 2 or self.feature_count % self.module_count != 0:
            raise ValueError("module_count must divide feature_count")
        if not self.active_features or any(
            k < 1 or k > self.representation_dim for k in self.active_features
        ):
            raise ValueError("every k must lie in [1, D]")
        if tuple(sorted(set(self.sample_counts))) != self.sample_counts:
            raise ValueError("sample_counts must be unique and increasing")
        if any(n <= self.gride_range_max for n in self.sample_counts):
            raise ValueError("every sample count must exceed the largest GRIDE rank")
        if self.repeats < 1 or self.gride_n_jobs < 1:
            raise ValueError("repeats and gride_n_jobs must be positive")
        if self.amplitude_log_scale <= 0.0 or self.geometry_diagnostic_samples < 1:
            raise ValueError("amplitude scale and diagnostic sample count must be positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StructuredCondition:
    label: str
    support_module_bias: float
    zipf_alpha: float
    amplitude_correlation: float
    coherence: float
    noise_scale: float = 0.0


@dataclass(frozen=True)
class StructuredArtifacts:
    output_dir: Path
    profile_path: Path
    summary_path: Path
    interpretation_path: Path
    figure_path: Path


def conditions_for_suite(suite: StructuredSuite) -> tuple[StructuredCondition, ...]:
    """Return predeclared single-factor sweeps and the named combined controls."""

    if suite == "support":
        return tuple(
            StructuredCondition(f"support_bias={value:g}", value, 0.0, 0.0, 0.0)
            for value in (0.0, 1.0, 2.0, 4.0)
        )
    if suite == "amplitude":
        return tuple(
            StructuredCondition(f"amplitude_rho={value:g}", 4.0, 0.0, value, 0.0)
            for value in (0.0, 0.5, 0.9, 1.0)
        )
    if suite == "geometry":
        return tuple(
            StructuredCondition(f"coherence={value:g}", 4.0, 0.0, 0.0, value)
            for value in (0.0, 0.25, 0.5, 0.75, 0.9)
        )
    if suite == "frequency":
        return tuple(
            StructuredCondition(f"zipf_alpha={value:g}", 4.0, value, 0.0, 0.0)
            for value in (0.0, 0.5, 1.0, 1.5)
        )
    if suite == "noise":
        return tuple(
            StructuredCondition(f"noise={value:g}", 4.0, 1.0, 0.9, 0.75, value)
            for value in (0.0, 0.01, 0.05, 0.2)
        )
    return (
        StructuredCondition("baseline", 0.0, 0.0, 0.0, 0.0),
        StructuredCondition("modular_support", 4.0, 0.0, 0.0, 0.0),
        StructuredCondition("zipf_frequencies", 0.0, 1.0, 0.0, 0.0),
        StructuredCondition("shared_amplitudes", 4.0, 0.0, 0.9, 0.0),
        StructuredCondition("coherent_geometry", 4.0, 0.0, 0.0, 0.75),
        StructuredCondition("llm_like", 4.0, 1.0, 0.9, 0.75),
        StructuredCondition("llm_like_exact_shared", 4.0, 1.0, 1.0, 0.75),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _active_geometry_diagnostics(
    matrix: np.ndarray, supports: np.ndarray, sample_count: int
) -> tuple[float, float, float, float]:
    conditions: list[float] = []
    smallest: list[float] = []
    full_rank = 0
    for support in supports[:sample_count]:
        singular_values = np.linalg.svd(matrix[:, support], compute_uv=False)
        smallest.append(float(singular_values[-1]))
        conditions.append(float(singular_values[0] / max(singular_values[-1], 1e-15)))
        full_rank += int(singular_values[-1] > 1e-10)
    return (
        full_rank / min(sample_count, len(supports)),
        float(np.mean(conditions)),
        float(np.quantile(conditions, 0.95)),
        float(np.min(smallest)),
    )


def _neighbor_diagnostics(
    supports: np.ndarray,
    context_ids: np.ndarray,
    groups: np.ndarray,
    neighbor_indices: np.ndarray,
) -> tuple[float, float, float, float]:
    one_nn = neighbor_indices[:, 1]
    neighbor_supports = supports[one_nn]
    exact_support = float(np.mean(np.all(supports == neighbor_supports, axis=1)))
    intersections = (
        supports[:, :, None] == neighbor_supports[:, None, :]
    ).any(axis=2).sum(axis=1)
    support_jaccard = float(np.mean(intersections / (2 * supports.shape[1] - intersections)))

    active_groups = groups[supports]
    neighbor_groups = groups[neighbor_supports]
    module_count = int(groups.max()) + 1
    module_masks = np.zeros((len(supports), module_count), dtype=bool)
    neighbor_masks = np.zeros_like(module_masks)
    rows = np.repeat(np.arange(len(supports)), supports.shape[1])
    module_masks[rows, active_groups.ravel()] = True
    neighbor_masks[rows, neighbor_groups.ravel()] = True
    module_intersection = np.logical_and(module_masks, neighbor_masks).sum(axis=1)
    module_union = np.logical_or(module_masks, neighbor_masks).sum(axis=1)
    module_jaccard = float(np.mean(module_intersection / module_union))
    same_context = float(np.mean(context_ids == context_ids[one_nn]))
    return exact_support, support_jaccard, module_jaccard, same_context


def _rank_context_fraction(
    context_ids: np.ndarray, neighbor_indices: np.ndarray, rank: int
) -> float:
    neighbors = neighbor_indices[:, 1 : rank + 1]
    return float(np.mean(context_ids[neighbors] == context_ids[:, None]))


def _rank_exact_support_fraction(
    supports: np.ndarray, neighbor_indices: np.ndarray, rank: int
) -> float:
    neighbors = neighbor_indices[:, 1 : rank + 1]
    return float(np.mean(np.all(supports[neighbors] == supports[:, None, :], axis=2)))


def _condition_means(
    rows: list[dict[str, object]], fields: tuple[str, ...]
) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), int(row["k"]), int(row["sample_count"]))].append(row)
    means: list[dict[str, float | str]] = []
    for (condition, k, n_samples), group in sorted(grouped.items()):
        result: dict[str, float | str] = {
            "condition": condition,
            "k": float(k),
            "sample_count": float(n_samples),
        }
        for field in fields:
            result[field] = float(np.mean([float(row[field]) for row in group]))
        means.append(result)
    return means


def _safe_spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _interpret(
    config: StructuredExperimentConfig, summary_rows: list[dict[str, object]]
) -> str:
    fields = (
        "gride_rank_2",
        "gride_over_k",
        "mean_active_modules",
        "mean_log_amplitude_covariance_pr",
        "mean_feature_source_rank",
        "mean_population_small_scale_rank",
        "mean_active_condition_number",
        "one_nn_same_context_fraction",
    )
    means = _condition_means(summary_rows, fields)
    max_n = max(config.sample_counts)
    displayed = [row for row in means if int(row["sample_count"]) == max_n]
    raw_rho = _safe_spearman(
        [float(row["k"]) for row in displayed],
        [float(row["gride_rank_2"]) for row in displayed],
    )
    effective_rho = _safe_spearman(
        [float(row["mean_log_amplitude_covariance_pr"]) for row in displayed],
        [float(row["gride_rank_2"]) for row in displayed],
    )
    source_rho = _safe_spearman(
        [float(row["mean_feature_source_rank"]) for row in displayed],
        [float(row["gride_rank_2"]) for row in displayed],
    )
    within_ten = [abs(float(row["gride_over_k"]) - 1.0) <= 0.10 for row in displayed]

    lines = [
        f"# Part 2 structured-feature report: {config.suite}",
        "",
        "This report is descriptive evidence for a controlled toy generator. The full GRIDE rank profile, paired interventions, and known latent diagnostics take priority over any single correlation.",
        "",
        f"- Rank-2 cells within 10% of raw k at N={max_n:,}: {sum(within_ten)}/{len(within_ten)}.",
        f"- Spearman(raw k, rank-2 GRIDE ID): {raw_rho:.3f}.",
        f"- Spearman(log-amplitude covariance participation ratio, rank-2 GRIDE ID): {effective_rho:.3f}.",
        f"- Spearman(feature-source rank, rank-2 GRIDE ID): {source_rho:.3f}.",
        "",
        "These pooled correlations are diagnostics only. Raw-k tracking is supported only inside conditions whose calibration and scale profiles are also stable.",
        "",
        f"## Condition means at N={max_n:,}",
        "",
        "| condition | k | GRIDE ID | ID/k | active modules | amplitude PR | feature-source rank | small-scale rank | active condition number | 1-NN same context |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in displayed:
        lines.append(
            f"| {row['condition']} | {int(row['k'])} | {float(row['gride_rank_2']):.3f} | "
            f"{float(row['gride_over_k']):.3f} | {float(row['mean_active_modules']):.2f} | "
            f"{float(row['mean_log_amplitude_covariance_pr']):.2f} | "
            f"{float(row['mean_feature_source_rank']):.2f} | "
            f"{float(row['mean_population_small_scale_rank']):.2f} | "
            f"{float(row['mean_active_condition_number']):.2f} | "
            f"{float(row['one_nn_same_context_fraction']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "The experiment can show when GRIDE is empirically associated with raw k or with known effective degrees of freedom in this generator. It cannot establish that a learned LLM feature dictionary is canonical, that GRIDE measures computation, or that the same scale is valid for real residual-stream data.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_structured_results(
    output_dir: Path,
    profile_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> Path:
    """Create one overview retaining calibration, scale, and mechanism diagnostics."""

    summary_fields = (
        "gride_rank_2",
        "gride_over_k",
        "mean_active_modules",
        "mean_log_amplitude_covariance_pr",
        "mean_feature_source_rank",
        "mean_active_condition_number",
        "one_nn_same_context_fraction",
    )
    means = _condition_means(summary_rows, summary_fields)
    conditions = list(dict.fromkeys(str(row["condition"]) for row in summary_rows))
    colors = {condition: plt.cm.tab10(index % 10) for index, condition in enumerate(conditions)}
    max_n = max(int(row["sample_count"]) for row in summary_rows)
    max_k = max(int(row["k"]) for row in summary_rows)
    displayed = [row for row in means if int(row["sample_count"]) == max_n]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    calibration_axis = axes[0, 0]
    for condition in conditions:
        cells = sorted(
            (int(row["k"]), float(row["gride_over_k"]))
            for row in displayed
            if row["condition"] == condition
        )
        calibration_axis.plot(
            [cell[0] for cell in cells],
            [cell[1] for cell in cells],
            marker="o",
            color=colors[condition],
            label=condition,
        )
    calibration_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    calibration_axis.axhspan(0.9, 1.1, color="black", alpha=0.06)
    calibration_axis.set_xlabel("active features k")
    calibration_axis.set_ylabel("rank-2 GRIDE ID / k")
    calibration_axis.set_title(f"Raw-k calibration at N={max_n:,}")
    calibration_axis.legend(fontsize=7)

    effective_axis = axes[0, 1]
    for condition in conditions:
        cells = [row for row in displayed if row["condition"] == condition]
        effective_axis.scatter(
            [float(row["mean_log_amplitude_covariance_pr"]) for row in cells],
            [float(row["gride_rank_2"]) for row in cells],
            color=colors[condition],
            label=condition,
            alpha=0.75,
        )
    upper = max(float(row["gride_rank_2"]) for row in displayed) * 1.05
    effective_axis.plot([0, upper], [0, upper], color="black", linestyle="--", linewidth=1)
    effective_axis.set_xlabel("log-amplitude covariance participation ratio")
    effective_axis.set_ylabel("rank-2 GRIDE ID")
    effective_axis.set_title("Effective amplitude variation")

    grouped_profiles: dict[tuple[str, int, int, int], list[float]] = defaultdict(list)
    for row in profile_rows:
        grouped_profiles[
            (
                str(row["condition"]),
                int(row["k"]),
                int(row["sample_count"]),
                int(row["rank"]),
            )
        ].append(float(row["gride_id"]))
    profile_axis = axes[0, 2]
    for condition in conditions:
        cells = sorted(
            (rank, float(np.mean(values)) / max_k)
            for (cell_condition, k, n_samples, rank), values in grouped_profiles.items()
            if cell_condition == condition and k == max_k and n_samples == max_n
        )
        profile_axis.plot(
            [cell[0] for cell in cells],
            [cell[1] for cell in cells],
            marker="o",
            color=colors[condition],
            label=condition,
        )
    profile_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    profile_axis.set_xscale("log", base=2)
    profile_axis.set_xlabel("GRIDE upper-neighbour rank")
    profile_axis.set_ylabel(f"GRIDE ID / k (k={max_k})")
    profile_axis.set_title("Scale dependence")

    module_axis = axes[1, 0]
    for condition in conditions:
        cells = sorted(
            (int(row["k"]), float(row["mean_active_modules"]) / int(row["k"]))
            for row in displayed
            if row["condition"] == condition
        )
        module_axis.plot(
            [cell[0] for cell in cells],
            [cell[1] for cell in cells],
            marker="o",
            color=colors[condition],
            label=condition,
        )
    module_axis.set_xlabel("active features k")
    module_axis.set_ylabel("active modules / k")
    module_axis.set_title("Co-activation structure")

    geometry_axis = axes[1, 1]
    context_axis = axes[1, 2]
    for condition in conditions:
        cells = [row for row in displayed if row["condition"] == condition]
        geometry_axis.scatter(
            [float(row["mean_active_condition_number"]) for row in cells],
            [float(row["gride_over_k"]) for row in cells],
            color=colors[condition],
            alpha=0.75,
        )
        context_axis.scatter(
            [float(row["one_nn_same_context_fraction"]) for row in cells],
            [float(row["gride_over_k"]) for row in cells],
            color=colors[condition],
            alpha=0.75,
        )
    geometry_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    geometry_axis.set_xlabel("mean condition number of active W_S")
    geometry_axis.set_ylabel("rank-2 GRIDE ID / k")
    geometry_axis.set_title("Feature geometry")
    context_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    context_axis.set_xlabel("1-NN same-context fraction")
    context_axis.set_ylabel("rank-2 GRIDE ID / k")
    context_axis.set_title("Neighborhood context structure")

    fig.tight_layout()
    figure_path = output_dir / "overview.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


def run_structured_experiment(
    config: StructuredExperimentConfig,
    output_dir: Path,
    *,
    show_progress: bool = True,
) -> StructuredArtifacts:
    """Run one complete Part 2 suite and save raw profiles and diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    groups = make_feature_groups(config.feature_count, config.module_count)
    conditions = conditions_for_suite(config.suite)
    total = config.repeats * len(conditions) * len(config.active_features) * len(config.sample_counts)
    progress = tqdm(
        total=total,
        desc=f"part2-{config.suite}: complete measurements",
        unit="measurement",
        disable=not show_progress or not sys.stderr.isatty(),
        dynamic_ncols=True,
    )
    completed = 0
    profile_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    try:
        for repeat in range(config.repeats):
            matrix_cache: dict[float, np.ndarray] = {}
            geometry_cache: dict[float, tuple[float, float]] = {}
            for condition in conditions:
                if condition.coherence not in matrix_cache:
                    matrix_rng = np.random.default_rng(
                        # Reset the same underlying independent and module vectors
                        # at every coherence value.  Only the named mixing weight
                        # changes in the geometry intervention.
                        np.random.SeedSequence([config.seed, repeat, 301])
                    )
                    matrix_cache[condition.coherence] = make_modular_feature_matrix(
                        config.representation_dim,
                        config.feature_count,
                        groups,
                        condition.coherence,
                        matrix_rng,
                    )
                    geometry_cache[condition.coherence] = feature_geometry_diagnostics(
                        matrix_cache[condition.coherence], groups
                    )
                matrix = matrix_cache[condition.coherence]
                within_dot, between_dot = geometry_cache[condition.coherence]

                for k in config.active_features:
                    for n_samples in config.sample_counts:
                        samples = sample_structured_representations(
                            matrix,
                            groups,
                            k,
                            n_samples,
                            support_module_bias=condition.support_module_bias,
                            zipf_alpha=condition.zipf_alpha,
                            amplitude_correlation=condition.amplitude_correlation,
                            amplitude_log_scale=config.amplitude_log_scale,
                            noise_scale=condition.noise_scale,
                            seed_parts=(config.seed, repeat, k, 401),
                        )
                        profile = measure_gride(
                            samples.representations,
                            config.gride_range_max,
                            config.gride_n_jobs,
                        )
                        assert profile.neighbor_indices is not None
                        frequency_entropy, frequency_cv = feature_frequency_diagnostics(
                            samples.supports, config.feature_count
                        )
                        full_rank_fraction, mean_condition, p95_condition, min_singular = (
                            _active_geometry_diagnostics(
                                matrix,
                                samples.supports,
                                min(config.geometry_diagnostic_samples, n_samples),
                            )
                        )
                        exact_support, support_jaccard, module_jaccard, same_context = (
                            _neighbor_diagnostics(
                                samples.supports,
                                samples.context_ids,
                                groups,
                                profile.neighbor_indices,
                            )
                        )
                        mean_active_modules = float(np.mean(samples.active_module_counts))
                        mean_amplitude_pr = float(
                            np.mean(samples.log_amplitude_covariance_pr)
                        )
                        mean_feature_source_rank = float(
                            np.mean(samples.amplitude_source_ranks)
                        )
                        mean_population_rank = (
                            float(config.representation_dim)
                            if condition.noise_scale > 0.0
                            else mean_feature_source_rank
                        )

                        for rank, scale, estimate, error in zip(
                            profile.ranks,
                            profile.scales,
                            profile.ids,
                            profile.errors,
                            strict=True,
                        ):
                            profile_rows.append(
                                {
                                    "experiment": "part2-structured",
                                    "suite": config.suite,
                                    "condition": condition.label,
                                    "repeat": repeat,
                                    "representation_dim": config.representation_dim,
                                    "feature_count": config.feature_count,
                                    "module_count": config.module_count,
                                    "k": k,
                                    "sample_count": n_samples,
                                    "support_module_bias": condition.support_module_bias,
                                    "zipf_alpha": condition.zipf_alpha,
                                    "amplitude_correlation": condition.amplitude_correlation,
                                    "coherence": condition.coherence,
                                    "noise_scale": condition.noise_scale,
                                    "rank": int(rank),
                                    "scale": float(scale),
                                    "gride_id": float(estimate),
                                    "gride_error": float(error),
                                    "same_context_neighbor_fraction": _rank_context_fraction(
                                        samples.context_ids, profile.neighbor_indices, int(rank)
                                    ),
                                    "exact_support_neighbor_fraction": _rank_exact_support_fraction(
                                        samples.supports, profile.neighbor_indices, int(rank)
                                    ),
                                }
                            )

                        rank_2_id = float(profile.ids[0])
                        summary_rows.append(
                            {
                                "experiment": "part2-structured",
                                "suite": config.suite,
                                "condition": condition.label,
                                "repeat": repeat,
                                "representation_dim": config.representation_dim,
                                "feature_count": config.feature_count,
                                "module_count": config.module_count,
                                "k": k,
                                "sample_count": n_samples,
                                "support_module_bias": condition.support_module_bias,
                                "zipf_alpha": condition.zipf_alpha,
                                "amplitude_correlation": condition.amplitude_correlation,
                                "coherence": condition.coherence,
                                "noise_scale": condition.noise_scale,
                                "gride_rank_2": rank_2_id,
                                "gride_over_k": rank_2_id / k,
                                "mean_active_modules": mean_active_modules,
                                "mean_log_amplitude_covariance_pr": mean_amplitude_pr,
                                "mean_feature_source_rank": mean_feature_source_rank,
                                "mean_population_small_scale_rank": mean_population_rank,
                                "mean_within_module_dot": within_dot,
                                "mean_between_module_dot": between_dot,
                                "active_full_rank_fraction": full_rank_fraction,
                                "mean_active_condition_number": mean_condition,
                                "p95_active_condition_number": p95_condition,
                                "min_active_singular_value": min_singular,
                                "feature_frequency_normalized_entropy": frequency_entropy,
                                "feature_frequency_cv": frequency_cv,
                                "one_nn_exact_support_fraction": exact_support,
                                "one_nn_support_jaccard": support_jaccard,
                                "one_nn_module_jaccard": module_jaccard,
                                "one_nn_same_context_fraction": same_context,
                                "signal_rms_before_noise": samples.signal_rms,
                            }
                        )
                        completed += 1
                        progress.update(1)
                        if show_progress and not sys.stderr.isatty():
                            print(
                                f"[{completed}/{total}] part2-{config.suite} complete: "
                                f"repeat={repeat + 1}/{config.repeats}; {condition.label}; "
                                f"k={k}; N={n_samples}",
                                flush=True,
                            )
    finally:
        progress.close()

    profile_path = output_dir / "gride_profiles.csv"
    summary_path = output_dir / "summary.csv"
    interpretation_path = output_dir / "interpretation.md"
    _write_csv(profile_path, profile_rows)
    _write_csv(summary_path, summary_rows)
    interpretation_path.write_text(_interpret(config, summary_rows), encoding="utf-8")
    figure_path = plot_structured_results(output_dir, profile_rows, summary_rows)
    return StructuredArtifacts(
        output_dir=output_dir,
        profile_path=profile_path,
        summary_path=summary_path,
        interpretation_path=interpretation_path,
        figure_path=figure_path,
    )

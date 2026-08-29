"""Controlled designs and decision reports for hypotheses 1--3."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .generator import (
    make_support_bank,
    make_feature_matrix,
    mean_off_diagonal_dot_product,
    sample_sparse_representations,
    sample_support_pool_representations,
)
from .metrics import measure_gride, measure_linear_accessibility


@dataclass(frozen=True)
class Condition:
    representation_dim: int
    feature_count: int
    k: int
    rho: float = 0.0
    support_pool_size: int | None = None
    id_sample_count: int | None = None

    @property
    def label(self) -> str:
        base = f"D={self.representation_dim};m={self.feature_count};k={self.k};rho={self.rho:g}"
        if self.support_pool_size is not None and self.id_sample_count is not None:
            return f"{base};B={self.support_pool_size};N={self.id_sample_count}"
        return base


@dataclass(frozen=True)
class ExperimentArtifacts:
    output_dir: Path
    profile_path: Path
    summary_path: Path
    verdict_path: Path


def conditions_for(config: ExperimentConfig) -> list[Condition]:
    """Return the complete intervention grid for one stated hypothesis."""

    if config.experiment == "load-capacity":
        return [
            Condition(config.representation_dim, m, k)
            for m in config.capacity_values
            for k in config.active_features
        ]
    if config.experiment == "load-ratio":
        return [
            Condition(d, config.feature_count, k)
            for d in config.representation_dims
            for k in config.active_features
        ]
    if config.experiment == "support-pool":
        return [
            Condition(
                config.representation_dim,
                config.feature_count,
                k,
                support_pool_size=pool_size,
                id_sample_count=n_samples,
            )
            for k in config.active_features
            for pool_size in config.support_pool_sizes
            for n_samples in config.id_sample_values
        ]
    return [
        Condition(config.representation_dim, config.feature_count, config.fixed_k, rho)
        for rho in config.correlations
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _matrix_for_condition(
    config: ExperimentConfig,
    repeat: int,
    condition_index: int,
    condition: Condition,
    largest_capacity_matrix: np.ndarray | None,
) -> np.ndarray:
    """Keep every non-intervened quantity fixed wherever that is possible."""

    if config.experiment == "load-capacity":
        assert largest_capacity_matrix is not None
        # A capacity condition is literally a prefix of the same dictionary.
        return largest_capacity_matrix[:, : condition.feature_count]
    if config.experiment == "load-ratio":
        # All k values at a particular D receive an identical dictionary.
        matrix_rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, repeat, condition.representation_dim])
        )
        return make_feature_matrix(
            condition.representation_dim, condition.feature_count, 0.0, matrix_rng
        )
    if config.experiment == "support-pool":
        matrix_rng = np.random.default_rng(np.random.SeedSequence([config.seed, repeat, 73]))
        return make_feature_matrix(
            condition.representation_dim, condition.feature_count, 0.0, matrix_rng
        )
    matrix_rng = np.random.default_rng(
        np.random.SeedSequence([config.seed, repeat, condition_index])
    )
    return make_feature_matrix(
        condition.representation_dim,
        condition.feature_count,
        condition.rho,
        matrix_rng,
    )


def run_experiment(
    config: ExperimentConfig, output_dir: Path, *, show_progress: bool = True
) -> ExperimentArtifacts:
    """Run a complete hypothesis-specific sweep and save all raw measurements."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    profile_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    conditions = conditions_for(config)
    progress = tqdm(
        total=config.repeats * len(conditions),
        desc=f"{config.experiment}: complete measurements",
        unit="measurement",
        dynamic_ncols=True,
        # In a Slurm log, carriage-return bars are hard to inspect with tail.
        # Emit one durable completion line per measurement there instead.
        disable=not show_progress or not sys.stderr.isatty(),
    )
    completed_measurements = 0

    try:
        for repeat in range(config.repeats):
            largest_capacity_matrix = None
            support_banks: dict[int, np.ndarray] = {}
            if config.experiment == "load-capacity":
                dictionary_rng = np.random.default_rng(np.random.SeedSequence([config.seed, repeat]))
                largest_capacity_matrix = make_feature_matrix(
                    config.representation_dim, max(config.capacity_values), 0.0, dictionary_rng
                )
            elif config.experiment == "support-pool":
                for k in config.active_features:
                    bank_rng = np.random.default_rng(
                        np.random.SeedSequence([config.seed, repeat, k, 79])
                    )
                    support_banks[k] = make_support_bank(
                        config.feature_count, k, max(config.support_pool_sizes), bank_rng
                    )

            for condition_index, condition in enumerate(conditions):
                progress.set_postfix_str(f"repeat={repeat + 1}/{config.repeats}; {condition.label}")
                matrix = _matrix_for_condition(
                    config, repeat, condition_index, condition, largest_capacity_matrix
                )
                access = None
                if config.experiment == "support-pool":
                    assert condition.support_pool_size is not None
                    assert condition.id_sample_count is not None
                    supports = support_banks[condition.k][: condition.support_pool_size]
                    support_rngs = [
                        np.random.default_rng(
                            np.random.SeedSequence(
                                [config.seed, repeat, condition.k, support_index, 83]
                            )
                        )
                        for support_index in range(condition.support_pool_size)
                    ]
                    id_h, support_ids = sample_support_pool_representations(
                        matrix, supports, condition.id_sample_count, support_rngs
                    )
                    profile = measure_gride(
                        id_h,
                        config.gride_range_max,
                        config.gride_n_jobs,
                        support_ids=support_ids,
                    )
                else:
                    sample_rng = np.random.default_rng(
                        np.random.SeedSequence([config.seed, repeat, condition_index, 1])
                    )
                    id_h, _ = sample_sparse_representations(
                        matrix, condition.k, config.n_id_samples, sample_rng
                    )
                    train_h, train_labels = sample_sparse_representations(
                        matrix, condition.k, config.n_train, sample_rng
                    )
                    test_h, test_labels = sample_sparse_representations(
                        matrix, condition.k, config.n_test, sample_rng
                    )
                    profile = measure_gride(id_h, config.gride_range_max, config.gride_n_jobs)
                    access = measure_linear_accessibility(
                        train_h, train_labels, test_h, test_labels
                    )
                for rank, scale, estimate, error in zip(
                    profile.ranks, profile.scales, profile.ids, profile.errors, strict=True
                ):
                    row: dict[str, object] = {
                        "experiment": config.experiment,
                        "condition": condition.label,
                        "repeat": repeat,
                        "representation_dim": condition.representation_dim,
                        "feature_count": condition.feature_count,
                        "k": condition.k,
                        "rho": condition.rho,
                        "support_pool_size": condition.support_pool_size or 0,
                        "id_sample_count": condition.id_sample_count or config.n_id_samples,
                        "samples_per_support": (
                            condition.id_sample_count // condition.support_pool_size
                            if condition.id_sample_count is not None
                            and condition.support_pool_size is not None
                            else 0
                        ),
                        "rank": int(rank),
                        "scale": float(scale),
                        "gride_id": float(estimate),
                        "gride_error": float(error),
                    }
                    if profile.mean_same_support_fraction is not None:
                        rank_index = int(np.flatnonzero(profile.ranks == rank)[0])
                        assert profile.all_same_support_fraction is not None
                        row["mean_same_support_neighbor_fraction"] = float(
                            profile.mean_same_support_fraction[rank_index]
                        )
                        row["all_same_support_fraction"] = float(
                            profile.all_same_support_fraction[rank_index]
                        )
                    profile_rows.append(row)
                summary_row: dict[str, object] = {
                        "experiment": config.experiment,
                        "condition": condition.label,
                        "repeat": repeat,
                        "representation_dim": condition.representation_dim,
                        "feature_count": condition.feature_count,
                        "k": condition.k,
                        "rho": condition.rho,
                        "support_pool_size": condition.support_pool_size or 0,
                        "id_sample_count": condition.id_sample_count or config.n_id_samples,
                        "samples_per_support": (
                            condition.id_sample_count // condition.support_pool_size
                            if condition.id_sample_count is not None
                            and condition.support_pool_size is not None
                            else 0
                        ),
                        "nominal_load_ratio_k_over_d": condition.k / condition.representation_dim,
                        "mean_feature_dot_product": mean_off_diagonal_dot_product(matrix),
                        "local_gride_id_rank_2": float(profile.ids[0]),
                        "estimated_load_ratio_id_over_d": float(profile.ids[0] / condition.representation_dim),
                    }
                if access is not None:
                    summary_row.update(
                        {
                            "mean_feature_auroc": access.mean_auroc,
                            "std_feature_auroc": access.std_auroc,
                            "mean_feature_balanced_accuracy": access.mean_balanced_accuracy,
                            "mean_normalized_signed_margin": access.mean_normalized_signed_margin,
                            "std_normalized_signed_margin": access.std_normalized_signed_margin,
                        }
                    )
                else:
                    assert profile.mean_same_support_fraction is not None
                    assert profile.all_same_support_fraction is not None
                    summary_row.update(
                        {
                            "rank_2_relative_error_to_k": abs(float(profile.ids[0]) - condition.k)
                            / condition.k,
                            "rank_2_mean_same_support_neighbor_fraction": float(
                                profile.mean_same_support_fraction[0]
                            ),
                            "rank_2_all_same_support_fraction": float(
                                profile.all_same_support_fraction[0]
                            ),
                        }
                    )
                summary_rows.append(summary_row)
                completed_measurements += 1
                progress.update(1)
                if show_progress and not sys.stderr.isatty():
                    print(
                        f"[{completed_measurements}/{config.repeats * len(conditions)}] "
                        f"{config.experiment} complete: "
                        f"repeat={repeat + 1}/{config.repeats}; {condition.label}",
                        flush=True,
                    )
    finally:
        progress.close()

    profile_path = output_dir / "gride_profiles.csv"
    summary_path = output_dir / "summary.csv"
    verdict_path = output_dir / "interpretation.md"
    _write_csv(profile_path, profile_rows)
    _write_csv(summary_path, summary_rows)
    verdict_path.write_text(_interpret(config, summary_rows, profile_rows), encoding="utf-8")
    return ExperimentArtifacts(output_dir, profile_path, summary_path, verdict_path)


def _condition_means(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    return {
        condition: {
            key: float(np.mean([float(row[key]) for row in group]))
            for key in group[0]
            if key not in {"experiment", "condition"}
        }
        for condition, group in grouped.items()
    }


def _status(passed: bool) -> str:
    return "PASS" if passed else "NOT ESTABLISHED"


def _format(value: float) -> str:
    return "undefined" if np.isnan(value) else f"{value:.3f}"


def _sem(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("inf")


def _interpret_hypothesis_1(
    config: ExperimentConfig, means: dict[str, dict[str, float]]
) -> list[str]:
    by_k: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in means.values():
        by_k[int(row["k"])].append(row)
    mean_ids_by_k = {k: float(np.mean([row["local_gride_id_rank_2"] for row in group])) for k, group in by_k.items()}
    k_values = np.array(sorted(mean_ids_by_k), dtype=float)
    ids = np.array([mean_ids_by_k[int(k)] for k in k_values])
    k_id_rho = float(spearmanr(k_values, ids).statistic)
    recovery_error = np.abs(ids - k_values) / k_values
    capacity_spreads = {
        k: (max(row["local_gride_id_rank_2"] for row in group) - min(row["local_gride_id_rank_2"] for row in group))
        / np.mean([row["local_gride_id_rank_2"] for row in group])
        for k, group in by_k.items()
    }
    recovery_pass = bool(np.all(recovery_error <= 0.25))
    capacity_pass = bool(all(spread <= config.id_match_relative_tolerance for spread in capacity_spreads.values()))
    monotonic_pass = bool(k_id_rho >= 0.9)
    lines = [
        "## Hypothesis 1 — ID tracks load, not capacity",
        "",
        f"- Local-ID versus k monotonicity: Spearman rho={_format(k_id_rho)}; `{_status(monotonic_pass)}` (criterion >= 0.90).",
        f"- Recovery of k: `{_status(recovery_pass)}` (every capacity-averaged local ID must be within 25% of k).",
        f"- Capacity invariance: `{_status(capacity_pass)}` (within-k max spread across m must be <= {config.id_match_relative_tolerance:.0%}).",
        "",
        "| k | mean local ID | relative error to k | max m-induced ID spread |",
        "| ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {int(k)} | {mean_ids_by_k[int(k)]:.3f} | {recovery_error[i]:.1%} | {capacity_spreads[int(k)]:.1%} |"
        for i, k in enumerate(k_values)
    )
    conclusion = recovery_pass and capacity_pass and monotonic_pass
    lines.extend(
        [
            "",
            f"**Conclusion: {_status(conclusion)}.** "
            "Only a PASS supports the stated local-scale claim for this generator; otherwise the saved profiles show which prerequisite failed.",
        ]
    )
    return lines


def _interpret_support_pool(
    config: ExperimentConfig,
    summary_rows: list[dict[str, object]],
    profile_rows: list[dict[str, object]],
) -> list[str]:
    """Test conditional recovery and diagnose when pooled neighborhoods mix supports."""

    summary_means = list(_condition_means(summary_rows).values())
    rank_2_rows = sorted(
        summary_means,
        key=lambda row: (int(row["k"]), int(row["support_pool_size"]), int(row["id_sample_count"])),
    )
    fixed_support = [row for row in rank_2_rows if int(row["support_pool_size"]) == 1]
    fixed_errors = np.array([row["rank_2_relative_error_to_k"] for row in fixed_support])
    fixed_support_pass = bool(np.all(fixed_errors <= 0.25))

    purity_threshold = 0.90
    support_pure = [
        row
        for row in rank_2_rows
        if row["rank_2_all_same_support_fraction"] >= purity_threshold
    ]
    pooled_support_pure = [row for row in support_pure if int(row["support_pool_size"]) > 1]
    support_pure_pass = bool(pooled_support_pure) and all(
        row["rank_2_relative_error_to_k"] <= 0.25 for row in support_pure
    )

    profile_groups: dict[tuple[int, int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in profile_rows:
        key = (
            int(row["k"]),
            int(row["support_pool_size"]),
            int(row["id_sample_count"]),
            int(row["rank"]),
        )
        profile_groups[key].append(row)
    profile_means = []
    for (k, pool_size, n_samples, rank), group in profile_groups.items():
        profile_means.append(
            {
                "k": k,
                "support_pool_size": pool_size,
                "id_sample_count": n_samples,
                "rank": rank,
                "gride_id": float(np.mean([float(row["gride_id"]) for row in group])),
                "all_same_support_fraction": float(
                    np.mean([float(row["all_same_support_fraction"]) for row in group])
                ),
            }
        )
    purities = np.array([row["all_same_support_fraction"] for row in profile_means])
    relative_errors = np.array(
        [abs(row["gride_id"] - row["k"]) / row["k"] for row in profile_means]
    )
    purity_error_rho = (
        float("nan")
        if np.ptp(purities) == 0.0 or np.ptp(relative_errors) == 0.0
        else float(spearmanr(purities, relative_errors).statistic)
    )

    lines = [
        "## Hypothesis 1b — support-controlled local ID",
        "",
        "This experiment crosses support-pool size B and sample count N, while retaining every GRIDE rank r. "
        "Supports are nested across B, samples within each support are nested across N, and the dictionary is shared within a repeat.",
        "",
        f"- Fixed-support rank-2 recovery: `{_status(fixed_support_pass)}` "
        "(every B=1 mean ID must be within 25% of k).",
        f"- Pooled but support-pure rank-2 recovery: `{_status(support_pure_pass)}` "
        f"({len(pooled_support_pure)} B>1 cells had at least {purity_threshold:.0%} fully support-pure rank-2 neighborhoods).",
        f"- Spearman(neighborhood purity, absolute relative ID error), using every B/N/r cell: {_format(purity_error_rho)}. "
        "This is a mechanism diagnostic, not a pass criterion.",
        "",
        "### Fixed-support rank-2 estimates",
        "",
        "| k | N | mean rank-2 ID | relative error |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in fixed_support:
        lines.append(
            f"| {int(row['k'])} | {int(row['id_sample_count'])} | "
            f"{row['local_gride_id_rank_2']:.3f} | {row['rank_2_relative_error_to_k']:.1%} |"
        )

    lines.extend(
        [
            "",
            "### Largest support pool with support-pure rank-2 neighborhoods",
            "",
            "| k | N | largest support-pure B | ID / k | fully pure neighborhoods |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for k in sorted({int(row["k"]) for row in rank_2_rows}):
        for n_samples in sorted({int(row["id_sample_count"]) for row in rank_2_rows}):
            candidates = [
                row
                for row in rank_2_rows
                if int(row["k"]) == k
                and int(row["id_sample_count"]) == n_samples
                and row["rank_2_all_same_support_fraction"] >= purity_threshold
            ]
            best = max(candidates, key=lambda row: int(row["support_pool_size"]))
            lines.append(
                f"| {k} | {n_samples} | {int(best['support_pool_size'])} | "
                f"{best['local_gride_id_rank_2'] / k:.3f} | "
                f"{best['rank_2_all_same_support_fraction']:.1%} |"
            )

    conclusion = fixed_support_pass and support_pure_pass
    lines.extend(
        [
            "",
            f"**Conclusion: {_status(conclusion)}.** A PASS supports only the conditional finite-scale claim: "
            "GRIDE recovers k when its local neighborhoods remain on one exact-k support. "
            "It does not imply that pooled activation ID universally equals k.",
        ]
    )
    return lines


def _interpret_hypothesis_2(
    config: ExperimentConfig, means: dict[str, dict[str, float]]
) -> list[str]:
    rows = list(means.values())
    load = np.array([row["estimated_load_ratio_id_over_d"] for row in rows])
    aurocs = np.array([row["mean_feature_auroc"] for row in rows])
    margins = np.array([row["mean_normalized_signed_margin"] for row in rows])
    id_auroc_rho = float(spearmanr(load, aurocs).statistic)
    id_margin_rho = float(spearmanr(load, margins).statistic)
    # A direct collapse diagnostic: multiple (k,D) conditions with identical k/D
    # should have similar AUROC.  Singletons do not provide a collapse check.
    by_nominal_load: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_nominal_load[round(row["nominal_load_ratio_k_over_d"], 10)].append(row)
    repeated_ratios = [group for group in by_nominal_load.values() if len(group) > 1]
    collapse_spread = max(
        (max(row["mean_feature_auroc"] for row in group) - min(row["mean_feature_auroc"] for row in group)
        for group in repeated_ratios),
        default=float("inf"),
    )
    relationship_pass = id_auroc_rho <= -0.8 and id_margin_rho <= -0.8
    collapse_pass = collapse_spread <= config.auroc_effect_threshold
    lines = [
        "## Hypothesis 2 — load ratio predicts accessibility in isotropic dictionaries",
        "",
        f"- Spearman(local ID / D, AUROC) = {_format(id_auroc_rho)}; `{_status(id_auroc_rho <= -0.8)}` (criterion <= -0.80).",
        f"- Spearman(local ID / D, normalized signed margin) = {_format(id_margin_rho)}; `{_status(id_margin_rho <= -0.8)}` (criterion <= -0.80).",
        f"- Same-k/D AUROC collapse: maximum spread={_format(collapse_spread)}; `{_status(collapse_pass)}` (criterion <= {config.auroc_effect_threshold:.3f}).",
        "",
        "| D | k | local ID / D | mean AUROC | mean normalized margin |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda value: (value["representation_dim"], value["k"])):
        lines.append(
            f"| {int(row['representation_dim'])} | {int(row['k'])} | "
            f"{row['estimated_load_ratio_id_over_d']:.3f} | {row['mean_feature_auroc']:.4f} | "
            f"{row['mean_normalized_signed_margin']:.4f} |"
        )
    conclusion = relationship_pass and collapse_pass
    lines.extend(
        [
            "",
            f"**Conclusion: {_status(conclusion)}.** This conclusion is intentionally restricted to the isotropic random-dictionary regime.",
        ]
    )
    return lines


def _interpret_hypothesis_3(
    config: ExperimentConfig,
    summary_rows: list[dict[str, object]],
    profile_rows: list[dict[str, object]],
) -> list[str]:
    by_rho: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_rho[float(row["rho"])].append(row)
    baseline = by_rho[0.0]
    profile_by_rho_rank: dict[tuple[float, int], list[float]] = defaultdict(list)
    for row in profile_rows:
        profile_by_rho_rank[(float(row["rho"]), int(row["rank"]))].append(float(row["gride_id"]))
    baseline_profile = {
        rank: float(np.mean(values))
        for (rho, rank), values in profile_by_rho_rank.items()
        if rho == 0.0
    }
    lines = [
        "## Hypothesis 3 — ID is insufficient under a geometry intervention",
        "",
        "A valid counterexample requires both a matched GRIDE profile and a replicated AUROC decrease. "
        "A rho condition passes only if every saved-rank ID differs from rho=0 by at most "
        f"{config.id_match_relative_tolerance:.0%}, and its paired AUROC decrease has a 95% lower bound of at least "
        f"{config.auroc_effect_threshold:.3f}.",
        "",
        "| rho | mean feature dot product | max profile ID delta | paired AUROC delta | 95% lower-bound decrease | matched counterexample |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    any_counterexample = False
    baseline_by_repeat = {int(row["repeat"]): float(row["mean_feature_auroc"]) for row in baseline}
    for rho in sorted(by_rho):
        group = by_rho[rho]
        profile_deltas = [
            abs(float(np.mean(values)) - baseline_profile[rank]) / max(abs(baseline_profile[rank]), 1e-12)
            for (candidate_rho, rank), values in profile_by_rho_rank.items()
            if candidate_rho == rho and rank in baseline_profile
        ]
        max_profile_delta = max(profile_deltas, default=float("inf"))
        deltas = np.array(
            [float(row["mean_feature_auroc"]) - baseline_by_repeat[int(row["repeat"])] for row in group]
        )
        mean_delta = float(deltas.mean())
        lower_bound_decrease = -mean_delta - 1.96 * _sem(deltas)
        matched = rho != 0.0 and max_profile_delta <= config.id_match_relative_tolerance
        effect = lower_bound_decrease >= config.auroc_effect_threshold
        counterexample = matched and effect
        any_counterexample = any_counterexample or counterexample
        coherence = float(np.mean([float(row["mean_feature_dot_product"]) for row in group]))
        lines.append(
            f"| {rho:g} | {coherence:.3f} | {max_profile_delta:.1%} | {mean_delta:+.4f} | "
            f"{lower_bound_decrease:.4f} | {'YES' if counterexample else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"**Conclusion: {_status(any_counterexample)}.** "
            "A PASS establishes a counterexample to ID sufficiency within this controlled generator; a failure is inconclusive rather than support for sufficiency.",
        ]
    )
    return lines


def _interpret(
    config: ExperimentConfig,
    summary_rows: list[dict[str, object]],
    profile_rows: list[dict[str, object]],
) -> str:
    """Produce prespecified, bounded decision criteria rather than prose-only trends."""

    lines = [
        "# Hypothesis report",
        "",
        "This automated report is valid only for the saved seed, generator, sample sizes, and GRIDE ranks. "
        "A PASS is evidence for the stated toy-model claim, not a claim about learned or language-model representations.",
        "",
    ]
    means = _condition_means(summary_rows)
    if config.experiment == "load-capacity":
        lines.extend(_interpret_hypothesis_1(config, means))
    elif config.experiment == "support-pool":
        lines.extend(_interpret_support_pool(config, summary_rows, profile_rows))
    elif config.experiment == "load-ratio":
        lines.extend(_interpret_hypothesis_2(config, means))
    else:
        lines.extend(_interpret_hypothesis_3(config, summary_rows, profile_rows))
    return "\n".join(lines) + "\n"

#!/usr/bin/env python3
"""Paired analysis and publication-style plots for the Part 2 GRIDE suites."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, t


SUITES = ("support", "amplitude", "geometry", "frequency", "noise", "combined")
BASELINES = {
    "support": "support_bias=0",
    "amplitude": "amplitude_rho=0",
    "geometry": "coherence=0",
    "frequency": "zipf_alpha=0",
    "noise": "noise=0",
    "combined": "baseline",
}
CONDITION_ORDER = {
    "support": ("support_bias=0", "support_bias=1", "support_bias=2", "support_bias=4"),
    "amplitude": (
        "amplitude_rho=0",
        "amplitude_rho=0.5",
        "amplitude_rho=0.9",
        "amplitude_rho=1",
    ),
    "geometry": (
        "coherence=0",
        "coherence=0.25",
        "coherence=0.5",
        "coherence=0.75",
        "coherence=0.9",
    ),
    "frequency": (
        "zipf_alpha=0",
        "zipf_alpha=0.5",
        "zipf_alpha=1",
        "zipf_alpha=1.5",
    ),
    "noise": ("noise=0", "noise=0.01", "noise=0.05", "noise=0.2"),
    "combined": (
        "baseline",
        "modular_support",
        "zipf_frequencies",
        "shared_amplitudes",
        "coherent_geometry",
        "llm_like",
        "llm_like_exact_shared",
    ),
}
PANEL_TITLES = {
    "support": "A  Modular support",
    "amplitude": "B  Shared amplitudes",
    "geometry": "C  Coherent dictionary",
    "frequency": "D  Zipf feature frequencies",
    "noise": "E  Dense isotropic noise",
    "combined": "F  Named model variants",
}
PARAMETER_LABELS = {
    "support": r"module preference $\beta$",
    "amplitude": r"amplitude correlation $\rho$",
    "geometry": r"within-module coherence $\gamma$",
    "frequency": r"Zipf exponent $\alpha$",
    "noise": r"noise RMS / signal RMS $\sigma$",
}
COMBINED_SHORT = ("base", "module", "Zipf", "amp", "geom", "all", r"all+$\rho$=1")
COLORS = {2: "#3B6FB6", 4: "#E07A2D", 8: "#2F8F61"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_ci(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, mean, mean
    half_width = float(t.ppf(0.975, len(array) - 1) * np.std(array, ddof=1) / np.sqrt(len(array)))
    return mean, mean - half_width, mean + half_width


def _profile_index(rows: list[dict[str, str]]) -> dict[tuple[str, int, int, int, int], float]:
    indexed: dict[tuple[str, int, int, int, int], float] = {}
    for row in rows:
        key = (
            row["condition"],
            int(row["k"]),
            int(row["sample_count"]),
            int(row["rank"]),
            int(row["repeat"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate profile key: {key}")
        indexed[key] = float(row["gride_id"])
    return indexed


def _paired_row(
    suite: str,
    reference: str,
    target: str,
    k: int,
    sample_count: int,
    rank: int,
    indexed: dict[tuple[str, int, int, int, int], float],
) -> dict[str, object]:
    reference_by_repeat = {
        repeat: value
        for (condition, row_k, row_n, row_rank, repeat), value in indexed.items()
        if condition == reference and row_k == k and row_n == sample_count and row_rank == rank
    }
    target_by_repeat = {
        repeat: value
        for (condition, row_k, row_n, row_rank, repeat), value in indexed.items()
        if condition == target and row_k == k and row_n == sample_count and row_rank == rank
    }
    if reference_by_repeat.keys() != target_by_repeat.keys() or not reference_by_repeat:
        raise ValueError(
            f"unpaired contrast for {suite}: {reference} -> {target}, k={k}, "
            f"N={sample_count}, rank={rank}"
        )
    repeats = sorted(reference_by_repeat)
    reference_values = np.asarray([reference_by_repeat[r] for r in repeats])
    target_values = np.asarray([target_by_repeat[r] for r in repeats])
    deltas = target_values - reference_values
    reference_mean, _, _ = _mean_ci(reference_values)
    target_mean, _, _ = _mean_ci(target_values)
    delta_mean, delta_low, delta_high = _mean_ci(deltas)
    return {
        "suite": suite,
        "reference_condition": reference,
        "target_condition": target,
        "k": k,
        "sample_count": sample_count,
        "rank": rank,
        "repeat_count": len(repeats),
        "reference_mean": reference_mean,
        "target_mean": target_mean,
        "mean_delta": delta_mean,
        "delta_ci_low": delta_low,
        "delta_ci_high": delta_high,
        "mean_ratio": target_mean / reference_mean,
        "mean_delta_over_k": delta_mean / k,
        "negative_repeat_fraction": float(np.mean(deltas < 0.0)),
    }


def paired_effects(
    profiles_by_suite: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    """Compute repeat-paired contrasts at every fixed k, N, and GRIDE rank."""

    output: list[dict[str, object]] = []
    for suite in SUITES:
        rows = profiles_by_suite[suite]
        indexed = _profile_index(rows)
        ks = sorted({int(row["k"]) for row in rows})
        sample_counts = sorted({int(row["sample_count"]) for row in rows})
        ranks = sorted({int(row["rank"]) for row in rows})
        reference = BASELINES[suite]
        targets = [condition for condition in CONDITION_ORDER[suite] if condition != reference]
        for target in targets:
            for k in ks:
                for sample_count in sample_counts:
                    for rank in ranks:
                        output.append(
                            _paired_row(
                                suite,
                                reference,
                                target,
                                k,
                                sample_count,
                                rank,
                                indexed,
                            )
                        )
        if suite == "combined":
            for k in ks:
                for sample_count in sample_counts:
                    for rank in ranks:
                        output.append(
                            _paired_row(
                                suite,
                                "llm_like",
                                "llm_like_exact_shared",
                                k,
                                sample_count,
                                rank,
                                indexed,
                            )
                        )
    return output


def sample_size_effects(
    profiles_by_suite: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    """Compare the two nested sample counts within the same random realization."""

    output: list[dict[str, object]] = []
    for suite in SUITES:
        rows = profiles_by_suite[suite]
        indexed = _profile_index(rows)
        sample_counts = sorted({int(row["sample_count"]) for row in rows})
        if len(sample_counts) != 2:
            raise ValueError(f"expected exactly two sample counts in {suite}")
        small_n, large_n = sample_counts
        for condition in CONDITION_ORDER[suite]:
            for k in sorted({int(row["k"]) for row in rows}):
                for rank in sorted({int(row["rank"]) for row in rows}):
                    small = {
                        repeat: value
                        for (row_condition, row_k, row_n, row_rank, repeat), value in indexed.items()
                        if row_condition == condition
                        and row_k == k
                        and row_n == small_n
                        and row_rank == rank
                    }
                    large = {
                        repeat: value
                        for (row_condition, row_k, row_n, row_rank, repeat), value in indexed.items()
                        if row_condition == condition
                        and row_k == k
                        and row_n == large_n
                        and row_rank == rank
                    }
                    if small.keys() != large.keys() or not small:
                        raise ValueError(f"unpaired sample-size comparison in {suite}, {condition}")
                    repeats = sorted(small)
                    small_values = np.asarray([small[repeat] for repeat in repeats])
                    large_values = np.asarray([large[repeat] for repeat in repeats])
                    deltas = large_values - small_values
                    delta_mean, delta_low, delta_high = _mean_ci(deltas)
                    output.append(
                        {
                            "suite": suite,
                            "condition": condition,
                            "k": k,
                            "rank": rank,
                            "small_sample_count": small_n,
                            "large_sample_count": large_n,
                            "repeat_count": len(repeats),
                            "small_sample_mean": float(np.mean(small_values)),
                            "large_sample_mean": float(np.mean(large_values)),
                            "mean_delta": delta_mean,
                            "delta_ci_low": delta_low,
                            "delta_ci_high": delta_high,
                            "same_sign_repeat_fraction": float(
                                max(np.mean(deltas < 0.0), np.mean(deltas > 0.0))
                            ),
                        }
                    )
    return output


def tracking_diagnostics(
    profiles_by_suite: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    """Summarize monotone k-ordering separately from numerical calibration."""

    output: list[dict[str, object]] = []
    for suite in SUITES:
        rows = profiles_by_suite[suite]
        for condition in CONDITION_ORDER[suite]:
            for sample_count in sorted({int(row["sample_count"]) for row in rows}):
                for rank in sorted({int(row["rank"]) for row in rows}):
                    selected = [
                        row
                        for row in rows
                        if row["condition"] == condition
                        and int(row["sample_count"]) == sample_count
                        and int(row["rank"]) == rank
                    ]
                    repeats = sorted({int(row["repeat"]) for row in selected})
                    monotone: list[bool] = []
                    correlations: list[float] = []
                    relative_errors: list[float] = []
                    within_ten: list[bool] = []
                    for repeat in repeats:
                        repeat_rows = sorted(
                            (row for row in selected if int(row["repeat"]) == repeat),
                            key=lambda row: int(row["k"]),
                        )
                        ks = np.asarray([int(row["k"]) for row in repeat_rows], dtype=float)
                        ids = np.asarray([float(row["gride_id"]) for row in repeat_rows])
                        monotone.append(bool(np.all(np.diff(ids) > 0.0)))
                        correlations.append(float(spearmanr(ks, ids).statistic))
                        relative_errors.extend(np.abs(ids / ks - 1.0))
                        within_ten.extend(np.abs(ids / ks - 1.0) <= 0.10)
                    output.append(
                        {
                            "suite": suite,
                            "condition": condition,
                            "sample_count": sample_count,
                            "rank": rank,
                            "repeat_count": len(repeats),
                            "strictly_increasing_repeat_fraction": float(np.mean(monotone)),
                            "mean_within_repeat_spearman": float(np.mean(correlations)),
                            "mean_absolute_relative_error": float(np.mean(relative_errors)),
                            "within_ten_percent_cell_fraction": float(np.mean(within_ten)),
                        }
                    )
    return output


def cell_means(profiles_by_suite: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for suite in SUITES:
        grouped: dict[tuple[str, int, int, int], list[float]] = defaultdict(list)
        for row in profiles_by_suite[suite]:
            grouped[
                (
                    row["condition"],
                    int(row["k"]),
                    int(row["sample_count"]),
                    int(row["rank"]),
                )
            ].append(float(row["gride_id"]))
        for condition in CONDITION_ORDER[suite]:
            for (row_condition, k, sample_count, rank), values in sorted(grouped.items()):
                if row_condition != condition:
                    continue
                mean, low, high = _mean_ci(values)
                output.append(
                    {
                        "suite": suite,
                        "condition": condition,
                        "k": k,
                        "sample_count": sample_count,
                        "rank": rank,
                        "repeat_count": len(values),
                        "gride_id_mean": mean,
                        "gride_id_ci_low": low,
                        "gride_id_ci_high": high,
                        "gride_over_k_mean": mean / k,
                        "gride_over_k_ci_low": low / k,
                        "gride_over_k_ci_high": high / k,
                    }
                )
    return output


def combined_diagnostics(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    fields = (
        "gride_rank_2",
        "mean_active_modules",
        "mean_log_amplitude_covariance_pr",
        "mean_feature_source_rank",
        "mean_population_small_scale_rank",
        "mean_active_condition_number",
        "one_nn_same_context_fraction",
        "one_nn_exact_support_fraction",
    )
    max_n = max(int(row["sample_count"]) for row in summary_rows)
    output: list[dict[str, object]] = []
    for condition in CONDITION_ORDER["combined"]:
        for k in sorted({int(row["k"]) for row in summary_rows}):
            selected = [
                row
                for row in summary_rows
                if row["condition"] == condition
                and int(row["k"]) == k
                and int(row["sample_count"]) == max_n
            ]
            if not selected:
                continue
            result: dict[str, object] = {
                "condition": condition,
                "k": k,
                "sample_count": max_n,
                "repeat_count": len(selected),
            }
            for field in fields:
                mean, low, high = _mean_ci([float(row[field]) for row in selected])
                result[f"{field}_mean"] = mean
                result[f"{field}_ci_low"] = low
                result[f"{field}_ci_high"] = high
            output.append(result)
    return output


def _lookup_cell(
    cells: list[dict[str, object]], suite: str, condition: str, k: int, n: int, rank: int
) -> dict[str, object]:
    matches = [
        row
        for row in cells
        if row["suite"] == suite
        and row["condition"] == condition
        and row["k"] == k
        and row["sample_count"] == n
        and row["rank"] == rank
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one cell, found {len(matches)}")
    return matches[0]


def plot_single_factor_effects(cells: list[dict[str, object]], path: Path) -> None:
    max_n = max(int(row["sample_count"]) for row in cells)
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.3), constrained_layout=True)
    for axis, suite in zip(axes.flat, SUITES, strict=True):
        conditions = CONDITION_ORDER[suite]
        x = np.arange(len(conditions), dtype=float)
        for k in (2, 4, 8):
            selected = [_lookup_cell(cells, suite, condition, k, max_n, 2) for condition in conditions]
            means = np.asarray([float(row["gride_over_k_mean"]) for row in selected])
            lows = np.asarray([float(row["gride_over_k_ci_low"]) for row in selected])
            highs = np.asarray([float(row["gride_over_k_ci_high"]) for row in selected])
            axis.errorbar(
                x,
                means,
                yerr=np.vstack((means - lows, highs - means)),
                color=COLORS[k],
                marker="o",
                linewidth=1.8,
                capsize=2.5,
                label=f"k={k}",
            )
        axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        axis.set_title(PANEL_TITLES[suite], loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("rank-2 GRIDE ID / k")
        axis.grid(alpha=0.22)
        if suite == "combined":
            axis.set_xticks(x, COMBINED_SHORT, rotation=30, ha="right")
        else:
            values = [float(condition.split("=")[1]) for condition in conditions]
            axis.set_xticks(x, [f"{value:g}" for value in values])
            axis.set_xlabel(PARAMETER_LABELS[suite])
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=9)
    figure.suptitle(
        f"Part 2 interventions at N={max_n:,} (mean and 95% t intervals over 5 paired repeats)",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_combined_profiles(
    cells: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    path: Path,
) -> None:
    max_n = max(int(row["sample_count"]) for row in cells)
    profile_conditions = (
        ("baseline", "baseline", "#777777"),
        ("llm_like", "all structural conditions", "#7C3FA1"),
        ("llm_like_exact_shared", r"all + exact shared amplitude ($\rho=1$)", "#C44E52"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14.4, 8.4), constrained_layout=True)
    for axis, k in zip(axes[0], (2, 4, 8), strict=True):
        for condition, label, color in profile_conditions:
            selected = sorted(
                (
                    row
                    for row in cells
                    if row["suite"] == "combined"
                    and row["condition"] == condition
                    and row["k"] == k
                    and row["sample_count"] == max_n
                ),
                key=lambda row: int(row["rank"]),
            )
            ranks = np.asarray([int(row["rank"]) for row in selected])
            means = np.asarray([float(row["gride_over_k_mean"]) for row in selected])
            lows = np.asarray([float(row["gride_over_k_ci_low"]) for row in selected])
            highs = np.asarray([float(row["gride_over_k_ci_high"]) for row in selected])
            axis.plot(ranks, means, marker="o", linewidth=1.8, color=color, label=label)
            axis.fill_between(ranks, lows, highs, color=color, alpha=0.14)
        axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        axis.set_xscale("log", base=2)
        axis.set_xticks((2, 4, 8, 16, 32, 64), ("2", "4", "8", "16", "32", "64"))
        axis.set_xlabel("GRIDE neighbor rank")
        axis.set_ylabel("GRIDE ID / k")
        axis.set_title(
            f"A{(2, 4, 8).index(k) + 1}  k={k}", loc="left", fontsize=11, fontweight="bold"
        )
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    for condition, label, color in profile_conditions:
        selected = [_lookup_cell(cells, "combined", condition, k, max_n, 2) for k in (2, 4, 8)]
        means = np.asarray([float(row["gride_id_mean"]) for row in selected])
        lows = np.asarray([float(row["gride_id_ci_low"]) for row in selected])
        highs = np.asarray([float(row["gride_id_ci_high"]) for row in selected])
        axis.errorbar(
            (2, 4, 8),
            means,
            yerr=np.vstack((means - lows, highs - means)),
            marker="o",
            linewidth=1.8,
            capsize=2.5,
            color=color,
            label=label,
        )
    axis.plot((2, 8), (2, 8), color="#333333", linestyle="--", linewidth=1.0, label="ID = k")
    axis.set_xticks((2, 4, 8))
    axis.set_xlabel("active-feature count k")
    axis.set_ylabel("rank-2 GRIDE ID")
    axis.set_title("B1  Tracking is not calibration", loc="left", fontsize=11, fontweight="bold")
    axis.grid(alpha=0.22)

    axis = axes[1, 1]
    for condition, label, color, marker in (
        ("llm_like", r"$\rho=0.9$", "#7C3FA1", "o"),
        ("llm_like_exact_shared", r"$\rho=1$", "#C44E52", "s"),
    ):
        selected = [row for row in diagnostics if row["condition"] == condition]
        x = np.asarray([float(row["mean_feature_source_rank_mean"]) for row in selected])
        y = np.asarray([float(row["gride_rank_2_mean"]) for row in selected])
        y_low = np.asarray([float(row["gride_rank_2_ci_low"]) for row in selected])
        y_high = np.asarray([float(row["gride_rank_2_ci_high"]) for row in selected])
        axis.errorbar(
            x,
            y,
            yerr=np.vstack((y - y_low, y_high - y)),
            color=color,
            marker=marker,
            linewidth=1.5,
            capsize=2.5,
            label=label,
        )
        for row, x_value, y_value in zip(selected, x, y, strict=True):
            axis.annotate(f"k={row['k']}", (x_value, y_value), xytext=(4, 3), textcoords="offset points", fontsize=8)
    maximum = max(axis.get_xlim()[1], axis.get_ylim()[1])
    axis.plot((0, maximum), (0, maximum), color="#333333", linestyle="--", linewidth=1.0)
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("known feature-source rank")
    axis.set_ylabel("rank-2 GRIDE ID")
    axis.set_title("B2  Pooled ID vs local source rank", loc="left", fontsize=11, fontweight="bold")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.22)

    axis = axes[1, 2]
    noise_conditions = CONDITION_ORDER["noise"]
    noise_values = np.asarray([float(condition.split("=")[1]) for condition in noise_conditions])
    for k in (2, 4, 8):
        selected = [_lookup_cell(cells, "noise", condition, k, max_n, 2) for condition in noise_conditions]
        means = np.asarray([float(row["gride_id_mean"]) for row in selected])
        lows = np.asarray([float(row["gride_id_ci_low"]) for row in selected])
        highs = np.asarray([float(row["gride_id_ci_high"]) for row in selected])
        axis.errorbar(
            noise_values,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            color=COLORS[k],
            marker="o",
            linewidth=1.8,
            capsize=2.5,
            label=f"k={k}",
        )
    axis.set_xlabel("noise RMS / signal RMS")
    axis.set_ylabel("rank-2 GRIDE ID")
    axis.set_title("B3  Dense noise adds scale-dependent ID", loc="left", fontsize=11, fontweight="bold")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.22)

    figure.suptitle(
        f"All LLM-like conditions together at N={max_n:,}: monotone tracking survives, equality usually does not",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _print_key_table(
    cells: list[dict[str, object]], paired: list[dict[str, object]], diagnostics: list[dict[str, object]]
) -> None:
    max_n = max(int(row["sample_count"]) for row in cells)
    print(f"N={max_n:,}, GRIDE rank=2")
    print("condition                  k       ID [95% CI]       ID/k    source rank")
    for condition in ("baseline", "llm_like", "llm_like_exact_shared"):
        for k in (2, 4, 8):
            cell = _lookup_cell(cells, "combined", condition, k, max_n, 2)
            diagnostic = next(
                row for row in diagnostics if row["condition"] == condition and row["k"] == k
            )
            print(
                f"{condition:26s} {k:1d}  {cell['gride_id_mean']:6.3f} "
                f"[{cell['gride_id_ci_low']:6.3f}, {cell['gride_id_ci_high']:6.3f}]  "
                f"{cell['gride_over_k_mean']:6.3f}  {diagnostic['mean_feature_source_rank_mean']:6.3f}"
            )
    print("\nPaired rank-2 effects at N=max")
    print("contrast                                  k      delta [95% CI]   repeats decreasing")
    wanted = (
        ("combined", "baseline", "llm_like"),
        ("combined", "llm_like", "llm_like_exact_shared"),
        ("noise", "noise=0", "noise=0.2"),
    )
    for suite, reference, target in wanted:
        for k in (2, 4, 8):
            row = next(
                row
                for row in paired
                if row["suite"] == suite
                and row["reference_condition"] == reference
                and row["target_condition"] == target
                and row["k"] == k
                and row["sample_count"] == max_n
                and row["rank"] == 2
            )
            print(
                f"{reference + ' -> ' + target:41s} {k:1d}  {row['mean_delta']:7.3f} "
                f"[{row['delta_ci_low']:7.3f}, {row['delta_ci_high']:7.3f}]  "
                f"{int(round(5 * float(row['negative_repeat_fraction'])))}/5"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default="169194")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--assets-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()

    profiles_by_suite: dict[str, list[dict[str, str]]] = {}
    summaries_by_suite: dict[str, list[dict[str, str]]] = {}
    for suite in SUITES:
        suite_dir = args.results_root / f"part2-{suite}-{args.job_id}"
        profiles_by_suite[suite] = _read_csv(suite_dir / "gride_profiles.csv")
        summaries_by_suite[suite] = _read_csv(suite_dir / "summary.csv")

    paired = paired_effects(profiles_by_suite)
    sample_effects = sample_size_effects(profiles_by_suite)
    tracking = tracking_diagnostics(profiles_by_suite)
    cells = cell_means(profiles_by_suite)
    diagnostics = combined_diagnostics(summaries_by_suite["combined"])

    output_dir = args.results_root / f"part2-analysis-{args.job_id}"
    _write_csv(output_dir / "paired_effects.csv", paired)
    _write_csv(output_dir / "sample_size_effects.csv", sample_effects)
    _write_csv(output_dir / "tracking_diagnostics.csv", tracking)
    _write_csv(output_dir / "cell_means.csv", cells)
    _write_csv(output_dir / "combined_diagnostics.csv", diagnostics)
    plot_single_factor_effects(cells, args.assets_dir / "part2-single-factor-effects.png")
    plot_combined_profiles(cells, diagnostics, args.assets_dir / "part2-combined-profiles.png")
    _print_key_table(cells, paired, diagnostics)
    print(f"\nWrote {len(paired)} paired contrasts to {output_dir / 'paired_effects.csv'}")
    print(
        f"Wrote {len(sample_effects)} nested-N contrasts to "
        f"{output_dir / 'sample_size_effects.csv'}"
    )
    print(f"Wrote {len(cells)} cell estimates to {output_dir / 'cell_means.csv'}")


if __name__ == "__main__":
    main()

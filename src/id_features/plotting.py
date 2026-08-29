"""Minimal figures that retain GRIDE's scale dependence."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _plot_support_pool(results_dir: Path, profile_rows: list[dict[str, str]]) -> Path:
    """Plot rank-2 recovery by samples/support and the rank-induced purity transition."""

    grouped: dict[tuple[int, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in profile_rows:
        key = (
            int(row["k"]),
            int(row["support_pool_size"]),
            int(row["id_sample_count"]),
            int(row["rank"]),
        )
        grouped[key].append(row)
    means = {
        key: {
            "id": float(np.mean([float(row["gride_id"]) for row in rows])),
            "purity": float(
                np.mean([float(row["all_same_support_fraction"]) for row in rows])
            ),
        }
        for key, rows in grouped.items()
    }
    k_values = sorted({key[0] for key in means})
    b_values = sorted({key[1] for key in means})

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    flat_axes = list(axes.flat)
    for axis, k in zip(flat_axes, k_values, strict=False):
        for pool_size in b_values:
            cells = sorted(
                (
                    n_samples // pool_size,
                    value["id"] / k,
                )
                for (cell_k, b, n_samples, rank), value in means.items()
                if cell_k == k and b == pool_size and rank == 2
            )
            axis.plot(
                [cell[0] for cell in cells],
                [cell[1] for cell in cells],
                marker="o",
                label=f"B={pool_size}",
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.axhspan(0.75, 1.25, color="black", alpha=0.06)
        axis.set_xscale("log", base=2)
        axis.set_title(f"k={k}: local recovery")
        axis.set_xlabel("samples per support N/B")
        axis.set_ylabel("rank-2 GRIDE ID / k")
        axis.legend(fontsize=8)

    diagnostic_axis = flat_axes[-1]
    for k in k_values:
        normalized_rank = []
        purity = []
        for (cell_k, pool_size, n_samples, rank), value in means.items():
            if cell_k == k:
                normalized_rank.append(rank / (n_samples / pool_size))
                purity.append(value["purity"])
        diagnostic_axis.scatter(normalized_rank, purity, alpha=0.55, s=22, label=f"k={k}")
    diagnostic_axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
    diagnostic_axis.set_xscale("log", base=2)
    diagnostic_axis.set_ylim(-0.04, 1.04)
    diagnostic_axis.set_xlabel("GRIDE rank / samples per support")
    diagnostic_axis.set_ylabel("fully same-support neighborhood fraction")
    diagnostic_axis.set_title("Rank-induced support crossing")
    diagnostic_axis.legend(fontsize=8)

    for axis in flat_axes[len(k_values) : -1]:
        axis.set_visible(False)
    fig.tight_layout()
    output = results_dir / "overview.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _plot_manifold_geometry(
    results_dir: Path,
    profile_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> Path:
    """Relate known-ID recovery to scale, anisotropy, density, and support changes."""

    grouped: dict[tuple[str, int, int, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        key = (
            row["representation_kind"],
            int(row["intrinsic_dim"]),
            int(row["sample_count"]),
            int(row["expected_active"]),
            float(row["feature_strength"]),
        )
        grouped[key].append(row)
    def finite_mean(rows: list[dict[str, str]], field: str) -> float:
        values = np.asarray([float(row[field]) for row in rows])
        return float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")

    means = {
        key: {
            field: finite_mean(rows, field)
            for field in (
                "local_gride_id_rank_2",
                "rank_2_relative_error_to_true_id",
                "mean_latent_1nn_chord",
                "mean_local_condition_number",
                "local_log_volume_std",
                "mean_1nn_active_jaccard",
            )
        }
        for key, rows in grouped.items()
    }
    combined = [(key, value) for key, value in means.items() if key[0] == "combined"]
    d_values = sorted({key[1] for key, _ in combined})
    markers = {
        intrinsic_dim: ("o", "s", "^", "D", "v")[index % 5]
        for index, intrinsic_dim in enumerate(d_values)
    }
    strengths = sorted({key[4] for key, _ in combined})
    colors = {strength: plt.cm.viridis(index / max(len(strengths) - 1, 1)) for index, strength in enumerate(strengths)}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    baseline_axis = axes[0, 0]
    for intrinsic_dim in d_values:
        cells = sorted(
            (key[2], value["local_gride_id_rank_2"] / intrinsic_dim)
            for key, value in means.items()
            if key[0] == "sphere" and key[1] == intrinsic_dim
        )
        baseline_axis.plot(
            [cell[0] for cell in cells],
            [cell[1] for cell in cells],
            marker="o",
            label=f"d={intrinsic_dim}",
        )
    baseline_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    baseline_axis.axhspan(0.9, 1.1, color="black", alpha=0.06)
    baseline_axis.set_xscale("log", base=2)
    baseline_axis.set_xlabel("sample count N")
    baseline_axis.set_ylabel("rank-2 GRIDE ID / d")
    baseline_axis.set_title("Uniform boundaryless baseline")
    baseline_axis.legend(fontsize=8)

    strength_axis = axes[0, 1]
    for intrinsic_dim in d_values:
        for strength in strengths:
            ratios = [
                value["local_gride_id_rank_2"] / intrinsic_dim
                for key, value in combined
                if key[1] == intrinsic_dim and key[4] == strength
            ]
            strength_axis.scatter(
                [strength] * len(ratios),
                ratios,
                color=colors[strength],
                alpha=0.55,
                s=24,
                marker=markers[intrinsic_dim],
            )
    strength_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    strength_axis.axhspan(0.9, 1.1, color="black", alpha=0.06)
    strength_axis.set_xscale("log", base=2)
    strength_axis.set_xlabel("sparse feature strength gamma")
    strength_axis.set_ylabel("rank-2 GRIDE ID / d")
    strength_axis.set_title("Known-ID nonlinear embeddings")

    radius_axis = axes[0, 2]
    for intrinsic_dim in d_values:
        cells = [(value["mean_latent_1nn_chord"], value["rank_2_relative_error_to_true_id"]) for key, value in combined if key[1] == intrinsic_dim]
        radius_axis.scatter(
            [cell[0] for cell in cells],
            [cell[1] for cell in cells],
            alpha=0.55,
            s=26,
            label=f"d={intrinsic_dim}",
        )
    radius_axis.set_xlabel("mean latent 1-NN chord")
    radius_axis.set_ylabel("rank-2 relative ID error")
    radius_axis.set_title("Finite neighborhood radius")
    radius_axis.legend(fontsize=8)

    diagnostic_specs = (
        ("mean_local_condition_number", "mean local condition number", "Local anisotropy"),
        ("local_log_volume_std", "std of local log-volume", "Density / metric variation"),
        ("mean_1nn_active_jaccard", "mean 1-NN active-set Jaccard", "Support continuity"),
    )
    for axis, (field, label, title) in zip(axes[1], diagnostic_specs, strict=True):
        for strength in strengths:
            cells = [
                (value[field], value["rank_2_relative_error_to_true_id"])
                for key, value in combined
                if key[4] == strength
            ]
            axis.scatter(
                [cell[0] for cell in cells],
                [cell[1] for cell in cells],
                color=colors[strength],
                alpha=0.6,
                s=26,
                label=f"gamma={strength:g}",
            )
        axis.set_xlabel(label)
        axis.set_ylabel("rank-2 relative ID error")
        axis.set_title(title)
    axes[1, 0].legend(fontsize=8)

    fig.tight_layout()
    output = results_dir / "overview.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _plot_bid_audit(
    results_dir: Path,
    summary_rows: list[dict[str, str]],
) -> Path:
    """Compare continuous-ID recovery, binary encodings, and fit quantiles."""

    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(int(row["k"]), int(row["support_pool_size"]))].append(row)

    def mean_field(k: int, pool_size: int, field: str) -> float:
        values = np.asarray([float(row[field]) for row in grouped[(k, pool_size)]])
        return float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")

    k_values = sorted({key[0] for key in grouped})
    pool_sizes = sorted({key[1] for key in grouped})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    fixed_axis = axes[0]
    fixed_axis.plot(k_values, k_values, color="black", linestyle="--", label="target k")
    for field, label, marker in (
        ("gride_rank_2", "GRIDE rank 2", "o"),
        ("independent_bits_bid", "independent-bit BID", "s"),
        ("latent_sign_bid", "latent-sign BID", "^"),
        ("activation_sign_bid", "activation-sign BID", "D"),
        ("activation_two_bit_bid", "activation two-bit BID", "v"),
    ):
        fixed_axis.plot(
            k_values,
            [mean_field(k, 1, field) for k in k_values],
            marker=marker,
            label=label,
        )
    fixed_axis.set_xlabel("known continuous dimension k")
    fixed_axis.set_ylabel("estimated dimension")
    fixed_axis.set_title("Fixed support (B=1)")
    fixed_axis.legend(fontsize=7)

    pooled_axis = axes[1]
    for pool_size in pool_sizes:
        for field, label, linestyle in (
            ("activation_sign_bid", "one bit", "-"),
            ("activation_two_bit_bid", "two bit", ":"),
        ):
            pooled_axis.plot(
                k_values,
                [mean_field(k, pool_size, field) / k for k in k_values],
                marker="o",
                linestyle=linestyle,
                label=f"B={pool_size}, {label}",
            )
    pooled_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    pooled_axis.axhspan(0.9, 1.1, color="black", alpha=0.06)
    pooled_axis.set_xlabel("k")
    pooled_axis.set_ylabel("activation BID / k")
    pooled_axis.set_title("Support mixing and quantization")
    pooled_axis.legend(fontsize=7)

    with (results_dir / "bid_fits.csv").open(encoding="utf-8") as file:
        fit_rows = [
            row
            for row in csv.DictReader(file)
            if row["fit_status"] == "ok" and row["binary_view"] == "activation-sign"
        ]
    quantile_axis = axes[2]
    for pool_size in pool_sizes:
        alpha_values = sorted(
            {
                float(row["alpha_max"])
                for row in fit_rows
                if int(row["support_pool_size"]) == pool_size
            }
        )
        ratios = []
        for alpha in alpha_values:
            cells = [
                float(row["bid"]) / int(row["k"])
                for row in fit_rows
                if int(row["support_pool_size"]) == pool_size
                and np.isclose(float(row["alpha_max"]), alpha)
            ]
            ratios.append(float(np.mean(cells)))
        quantile_axis.plot(alpha_values, ratios, marker="o", label=f"B={pool_size}")
    quantile_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    quantile_axis.axhspan(0.9, 1.1, color="black", alpha=0.06)
    quantile_axis.set_xlabel("alpha_max (fitted distance quantile)")
    quantile_axis.set_ylabel("mean activation-sign BID / k")
    quantile_axis.set_title("BID scale stability")
    quantile_axis.legend(fontsize=8)

    fig.tight_layout()
    output = results_dir / "overview.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_results(results_dir: Path) -> Path:
    """Create one two-panel PNG from a completed results directory."""

    with (results_dir / "gride_profiles.csv").open(encoding="utf-8") as file:
        profile_rows = list(csv.DictReader(file))
    with (results_dir / "summary.csv").open(encoding="utf-8") as file:
        summary_rows = list(csv.DictReader(file))
    if not profile_rows or not summary_rows:
        raise ValueError("results directory does not contain non-empty CSV files")

    experiment = summary_rows[0]["experiment"]
    if experiment == "support-pool":
        return _plot_support_pool(results_dir, profile_rows)
    if experiment == "manifold-geometry":
        return _plot_manifold_geometry(results_dir, profile_rows, summary_rows)
    if experiment == "bid-audit":
        return _plot_bid_audit(results_dir, summary_rows)
    if experiment == "part2-structured":
        from .structured_experiment import plot_structured_results

        return plot_structured_results(results_dir, profile_rows, summary_rows)
    condition_name = "condition"
    profiles: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in profile_rows:
        profiles[row[condition_name]].append(row)

    fig, (profile_ax, access_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
    for condition, rows in sorted(profiles.items()):
        by_rank: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            by_rank[int(row["rank"])].append(float(row["gride_id"]))
        ranks = np.array(sorted(by_rank))
        ids = np.array([np.mean(by_rank[rank]) for rank in ranks])
        profile_ax.plot(ranks, ids, marker="o", label=condition)
    profile_ax.set_xscale("log", base=2)
    profile_ax.set_xlabel("GRIDE upper neighbour rank")
    profile_ax.set_ylabel("intrinsic dimension")
    profile_ax.set_title("Multi-scale GRIDE profile")
    profile_ax.legend(fontsize=8)

    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        by_condition[row[condition_name]].append(row)
    conditions = sorted(by_condition)
    local_ids = np.array(
        [np.mean([float(r["local_gride_id_rank_2"]) for r in by_condition[c]]) for c in conditions]
    )
    aurocs = np.array(
        [np.mean([float(r["mean_feature_auroc"]) for r in by_condition[c]]) for c in conditions]
    )
    access_ax.scatter(local_ids, aurocs, s=55)
    for x, y, condition in zip(local_ids, aurocs, conditions, strict=True):
        access_ax.annotate(condition, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    access_ax.set_xlabel("local GRIDE ID (rank 2)")
    access_ax.set_ylabel("mean held-out feature AUROC")
    access_ax.set_title("ID and linear feature accessibility")
    fig.tight_layout()
    output = results_dir / "overview.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output

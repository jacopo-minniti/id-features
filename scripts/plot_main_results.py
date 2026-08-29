#!/usr/bin/env python3
"""Create the three compact figures used by docs/main.md."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


COLORS = {2: "#3B6FB6", 4: "#E07A2D", 8: "#2F8F61", 16: "#C44E52"}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    half = float(t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array)))
    return mean, mean - half, mean + half


def _errorbar(axis: plt.Axes, x: list[float], grouped: list[list[float]], **kwargs: object) -> None:
    stats = [_mean_ci(values) for values in grouped]
    means = np.asarray([row[0] for row in stats])
    lows = np.asarray([row[1] for row in stats])
    highs = np.asarray([row[2] for row in stats])
    axis.errorbar(
        x,
        means,
        yerr=np.vstack((means - lows, highs - means)),
        capsize=2.5,
        linewidth=1.8,
        marker="o",
        **kwargs,
    )


def plot_part1(root: Path, assets: Path) -> None:
    rows = _read(root / "results/part1-fixed-support-169169/summary.csv")
    rows = [row for row in rows if int(row["id_sample_count"]) == 16_384]
    ks = sorted({int(row["k"]) for row in rows})
    grouped = [
        [float(row["local_gride_id_rank_2"]) for row in rows if int(row["k"]) == k]
        for k in ks
    ]
    figure, axis = plt.subplots(figsize=(6.3, 4.2), constrained_layout=True)
    _errorbar(axis, ks, grouped, color="#3B6FB6", label="GRIDE, rank 2")
    axis.plot((1, 16), (1, 16), color="#222222", linestyle="--", label="population ID = k")
    axis.set_xticks(ks)
    axis.set_xlabel("active features k")
    axis.set_ylabel("intrinsic dimension")
    axis.set_title("Fixed support: theorem and finite-sample GRIDE", loc="left", fontweight="bold")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(assets / "main-part1.png", dpi=190)
    plt.close(figure)


def plot_part2(root: Path, assets: Path) -> None:
    sphere = _read(root / "results/manifold-geometry-168480/summary.csv")
    sphere = [row for row in sphere if row["representation_kind"] == "sphere"]
    profiles = _read(root / "results/part1-fixed-support-169169/gride_profiles.csv")
    profiles = [row for row in profiles if int(row["id_sample_count"]) == 16_384]

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    axis = axes[0]
    for n_samples, color in ((4_096, "#6B8ECA"), (16_384, "#274E87")):
        selected = [row for row in sphere if int(row["sample_count"]) == n_samples]
        dims = sorted({int(row["intrinsic_dim"]) for row in selected})
        grouped = [
            [
                float(row["local_gride_id_rank_2"]) / d
                for row in selected
                if int(row["intrinsic_dim"]) == d
            ]
            for d in dims
        ]
        _errorbar(axis, dims, grouped, color=color, label=f"N={n_samples:,}")
    axis.axhline(1.0, color="#222222", linestyle="--")
    axis.set_xticks((4, 8, 16))
    axis.set_xlabel("known sphere dimension d")
    axis.set_ylabel("rank-2 GRIDE ID / d")
    axis.set_title("A  Finite-sample bias on a known manifold", loc="left", fontweight="bold")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)

    axis = axes[1]
    for k in (2, 4, 8, 16):
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in profiles:
            if int(row["k"]) == k:
                grouped[int(row["rank"])].append(float(row["gride_id"]) / k)
        ranks = sorted(grouped)
        stats = [_mean_ci(grouped[rank]) for rank in ranks]
        means = np.asarray([value[0] for value in stats])
        lows = np.asarray([value[1] for value in stats])
        highs = np.asarray([value[2] for value in stats])
        axis.plot(ranks, means, color=COLORS[k], marker="o", linewidth=1.8, label=f"k={k}")
        axis.fill_between(ranks, lows, highs, color=COLORS[k], alpha=0.13)
    axis.axhline(1.0, color="#222222", linestyle="--")
    axis.set_xscale("log", base=2)
    axis.set_xticks((2, 4, 8, 16, 32, 64), ("2", "4", "8", "16", "32", "64"))
    axis.set_xlabel("GRIDE neighbor rank")
    axis.set_ylabel("GRIDE ID / k")
    axis.set_title("B  The estimate depends on neighborhood scale", loc="left", fontweight="bold")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=2)
    figure.savefig(assets / "main-part2-gride-limits.png", dpi=190)
    plt.close(figure)


def plot_part3(root: Path, assets: Path, pool_dir: Path) -> None:
    combined = _read(root / "results/part2-analysis-169194/cell_means.csv")
    pool = _read(pool_dir / "cell_means.csv")
    max_n = 16_384

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), constrained_layout=True)
    axis = axes[0]
    styles = (
        ("baseline", "unstructured", "#777777"),
        ("llm_like", "all structure", "#7C3FA1"),
        ("llm_like_exact_shared", r"all + $\rho=1$", "#C44E52"),
    )
    for condition, label, color in styles:
        selected = [
            row
            for row in combined
            if row["suite"] == "combined"
            and row["condition"] == condition
            and int(row["sample_count"]) == max_n
            and int(row["rank"]) == 2
        ]
        selected.sort(key=lambda row: int(row["k"]))
        ks = [int(row["k"]) for row in selected]
        means = np.asarray([float(row["gride_id_mean"]) for row in selected])
        lows = np.asarray([float(row["gride_id_ci_low"]) for row in selected])
        highs = np.asarray([float(row["gride_id_ci_high"]) for row in selected])
        axis.errorbar(
            ks,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            color=color,
            marker="o",
            linewidth=1.8,
            capsize=2.5,
            label=label,
        )
    axis.plot((2, 8), (2, 8), color="#222222", linestyle="--", label="ID = k")
    axis.set_xticks((2, 4, 8))
    axis.set_xlabel("active features k")
    axis.set_ylabel("rank-2 GRIDE ID")
    axis.set_title("A  Structure lowers ID but does not count k", loc="left", fontweight="bold")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    for k in (4, 8):
        selected = [
            row
            for row in pool
            if np.isclose(float(row["amplitude_correlation"]), 1.0) and int(row["k"]) == k
        ]
        selected.sort(key=lambda row: float(row["samples_per_support"]))
        x = np.asarray([float(row["samples_per_support"]) for row in selected])
        y = np.asarray([float(row["gride_over_source_mean"]) for row in selected])
        lows = np.asarray([float(row["gride_over_source_ci_low"]) for row in selected])
        highs = np.asarray([float(row["gride_over_source_ci_high"]) for row in selected])
        axis.errorbar(
            x,
            y,
            yerr=np.vstack((y - lows, highs - y)),
            color=COLORS[k],
            marker="o",
            linewidth=1.8,
            capsize=2.5,
            label=f"k={k}",
        )
    axis.axhline(1.0, color="#222222", linestyle="--", label="ID = source rank")
    axis.set_xscale("log", base=2)
    axis.set_xticks((1, 4, 16, 64, 256, 1024, 4096, 16384))
    axis.set_xlabel("samples per exact support N/B")
    axis.set_ylabel("GRIDE ID / known source rank")
    axis.set_title("B  Repeated supports recover source dimension", loc="left", fontweight="bold")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8)

    axis = axes[2]
    noise_conditions = ("noise=0", "noise=0.01", "noise=0.05", "noise=0.2")
    noise_values = (0.0, 0.01, 0.05, 0.2)
    for k in (2, 4, 8):
        selected = [
            next(
                row
                for row in combined
                if row["suite"] == "noise"
                and row["condition"] == condition
                and int(row["k"]) == k
                and int(row["sample_count"]) == max_n
                and int(row["rank"]) == 2
            )
            for condition in noise_conditions
        ]
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
    axis.set_title("C  Dense residual variation raises ID", loc="left", fontweight="bold")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(assets / "main-part3-structured.png", dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    plot_part1(args.root, args.assets_dir)
    plot_part2(args.root, args.assets_dir)
    plot_part3(args.root, args.assets_dir, args.pool_dir)


if __name__ == "__main__":
    main()

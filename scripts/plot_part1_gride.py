#!/usr/bin/env python3
"""Plot the fixed-support GRIDE validation used in the Part 1 note."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, 0.0
    return mean, float(1.96 * array.std(ddof=1) / sqrt(len(array)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with (args.results_dir / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    if config["experiment"] != "support-pool" or config["support_pool_sizes"] != [1]:
        raise ValueError("Part 1 plotting requires a fixed-support (B=1) support-pool run")

    with (args.results_dir / "gride_profiles.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("gride_profiles.csv is empty")

    grouped: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["k"]), int(row["id_sample_count"]), int(row["rank"]))].append(
            float(row["gride_id"])
        )

    k_values = sorted({key[0] for key in grouped})
    n_values = sorted({key[1] for key in grouped})
    ranks = sorted({key[2] for key in grouped})
    colors = {
        n_samples: plt.cm.viridis(index / max(len(n_values) - 1, 1))
        for index, n_samples in enumerate(n_values)
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    recovery_axis = axes[0]
    recovery_axis.plot(k_values, k_values, color="black", linestyle="--", label="population ID = k")
    for n_samples in n_values:
        cells = [_mean_ci(grouped[(k, n_samples, 2)]) for k in k_values]
        recovery_axis.errorbar(
            k_values,
            [cell[0] for cell in cells],
            yerr=[cell[1] for cell in cells],
            marker="o",
            capsize=2,
            color=colors[n_samples],
            label=f"N={n_samples:,}",
        )
    recovery_axis.set_xlabel("active features k")
    recovery_axis.set_ylabel("rank-2 GRIDE ID")
    recovery_axis.set_title("Does GRIDE recover the theorem?")
    recovery_axis.legend(fontsize=8)

    ratio_axis = axes[1]
    ratio_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ratio_axis.axhspan(0.9, 1.1, color="black", alpha=0.07, label="within 10%")
    for n_samples in n_values:
        cells = [_mean_ci(grouped[(k, n_samples, 2)]) for k in k_values]
        ratio_axis.errorbar(
            k_values,
            [cell[0] / k for k, cell in zip(k_values, cells, strict=True)],
            yerr=[cell[1] / k for k, cell in zip(k_values, cells, strict=True)],
            marker="o",
            capsize=2,
            color=colors[n_samples],
            label=f"N={n_samples:,}",
        )
    ratio_axis.set_xlabel("active features k")
    ratio_axis.set_ylabel("rank-2 GRIDE ID / k")
    ratio_axis.set_title("Finite-sample calibration")
    ratio_axis.legend(fontsize=8)

    profile_axis = axes[2]
    largest_n = n_values[-1]
    for k in k_values:
        cells = [_mean_ci(grouped[(k, largest_n, rank)]) for rank in ranks]
        means = np.asarray([cell[0] / k for cell in cells])
        cis = np.asarray([cell[1] / k for cell in cells])
        profile_axis.plot(ranks, means, marker="o", label=f"k={k}")
        profile_axis.fill_between(ranks, means - cis, means + cis, alpha=0.12)
    profile_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    profile_axis.axhspan(0.9, 1.1, color="black", alpha=0.07)
    profile_axis.set_xscale("log", base=2)
    profile_axis.set_xlabel("GRIDE upper-neighbour rank")
    profile_axis.set_ylabel("GRIDE ID / k")
    profile_axis.set_title(f"Scale profile at N={largest_n:,}")
    profile_axis.legend(fontsize=8)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print("| k | N | mean rank-2 ID | relative error | 95% CI half-width |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for k in k_values:
        for n_samples in n_values:
            mean, ci = _mean_ci(grouped[(k, n_samples, 2)])
            print(
                f"| {k} | {n_samples:,} | {mean:.3f} | {abs(mean - k) / k:.1%} | {ci:.3f} |"
            )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

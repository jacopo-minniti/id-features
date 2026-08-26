"""Minimal figures that retain GRIDE's scale dependence."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_results(results_dir: Path) -> Path:
    """Create one two-panel PNG from a completed results directory."""

    with (results_dir / "gride_profiles.csv").open(encoding="utf-8") as file:
        profile_rows = list(csv.DictReader(file))
    with (results_dir / "summary.csv").open(encoding="utf-8") as file:
        summary_rows = list(csv.DictReader(file))
    if not profile_rows or not summary_rows:
        raise ValueError("results directory does not contain non-empty CSV files")

    experiment = summary_rows[0]["experiment"]
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

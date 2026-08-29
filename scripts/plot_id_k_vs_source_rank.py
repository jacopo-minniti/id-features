"""Plot actual support-density results normalized by k and source rank."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/part3-support-pool-169258/cell_means.csv"
OUTPUT = ROOT / "docs/assets/id-k-vs-source-rank.png"


def main() -> None:
    data = pd.read_csv(INPUT)
    data = data[np.isclose(data["amplitude_correlation"], 1.0)].copy()
    data = data.sort_values(["k", "samples_per_support"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharex=True)
    colors = {4: "#e67e22", 8: "#2e8b68"}

    panels = [
        (
            "ID / active-feature count",
            lambda frame: frame["gride_id_mean"] / frame["k"],
            lambda frame: frame["gride_id_ci_low"] / frame["k"],
            lambda frame: frame["gride_id_ci_high"] / frame["k"],
            r"GRIDE ID / $k$",
            r"A  Does GRIDE recover active-feature count $k$?",
        ),
        (
            "ID / source rank",
            lambda frame: frame["gride_over_source_mean"],
            lambda frame: frame["gride_over_source_ci_low"],
            lambda frame: frame["gride_over_source_ci_high"],
            "GRIDE ID / known source rank",
            "B  Does GRIDE recover source dimension?",
        ),
    ]

    for ax, (_, get_y, get_low, get_high, ylabel, title) in zip(axes, panels, strict=True):
        for k in (4, 8):
            frame = data[data["k"] == k]
            x = frame["samples_per_support"].to_numpy()
            y = get_y(frame).to_numpy()
            low = get_low(frame).to_numpy()
            high = get_high(frame).to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - low, high - y]),
                marker="o",
                markersize=6,
                linewidth=2.2,
                capsize=3,
                color=colors[k],
                label=fr"$k={k}$",
            )

        ax.axhline(1, color="#333333", linestyle="--", linewidth=1.6, label="perfect recovery")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"Samples per exact support $N/B$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13.5, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)

    ticks = 2 ** np.arange(0, 15, 2)
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels([fr"$2^{{{power}}}$" for power in range(0, 15, 2)])

    axes[0].set_ylim(0, 3.05)
    axes[1].set_ylim(0, 7.85)
    fig.suptitle(
        r"Actual support-density experiment with exact shared amplitudes ($\rho=1$)",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

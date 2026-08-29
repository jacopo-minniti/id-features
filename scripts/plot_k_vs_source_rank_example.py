"""Plot the distinction between active-feature count and source rank."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parents[1] / "docs/assets/k-vs-source-rank-example.png"


def main() -> None:
    feature_names = ["French", "France", "European", "Latin script"]
    relative_amplitudes = np.array([1.0, 0.8, 0.6, 0.5])

    # All four feature amplitudes are controlled by the same scalar source u.
    rng = np.random.default_rng(7)
    source_strength = np.sort(rng.uniform(0.15, 1.2, size=70))
    direction = np.array([1.0, 0.62])
    projected_activations = source_strength[:, None] * direction[None, :]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("Active-feature count and source rank are different", fontsize=17, weight="bold")

    ax = axes[0]
    bars = ax.bar(
        np.arange(len(feature_names)),
        relative_amplitudes,
        color=["#4776b4", "#5b8fc6", "#73a7d3", "#91bfdf"],
        width=0.68,
    )
    ax.set_title(r"Four SAE features are active: $k=4$", fontsize=14, weight="bold")
    ax.set_ylabel("Feature amplitude")
    ax.set_xticks(np.arange(len(feature_names)), feature_names, rotation=18, ha="right")
    ax.set_ylim(0, 1.22)
    ax.text(
        0.5,
        0.94,
        "All four are nonzero",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
    )
    for bar, value in zip(bars, relative_amplitudes, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.1f}", ha="center")

    ax = axes[1]
    scatter = ax.scatter(
        projected_activations[:, 0],
        projected_activations[:, 1],
        c=source_strength,
        cmap="viridis",
        s=35,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.plot(
        [0, projected_activations[:, 0].max() * 1.05],
        [0, projected_activations[:, 1].max() * 1.05],
        color="#333333",
        linewidth=1.4,
        linestyle="--",
    )
    ax.set_title(r"One shared factor varies: source rank $=1$", fontsize=14, weight="bold")
    ax.set_xlabel("Activation direction 1")
    ax.set_ylabel("Activation direction 2")
    ax.text(
        0.05,
        0.94,
        r"$(z_1,z_2,z_3,z_4)=u(1,.8,.6,.5)$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Shared source strength $u$")

    for panel in axes:
        panel.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.015,
        r"GRIDE sees the one-dimensional variation, not the four nonzero feature coordinates.",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.92))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

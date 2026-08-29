#!/usr/bin/env python3
"""Targeted support-pool check for the structured sparse-feature model."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from id_features.metrics import measure_gride
from id_features.structured import make_feature_groups, make_modular_feature_matrix


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    half_width = float(t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array)))
    return mean, mean - half_width, mean + half_width


def _within_module_ranks(groups: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(groups)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        ranks[indices] = np.arange(1, len(indices) + 1)
    return ranks


def _draw_unique_support_pool(
    groups: np.ndarray,
    k: int,
    pool_size: int,
    *,
    module_bias: float,
    zipf_alpha: float,
    seed_parts: tuple[int, ...],
) -> np.ndarray:
    """Draw an ordered pool of distinct structured supports in bounded batches."""

    feature_count = len(groups)
    module_count = int(groups.max()) + 1
    ranks = _within_module_ranks(groups)
    log_frequency = -zipf_alpha * np.log(ranks)
    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    batch_size = max(4_096, 2 * pool_size)
    for batch in range(64):
        rng = np.random.default_rng(np.random.SeedSequence([*seed_parts, batch]))
        contexts = rng.integers(module_count, size=batch_size)
        scores = rng.gumbel(size=(batch_size, feature_count))
        scores += log_frequency[None, :]
        scores += module_bias * (groups[None, :] == contexts[:, None])
        supports = np.argpartition(scores, kth=feature_count - k, axis=1)[:, -k:]
        supports.sort(axis=1)
        for support in supports:
            key = tuple(int(index) for index in support)
            if key not in seen:
                seen.add(key)
                unique.append(key)
                if len(unique) == pool_size:
                    return np.asarray(unique, dtype=np.int_)
    raise RuntimeError(f"could only draw {len(unique)} distinct supports; need {pool_size}")


def _balanced_support_ids(
    support_count: int, n_samples: int, seed_parts: tuple[int, ...]
) -> np.ndarray:
    if n_samples % support_count != 0:
        raise ValueError("support_count must divide n_samples")
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    blocks = [rng.permutation(support_count) for _ in range(n_samples // support_count)]
    return np.concatenate(blocks).astype(np.int_)


def _representations_from_pool(
    matrix: np.ndarray,
    groups: np.ndarray,
    support_pool: np.ndarray,
    support_ids: np.ndarray,
    amplitude_correlation: float,
    amplitude_log_scale: float,
    seed_parts: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    supports = support_pool[support_ids]
    n_samples, k = supports.shape
    module_count = int(groups.max()) + 1
    active_groups = groups[supports]
    shared_rng = np.random.default_rng(np.random.SeedSequence([*seed_parts, 1]))
    innovation_rng = np.random.default_rng(np.random.SeedSequence([*seed_parts, 2]))
    shared = shared_rng.normal(size=(n_samples, module_count))
    innovations = innovation_rng.normal(size=(n_samples, k))
    shared_values = np.take_along_axis(shared, active_groups, axis=1)
    standardized = (
        np.sqrt(amplitude_correlation) * shared_values
        + np.sqrt(1.0 - amplitude_correlation) * innovations
    )
    amplitudes = np.exp(amplitude_log_scale * standardized - 0.5 * amplitude_log_scale**2)
    representations = np.einsum("dnk,nk->nd", matrix[:, supports], amplitudes, optimize=True)

    module_counts = np.zeros((n_samples, module_count), dtype=np.int_)
    rows = np.repeat(np.arange(n_samples), k)
    np.add.at(module_counts, (rows, active_groups.ravel()), 1)
    active_modules = (module_counts > 0).sum(axis=1).astype(float)
    source_ranks = active_modules if np.isclose(amplitude_correlation, 1.0) else np.full(n_samples, k)
    return np.asarray(representations, dtype=float), supports, source_ranks


def run_task(args: argparse.Namespace) -> None:
    b_values = tuple(int(value) for value in args.b_values.split(","))
    n_values = tuple(int(value) for value in args.n_values.split(","))
    if tuple(sorted(b_values)) != b_values or tuple(sorted(n_values)) != n_values:
        raise ValueError("B and N values must be increasing")
    if any(n % b != 0 for b in b_values for n in n_values):
        raise ValueError("every B must divide every N for the balanced nested design")

    groups = make_feature_groups(args.features, args.modules)
    summary_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    total = args.repeats * len(b_values) * len(n_values)
    completed = 0
    for repeat in range(args.repeats):
        matrix_rng = np.random.default_rng(np.random.SeedSequence([args.seed, repeat, 31]))
        matrix = make_modular_feature_matrix(
            args.dim, args.features, groups, args.coherence, matrix_rng
        )
        largest_pool = _draw_unique_support_pool(
            groups,
            args.k,
            max(b_values),
            module_bias=args.module_bias,
            zipf_alpha=args.zipf_alpha,
            seed_parts=(args.seed, repeat, args.k, 32),
        )
        for support_count in b_values:
            pool = largest_pool[:support_count]
            support_ids = _balanced_support_ids(
                support_count,
                max(n_values),
                (args.seed, repeat, args.k, support_count, 33),
            )
            representations, _, source_ranks = _representations_from_pool(
                matrix,
                groups,
                pool,
                support_ids,
                args.rho,
                args.amplitude_log_scale,
                (args.seed, repeat, args.k, support_count, 34),
            )
            for n_samples in n_values:
                profile = measure_gride(
                    representations[:n_samples],
                    args.gride_range_max,
                    n_jobs=args.gride_jobs,
                    support_ids=support_ids[:n_samples],
                )
                if profile.mean_same_support_fraction is None or profile.all_same_support_fraction is None:
                    raise RuntimeError("support diagnostics were not returned")
                summary_rows.append(
                    {
                        "k": args.k,
                        "amplitude_correlation": args.rho,
                        "support_count": support_count,
                        "sample_count": n_samples,
                        "samples_per_support": n_samples / support_count,
                        "repeat": repeat,
                        "gride_rank_2": float(profile.ids[0]),
                        "mean_source_rank": float(source_ranks[:n_samples].mean()),
                        "rank_2_same_support_neighbor_fraction": float(
                            profile.mean_same_support_fraction[0]
                        ),
                        "rank_2_fully_support_pure_fraction": float(
                            profile.all_same_support_fraction[0]
                        ),
                    }
                )
                for rank, scale, estimate, error, same, pure in zip(
                    profile.ranks,
                    profile.scales,
                    profile.ids,
                    profile.errors,
                    profile.mean_same_support_fraction,
                    profile.all_same_support_fraction,
                    strict=True,
                ):
                    profile_rows.append(
                        {
                            "k": args.k,
                            "amplitude_correlation": args.rho,
                            "support_count": support_count,
                            "sample_count": n_samples,
                            "samples_per_support": n_samples / support_count,
                            "repeat": repeat,
                            "rank": int(rank),
                            "scale": float(scale),
                            "gride_id": float(estimate),
                            "gride_error": float(error),
                            "mean_same_support_neighbor_fraction": float(same),
                            "fully_support_pure_fraction": float(pure),
                            "mean_source_rank": float(source_ranks[:n_samples].mean()),
                        }
                    )
                completed += 1
                print(
                    f"[{completed}/{total}] k={args.k} rho={args.rho:g} B={support_count} "
                    f"N={n_samples} repeat={repeat + 1}/{args.repeats} ID2={profile.ids[0]:.3f}",
                    flush=True,
                )
    _write_csv(args.output / "summary.csv", summary_rows)
    _write_csv(args.output / "gride_profiles.csv", profile_rows)


def analyze(args: argparse.Namespace) -> None:
    summary_rows: list[dict[str, str]] = []
    profile_rows: list[dict[str, str]] = []
    for task_dir in sorted(args.input.glob("task-*")):
        summary_rows.extend(_read_csv(task_dir / "summary.csv"))
        profile_rows.extend(_read_csv(task_dir / "gride_profiles.csv"))
    if not summary_rows or not profile_rows:
        raise ValueError(f"no task results found under {args.input}")
    _write_csv(args.input / "summary.csv", summary_rows)
    _write_csv(args.input / "gride_profiles.csv", profile_rows)

    max_n = max(int(row["sample_count"]) for row in summary_rows)
    grouped: dict[tuple[float, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[
            (
                float(row["amplitude_correlation"]),
                int(row["k"]),
                int(row["support_count"]),
                int(row["sample_count"]),
            )
        ].append(row)

    cell_rows: list[dict[str, object]] = []
    for (rho, k, support_count, sample_count), rows in sorted(grouped.items()):
        ratio_values = [float(row["gride_rank_2"]) / float(row["mean_source_rank"]) for row in rows]
        ratio_mean, ratio_low, ratio_high = _mean_ci(ratio_values)
        id_mean, id_low, id_high = _mean_ci([float(row["gride_rank_2"]) for row in rows])
        source_mean, source_low, source_high = _mean_ci(
            [float(row["mean_source_rank"]) for row in rows]
        )
        purity_mean, purity_low, purity_high = _mean_ci(
            [float(row["rank_2_same_support_neighbor_fraction"]) for row in rows]
        )
        cell_rows.append(
            {
                "amplitude_correlation": rho,
                "k": k,
                "support_count": support_count,
                "sample_count": sample_count,
                "samples_per_support": sample_count / support_count,
                "repeat_count": len(rows),
                "gride_id_mean": id_mean,
                "gride_id_ci_low": id_low,
                "gride_id_ci_high": id_high,
                "source_rank_mean": source_mean,
                "source_rank_ci_low": source_low,
                "source_rank_ci_high": source_high,
                "gride_over_source_mean": ratio_mean,
                "gride_over_source_ci_low": ratio_low,
                "gride_over_source_ci_high": ratio_high,
                "same_support_fraction_mean": purity_mean,
                "same_support_fraction_ci_low": purity_low,
                "same_support_fraction_ci_high": purity_high,
            }
        )
    _write_csv(args.input / "cell_means.csv", cell_rows)

    colors = {4: "#E07A2D", 8: "#2F8F61"}
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    for axis, rho, title in zip(
        axes[:2],
        (0.9, 1.0),
        (r"A  Correlated amplitudes ($\rho=0.9$)", r"B  Exact shared amplitudes ($\rho=1$)"),
        strict=True,
    ):
        for k in (4, 8):
            selected = [
                row
                for row in cell_rows
                if np.isclose(float(row["amplitude_correlation"]), rho)
                and row["k"] == k
                and row["sample_count"] == max_n
            ]
            x = np.asarray([int(row["support_count"]) for row in selected])
            y = np.asarray([float(row["gride_over_source_mean"]) for row in selected])
            low = np.asarray([float(row["gride_over_source_ci_low"]) for row in selected])
            high = np.asarray([float(row["gride_over_source_ci_high"]) for row in selected])
            axis.errorbar(
                x,
                y,
                yerr=np.vstack((y - low, high - y)),
                color=colors[k],
                marker="o",
                linewidth=1.8,
                capsize=2.5,
                label=f"k={k}",
            )
        axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        axis.set_xscale("log", base=2)
        axis.set_xticks((1, 16, 256, 4096), ("1", "16", "256", "4096"))
        axis.set_xlabel("number of pooled exact supports B")
        axis.set_ylabel("rank-2 GRIDE ID / source rank")
        axis.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)

    axis = axes[2]
    for rho, linestyle in ((0.9, "-"), (1.0, "--")):
        for k in (4, 8):
            selected = [
                row
                for row in cell_rows
                if np.isclose(float(row["amplitude_correlation"]), rho)
                and row["k"] == k
                and row["sample_count"] == max_n
            ]
            x = np.asarray([int(row["support_count"]) for row in selected])
            y = np.asarray([float(row["same_support_fraction_mean"]) for row in selected])
            axis.plot(
                x,
                y,
                color=colors[k],
                linestyle=linestyle,
                marker="o",
                linewidth=1.7,
                label=fr"k={k}, $\rho$={rho:g}",
            )
    axis.set_xscale("log", base=2)
    axis.set_xticks((1, 16, 256, 4096), ("1", "16", "256", "4096"))
    axis.set_ylim(-0.04, 1.04)
    axis.set_xlabel("number of pooled exact supports B")
    axis.set_ylabel("same-support fraction among 2-NN")
    axis.set_title("C  Why pooled ID rises", loc="left", fontsize=10.5, fontweight="bold")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"Structured support-pool control at N={max_n:,}", fontsize=12.5)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=190)
    plt.close(figure)
    print(f"Aggregated {len(summary_rows)} measurements and wrote {args.figure}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--k", type=int, required=True, choices=(4, 8))
    run_parser.add_argument("--rho", type=float, required=True, choices=(0.9, 1.0))
    run_parser.add_argument("--dim", type=int, default=32)
    run_parser.add_argument("--features", type=int, default=256)
    run_parser.add_argument("--modules", type=int, default=8)
    run_parser.add_argument("--b-values", default="1,16,256,4096")
    run_parser.add_argument("--n-values", default="4096,16384")
    run_parser.add_argument("--repeats", type=int, default=3)
    run_parser.add_argument("--module-bias", type=float, default=4.0)
    run_parser.add_argument("--zipf-alpha", type=float, default=1.0)
    run_parser.add_argument("--coherence", type=float, default=0.75)
    run_parser.add_argument("--amplitude-log-scale", type=float, default=0.35)
    run_parser.add_argument("--gride-range-max", type=int, default=64)
    run_parser.add_argument("--gride-jobs", type=int, default=1)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.set_defaults(function=run_task)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--input", type=Path, required=True)
    analyze_parser.add_argument(
        "--figure", type=Path, default=Path("docs/assets/main-structured-support-pool.png")
    )
    analyze_parser.set_defaults(function=analyze)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

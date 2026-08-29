"""Run a controlled DADApy Binary Intrinsic Dimension audit."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .bid import (
    binary_diagnostics,
    population_center_and_scale,
    sign_binarize,
    support_mask_spins,
    two_bit_quantize,
)
from .config import BidAuditConfig
from .generator import make_feature_matrix, make_support_bank
from .metrics import GrideProfile, measure_gride


@dataclass(frozen=True)
class BidAuditArtifacts:
    output_dir: Path
    fit_path: Path
    histogram_path: Path
    profile_path: Path
    summary_path: Path
    verdict_path: Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _failed_profile(range_max: int) -> GrideProfile:
    ranks = 2 ** np.arange(1, int(np.floor(np.log2(range_max))) + 1)
    missing = np.full(len(ranks), np.nan)
    return GrideProfile(ranks, missing.copy(), missing.copy(), missing.copy())


def _measure_gride_safely(
    representations: np.ndarray,
    range_max: int,
    n_jobs: int,
    support_ids: np.ndarray,
) -> tuple[GrideProfile, str]:
    try:
        profile = measure_gride(
            representations, range_max, n_jobs=n_jobs, support_ids=support_ids
        )
        status = "ok" if np.all(np.isfinite(profile.ids)) else "invalid_nonfinite"
        return profile, status
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return _failed_profile(range_max), f"failed:{type(error).__name__}"


def _quantile_bin_count(hamming: object, alpha_max: float) -> tuple[int, int]:
    hamming.set_r_quantile(alpha_max)
    bin_count = int(hamming.r_idx) + 1
    r_max = int(hamming.r)
    hamming.r = None
    hamming.r_idx = None
    return bin_count, r_max


def _fit_binary_view(
    spins: np.ndarray,
    *,
    config: BidAuditConfig,
    repeat: int,
    k: int,
    support_pool_size: int,
    view: str,
    target_interpretation: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    # Keep JAX and its optional backend out of ordinary imports and unit tests.
    from dadapy.hamming import BID, Hamming

    diagnostics = binary_diagnostics(spins)
    hamming = Hamming(coordinates=spins, verbose=False)
    hamming.compute_distances()
    hamming.D_histogram(compute_flag=1)
    # Fits use only the histogram. Releasing the dense N x N array keeps the
    # paper's all-pairs method feasible as we move across binary views.
    hamming.distances = None

    fit_rows: list[dict[str, object]] = []
    histogram_rows: list[dict[str, object]] = []
    for alpha_index, alpha_max in enumerate(config.alpha_max_values):
        bin_count, prefit_r_max = _quantile_bin_count(hamming, alpha_max)
        common = {
            "experiment": "bid-audit",
            "condition": f"view={view};k={k};B={support_pool_size}",
            "repeat": repeat,
            "k": k,
            "support_pool_size": support_pool_size,
            "sample_count": config.sample_count,
            "binary_view": view,
            "target_interpretation": target_interpretation,
            "bit_count": diagnostics.bit_count,
            "alpha_max": alpha_max,
            "fit_bin_count": bin_count,
            "mean_absolute_bit_imbalance": diagnostics.mean_absolute_imbalance,
            "maximum_absolute_bit_imbalance": diagnostics.maximum_absolute_imbalance,
            "constant_bit_fraction": diagnostics.constant_bit_fraction,
            "unique_pattern_count": diagnostics.unique_pattern_count,
        }
        if bin_count < 3:
            fit_rows.append(
                {
                    **common,
                    "r_max": prefit_r_max,
                    "bid": float("nan"),
                    "d1": float("nan"),
                    "log_kl": float("nan"),
                    "kl": float("nan"),
                    "total_variation": float("nan"),
                    "max_probability_error": float("nan"),
                    "accepted_move_fraction": float("nan"),
                    "optimization_minutes": 0.0,
                    "fit_status": "insufficient_histogram_bins",
                }
            )
            continue

        try:
            estimator = BID(
                H=hamming,
                alphamin=0.0,
                alphamax=alpha_max,
                seed=int(
                    np.random.SeedSequence(
                        [config.seed, repeat, k, support_pool_size, alpha_index, 907]
                    ).generate_state(1)[0]
                ),
                delta=config.bid_delta,
                Nsteps=config.bid_steps,
                export_results=0,
                export_logKLs=0,
                L=diagnostics.bit_count,
            )
            estimator.computeBID()
            empirical = np.asarray(estimator.Pemp, dtype=np.float64)
            model = np.asarray(estimator.Pmodel, dtype=np.float64)
            finite = bool(
                np.isfinite(estimator.d0)
                and np.isfinite(estimator.d1)
                and np.all(np.isfinite(model))
            )
            status = "ok" if finite else "invalid_nonfinite"
            total_variation = float(0.5 * np.sum(np.abs(empirical - model)))
            max_probability_error = float(np.max(np.abs(empirical - model)))
            fit_rows.append(
                {
                    **common,
                    "r_max": int(estimator.rmax),
                    "bid": float(estimator.d0),
                    "d1": float(estimator.d1),
                    "log_kl": float(estimator.logKL),
                    "kl": float(np.exp(estimator.logKL)),
                    "total_variation": total_variation,
                    "max_probability_error": max_probability_error,
                    "accepted_move_fraction": float(estimator.Op.acc_ratio),
                    "optimization_minutes": float(estimator.optimization_elapsed_time),
                    "fit_status": status,
                }
            )
            for distance, probability, predicted in zip(
                estimator.remp, empirical, model, strict=True
            ):
                histogram_rows.append(
                    {
                        "experiment": "bid-audit",
                        "repeat": repeat,
                        "k": k,
                        "support_pool_size": support_pool_size,
                        "binary_view": view,
                        "alpha_max": alpha_max,
                        "hamming_distance": float(distance),
                        "empirical_probability": float(probability),
                        "model_probability": float(predicted),
                    }
                )
        except (AssertionError, FloatingPointError, RuntimeError, ValueError) as error:
            fit_rows.append(
                {
                    **common,
                    "r_max": prefit_r_max,
                    "bid": float("nan"),
                    "d1": float("nan"),
                    "log_kl": float("nan"),
                    "kl": float("nan"),
                    "total_variation": float("nan"),
                    "max_probability_error": float("nan"),
                    "accepted_move_fraction": float("nan"),
                    "optimization_minutes": float("nan"),
                    "fit_status": f"failed:{type(error).__name__}",
                }
            )
    return fit_rows, histogram_rows


def _sample_balanced_pool(
    matrix: np.ndarray,
    supports: np.ndarray,
    sample_count: int,
    *,
    seed: int,
    repeat: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_support = sample_count // len(supports)
    representations: list[np.ndarray] = []
    amplitudes: list[np.ndarray] = []
    support_ids: list[np.ndarray] = []
    for support_index, support in enumerate(supports):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, repeat, k, support_index, 401])
        )
        local_amplitudes = rng.uniform(0.5, 1.5, size=(per_support, k))
        amplitudes.append(local_amplitudes)
        representations.append(local_amplitudes @ matrix[:, support].T)
        support_ids.append(np.full(per_support, support_index, dtype=np.int_))
    return (
        np.concatenate(representations),
        np.concatenate(amplitudes),
        np.concatenate(support_ids),
    )


def _primary_bid(
    rows: list[dict[str, object]], view: str, primary_alpha: float
) -> float:
    matches = [
        row
        for row in rows
        if row["binary_view"] == view
        and np.isclose(float(row["alpha_max"]), primary_alpha)
        and row["fit_status"] == "ok"
    ]
    return float(matches[0]["bid"]) if len(matches) == 1 else float("nan")


def _aggregate(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, object]] = []
    for values, group in groups.items():
        result = dict(zip(keys, values, strict=True))
        for field, value in group[0].items():
            if field in keys or not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            numeric = np.asarray([float(row[field]) for row in group])
            result[field] = (
                float(np.nanmean(numeric)) if not np.all(np.isnan(numeric)) else float("nan")
            )
        output.append(result)
    return output


def _interpret(
    config: BidAuditConfig,
    fit_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> str:
    summary_means = _aggregate(summary_rows, ("k", "support_pool_size"))
    tolerance = config.id_relative_tolerance

    def values_for(field: str, pool_size: int = 1) -> list[tuple[int, float]]:
        return sorted(
            (int(row["k"]), float(row[field]))
            for row in summary_means
            if int(row["support_pool_size"]) == pool_size
        )

    independent = values_for("independent_bits_bid")
    latent = values_for("latent_sign_bid")
    fixed_sign = values_for("activation_sign_bid")
    fixed_two = values_for("activation_two_bit_bid")
    fixed_gride = values_for("gride_rank_2")

    def all_within(cells: list[tuple[int, float]]) -> bool:
        return bool(cells) and all(
            np.isfinite(value) and abs(value - k) / k <= tolerance for k, value in cells
        )

    independent_pass = all_within(independent)
    latent_pass = all_within(latent)
    sign_pass = all_within(fixed_sign)
    two_pass = all_within(fixed_two)

    def mean_relative_error(cells: list[tuple[int, float]]) -> float:
        errors = [abs(value - k) / k for k, value in cells if np.isfinite(value)]
        return float(np.mean(errors)) if errors else float("nan")

    gride_error = mean_relative_error(fixed_gride)
    sign_error = mean_relative_error(fixed_sign)
    two_error = mean_relative_error(fixed_two)

    lines = [
        "# Binary intrinsic-dimension audit",
        "",
        "For B=1, the continuous samples lie in a full-rank k-dimensional parallelepiped, so their population continuous intrinsic dimension is exactly k. BID instead estimates the dimension of the chosen binary code; equality with k is tested here rather than assumed.",
        "",
        f"- Independent k-bit calibration: **{'PASS' if independent_pass else 'NOT ESTABLISHED'}**.",
        f"- Sign-binarized latent amplitudes: **{'PASS' if latent_pass else 'NOT ESTABLISHED'}**.",
        f"- Sign-binarized activation coordinates recover k: **{'PASS' if sign_pass else 'NOT ESTABLISHED'}**.",
        f"- Paper-style two-bit activation coordinates recover k: **{'PASS' if two_pass else 'NOT ESTABLISHED'}**.",
        "",
        f"At fixed support, mean relative error is GRIDE={gride_error:.1%}, one-bit BID={sign_error:.1%}, and two-bit BID={two_error:.1%}. Activation BID uses the prespecified local alpha_max={config.primary_alpha_max:g}; exact independent-bit controls use the complete histogram (alpha_max=1).",
        "",
        "## Primary estimates",
        "",
        "| k | B | GRIDE rank 2 | independent bits | latent sign | activation sign BID | activation two-bit BID | support-mask BID |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_means, key=lambda item: (int(item["k"]), int(item["support_pool_size"]))):
        lines.append(
            f"| {int(row['k'])} | {int(row['support_pool_size'])} | "
            f"{float(row['gride_rank_2']):.3f} | {float(row['independent_bits_bid']):.3f} | "
            f"{float(row['latent_sign_bid']):.3f} | {float(row['activation_sign_bid']):.3f} | "
            f"{float(row['activation_two_bit_bid']):.3f} | {float(row['support_mask_bid']):.3f} |"
        )

    valid_fits = [row for row in fit_rows if row["fit_status"] == "ok"]
    fit_means = _aggregate(
        valid_fits, ("binary_view", "support_pool_size", "alpha_max")
    )
    lines.extend(
        [
            "",
            "## Quantile and model-fit audit",
            "",
            "| binary view | B | alpha_max | mean BID / k | mean TV fit error | valid fits |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        fit_means,
        key=lambda item: (
            str(item["binary_view"]),
            int(item["support_pool_size"]),
            float(item["alpha_max"]),
        ),
    ):
        view = str(row["binary_view"])
        pool_size = int(row["support_pool_size"])
        alpha = float(row["alpha_max"])
        group = [
            fit
            for fit in valid_fits
            if fit["binary_view"] == view
            and int(fit["support_pool_size"]) == pool_size
            and np.isclose(float(fit["alpha_max"]), alpha)
        ]
        ratios = [float(fit["bid"]) / int(fit["k"]) for fit in group]
        lines.append(
            f"| {view} | {pool_size} | {alpha:g} | {np.mean(ratios):.3f} | "
            f"{float(row['total_variation']):.4f} | {len(group)} |"
        )

    missing_count = len(fit_rows) - len(valid_fits)
    better = sign_pass and sign_error < gride_error
    lines.extend(
        [
            "",
            f"There were {missing_count} non-valid fits, including deliberately retained low-quantile fits with fewer than three populated distance bins. These are not silently averaged.",
            "",
            f"**Does BID provide a better estimate of continuous k here? {'YES' if better else 'NO'}.** "
            "The independent-bit and latent-sign controls test the implementation. Activation-sign and two-bit results test the scientifically relevant but stronger claim that binarizing h preserves its continuous dimension. Pooled-support and support-mask results are global binary-complexity diagnostics and are not proofs about conditional k.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_bid_audit_experiment(
    config: BidAuditConfig, output_dir: Path, *, show_progress: bool = True
) -> BidAuditArtifacts:
    """Run fixed- and pooled-support BID comparisons and save inspectable fits."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fit_rows: list[dict[str, object]] = []
    histogram_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    # One independent-bit histogram plus three binary views for each B. The
    # fixed pool substitutes latent-sign for the otherwise constant support mask.
    total = config.repeats * len(config.active_features) * (
        1 + 3 * len(config.support_pool_sizes)
    )
    progress = tqdm(
        total=total,
        desc="bid-audit: complete measurements",
        unit="measurement",
        dynamic_ncols=True,
        disable=not show_progress or not sys.stderr.isatty(),
    )
    completed = 0

    def mark_complete(label: str) -> None:
        nonlocal completed
        completed += 1
        progress.update(1)
        if show_progress and not sys.stderr.isatty():
            print(f"[{completed}/{total}] bid-audit complete: {label}", flush=True)

    try:
        for repeat in range(config.repeats):
            matrix_rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, repeat, 701])
            )
            matrix = make_feature_matrix(
                config.representation_dim, config.feature_count, 0.0, matrix_rng
            )
            for k in config.active_features:
                largest_pool = max(config.support_pool_sizes)
                support_rng = np.random.default_rng(
                    np.random.SeedSequence([config.seed, repeat, k, 307])
                )
                support_bank = make_support_bank(
                    config.feature_count, k, largest_pool, support_rng
                )

                independent_rng = np.random.default_rng(
                    np.random.SeedSequence([config.seed, repeat, k, 809])
                )
                independent_spins = independent_rng.choice(
                    np.array([-1, 1], dtype=np.int8), size=(config.sample_count, k)
                )
                local_fits, local_histograms = _fit_binary_view(
                    independent_spins,
                    config=config,
                    repeat=repeat,
                    k=k,
                    support_pool_size=0,
                    view="independent-bits",
                    target_interpretation="BID is exactly k in the population",
                )
                fit_rows.extend(local_fits)
                histogram_rows.extend(local_histograms)
                independent_bid = _primary_bid(local_fits, "independent-bits", 1.0)
                mark_complete(f"independent-bits;k={k};repeat={repeat}")

                for pool_size in config.support_pool_sizes:
                    supports = support_bank[:pool_size]
                    representations, amplitudes, support_ids = _sample_balanced_pool(
                        matrix,
                        supports,
                        config.sample_count,
                        seed=config.seed,
                        repeat=repeat,
                        k=k,
                    )
                    center, scale = population_center_and_scale(matrix, supports)
                    centered = representations - center
                    binary_views: list[tuple[str, np.ndarray, str]] = [
                        (
                            "activation-sign",
                            sign_binarize(centered),
                            "tested against k; equality is not guaranteed by BID theory",
                        ),
                        (
                            "activation-two-bit",
                            two_bit_quantize(centered, scale),
                            "paper-style quantization; equality with k is not guaranteed",
                        ),
                    ]
                    if pool_size == 1:
                        binary_views.append(
                            (
                                "latent-sign",
                                sign_binarize(amplitudes - 1.0),
                                "BID is exactly k in the population",
                            )
                        )
                    else:
                        binary_views.append(
                            (
                                "support-mask",
                                support_mask_spins(
                                    support_ids, supports, config.feature_count
                                ),
                                "support combinatorics, not continuous conditional k",
                            )
                        )

                    condition_fit_rows: list[dict[str, object]] = []
                    for view, spins, interpretation in binary_views:
                        local_fits, local_histograms = _fit_binary_view(
                            spins,
                            config=config,
                            repeat=repeat,
                            k=k,
                            support_pool_size=pool_size,
                            view=view,
                            target_interpretation=interpretation,
                        )
                        fit_rows.extend(local_fits)
                        condition_fit_rows.extend(local_fits)
                        histogram_rows.extend(local_histograms)
                        mark_complete(f"{view};k={k};B={pool_size};repeat={repeat}")

                    profile, gride_status = _measure_gride_safely(
                        representations,
                        config.gride_range_max,
                        config.gride_n_jobs,
                        support_ids,
                    )
                    for rank_index, (rank, physical_scale, estimate, error) in enumerate(
                        zip(
                            profile.ranks,
                            profile.scales,
                            profile.ids,
                            profile.errors,
                            strict=True,
                        )
                    ):
                        profile_rows.append(
                            {
                                "experiment": "bid-audit",
                                "condition": f"k={k};B={pool_size}",
                                "repeat": repeat,
                                "k": k,
                                "support_pool_size": pool_size,
                                "sample_count": config.sample_count,
                                "rank": int(rank),
                                "scale": float(physical_scale),
                                "gride_id": float(estimate),
                                "gride_error": float(error),
                                "relative_error_to_k": abs(float(estimate) - k) / k,
                                "mean_same_support_fraction": (
                                    float(profile.mean_same_support_fraction[rank_index])
                                    if profile.mean_same_support_fraction is not None
                                    else float("nan")
                                ),
                                "all_same_support_fraction": (
                                    float(profile.all_same_support_fraction[rank_index])
                                    if profile.all_same_support_fraction is not None
                                    else float("nan")
                                ),
                                "measurement_status": gride_status,
                            }
                        )

                    summary_rows.append(
                        {
                            "experiment": "bid-audit",
                            "condition": f"k={k};B={pool_size}",
                            "repeat": repeat,
                            "representation_dim": config.representation_dim,
                            "feature_count": config.feature_count,
                            "k": k,
                            "support_pool_size": pool_size,
                            "sample_count": config.sample_count,
                            "samples_per_support": config.sample_count // pool_size,
                            "population_continuous_id": k,
                            "global_binary_scale": scale,
                            "gride_rank_2": float(profile.ids[0]),
                            "gride_rank_2_relative_error": abs(float(profile.ids[0]) - k) / k,
                            "independent_bits_bid": independent_bid if pool_size == 1 else float("nan"),
                            "latent_sign_bid": _primary_bid(
                                condition_fit_rows, "latent-sign", 1.0
                            ),
                            "activation_sign_bid": _primary_bid(
                                condition_fit_rows,
                                "activation-sign",
                                config.primary_alpha_max,
                            ),
                            "activation_two_bit_bid": _primary_bid(
                                condition_fit_rows,
                                "activation-two-bit",
                                config.primary_alpha_max,
                            ),
                            "support_mask_bid": _primary_bid(
                                condition_fit_rows,
                                "support-mask",
                                1.0,
                            ),
                            "rank_2_same_support_fraction": (
                                float(profile.mean_same_support_fraction[0])
                                if profile.mean_same_support_fraction is not None
                                else float("nan")
                            ),
                            "rank_2_all_same_support_fraction": (
                                float(profile.all_same_support_fraction[0])
                                if profile.all_same_support_fraction is not None
                                else float("nan")
                            ),
                            "measurement_status": gride_status,
                        }
                    )
    finally:
        progress.close()

    fit_path = output_dir / "bid_fits.csv"
    histogram_path = output_dir / "bid_histograms.csv"
    profile_path = output_dir / "gride_profiles.csv"
    summary_path = output_dir / "summary.csv"
    verdict_path = output_dir / "interpretation.md"
    _write_csv(fit_path, fit_rows)
    _write_csv(histogram_path, histogram_rows)
    _write_csv(profile_path, profile_rows)
    _write_csv(summary_path, summary_rows)
    verdict_path.write_text(
        _interpret(config, fit_rows, summary_rows), encoding="utf-8"
    )
    return BidAuditArtifacts(
        output_dir,
        fit_path,
        histogram_path,
        profile_path,
        summary_path,
        verdict_path,
    )

"""Run and interpret the boundaryless sparse-feature geometry audit."""

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

from .config import ManifoldGeometryConfig
from .manifold import (
    LocalGeometry,
    combined_representation,
    feature_branch,
    local_geometry_from_singular_values,
    make_residual_sparse_map,
    nearest_neighbor_support_diagnostics,
    sample_unit_sphere,
    sparse_features,
    sparse_tangent_singular_values,
    threshold_for_expected_activity,
)
from .metrics import GrideProfile, measure_gride


@dataclass(frozen=True)
class ManifoldArtifacts:
    output_dir: Path
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
    representations: np.ndarray, range_max: int, n_jobs: int
) -> tuple[GrideProfile, str]:
    try:
        profile = measure_gride(representations, range_max, n_jobs)
        status = "ok" if np.all(np.isfinite(profile.ids)) else "invalid_nonfinite"
        return profile, status
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return _failed_profile(range_max), f"failed:{type(error).__name__}"


def _condition_label(
    representation_kind: str,
    intrinsic_dim: int,
    sample_count: int,
    expected_active: int,
    feature_strength: float,
) -> str:
    return (
        f"kind={representation_kind};d={intrinsic_dim};N={sample_count};"
        f"s={expected_active};gamma={feature_strength:g}"
    )


def _baseline_geometry() -> LocalGeometry:
    return LocalGeometry(1.0, 1.0, 1.0, 0.0)


def _append_measurement(
    profile_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    *,
    repeat: int,
    representation_kind: str,
    intrinsic_dim: int,
    sample_count: int,
    expected_active: int,
    feature_strength: float,
    threshold: float,
    representations: np.ndarray,
    latent_points: np.ndarray,
    active_sets: np.ndarray | None,
    feature_rms: float,
    geometry: LocalGeometry,
    profile: GrideProfile,
    measurement_status: str,
    population_id_known: bool,
    isometry_error: float,
) -> None:
    label = _condition_label(
        representation_kind, intrinsic_dim, sample_count, expected_active, feature_strength
    )
    for rank, scale, estimate, error in zip(
        profile.ranks, profile.scales, profile.ids, profile.errors, strict=True
    ):
        profile_rows.append(
            {
                "experiment": "manifold-geometry",
                "condition": label,
                "repeat": repeat,
                "representation_kind": representation_kind,
                "representation_dim": representations.shape[1],
                "feature_count": 0 if active_sets is None else active_sets.shape[1],
                "intrinsic_dim": intrinsic_dim,
                "population_id_known": int(population_id_known),
                "sample_count": sample_count,
                "expected_active": expected_active,
                "feature_strength": feature_strength,
                "threshold": threshold,
                "rank": int(rank),
                "scale": float(scale),
                "gride_id": float(estimate),
                "gride_error": float(error),
                "relative_error_to_true_id": (
                    abs(float(estimate) - intrinsic_dim) / intrinsic_dim
                    if population_id_known
                    else float("nan")
                ),
                "measurement_status": measurement_status,
            }
        )

    mean_jaccard = float("nan")
    exact_support = float("nan")
    mean_latent_distance = float("nan")
    mean_representation_distance = float("nan")
    if profile.neighbor_indices is not None:
        nearest = profile.neighbor_indices[:, 1]
        mean_latent_distance = float(
            np.linalg.norm(latent_points - latent_points[nearest], axis=1).mean()
        )
        mean_representation_distance = float(
            np.linalg.norm(representations - representations[nearest], axis=1).mean()
        )
        if active_sets is not None:
            mean_jaccard, exact_support = nearest_neighbor_support_diagnostics(
                active_sets, profile.neighbor_indices
            )

    active_counts = (
        active_sets.sum(axis=1) if active_sets is not None else np.zeros(sample_count, dtype=int)
    )
    rank_2_id = float(profile.ids[0])
    summary_rows.append(
        {
            "experiment": "manifold-geometry",
            "condition": label,
            "repeat": repeat,
            "representation_kind": representation_kind,
            "representation_dim": representations.shape[1],
            "feature_count": 0 if active_sets is None else active_sets.shape[1],
            "intrinsic_dim": intrinsic_dim,
            "population_id_known": int(population_id_known),
            "sample_count": sample_count,
            "expected_active": expected_active,
            "realized_mean_active": float(active_counts.mean()),
            "realized_std_active": float(active_counts.std(ddof=0)),
            "zero_active_fraction": float(np.mean(active_counts == 0)) if active_sets is not None else 0.0,
            "feature_strength": feature_strength,
            "threshold": threshold,
            "feature_rms_before_normalization": feature_rms,
            "local_full_rank_fraction": geometry.full_rank_fraction,
            "mean_local_condition_number": geometry.mean_condition_number,
            "p95_local_condition_number": geometry.p95_condition_number,
            "local_log_volume_std": geometry.log_volume_std,
            "mean_1nn_active_jaccard": mean_jaccard,
            "exact_1nn_support_match_fraction": exact_support,
            "mean_latent_1nn_chord": mean_latent_distance,
            "mean_representation_1nn_distance": mean_representation_distance,
            "max_isometry_distance_error": isometry_error,
            "local_gride_id_rank_2": rank_2_id,
            "rank_2_relative_error_to_true_id": (
                abs(rank_2_id - intrinsic_dim) / intrinsic_dim
                if population_id_known
                else float("nan")
            ),
            "measurement_status": measurement_status,
        }
    )


def _aggregate(
    rows: list[dict[str, object]], keys: tuple[str, ...]
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    aggregated: list[dict[str, object]] = []
    for key_values, group in groups.items():
        result = dict(zip(keys, key_values, strict=True))
        numeric_keys = [
            key
            for key, value in group[0].items()
            if key not in keys
            and key not in {"experiment", "condition", "representation_kind", "measurement_status"}
            and isinstance(value, (int, float, np.integer, np.floating))
        ]
        for numeric_key in numeric_keys:
            values = np.asarray([float(row[numeric_key]) for row in group])
            result[numeric_key] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
        aggregated.append(result)
    return aggregated


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.ptp(x[finite]) == 0.0 or np.ptp(y[finite]) == 0.0:
        return float("nan")
    return float(spearmanr(x[finite], y[finite]).statistic)


def _interpret(
    config: ManifoldGeometryConfig,
    summary_rows: list[dict[str, object]],
    profile_rows: list[dict[str, object]],
) -> str:
    summary_means = _aggregate(
        summary_rows,
        (
            "representation_kind",
            "intrinsic_dim",
            "sample_count",
            "expected_active",
            "feature_strength",
        ),
    )
    baseline = sorted(
        (row for row in summary_means if row["representation_kind"] == "sphere"),
        key=lambda row: (int(row["intrinsic_dim"]), int(row["sample_count"])),
    )
    residual = [row for row in summary_means if row["representation_kind"] == "residual"]
    combined = [row for row in summary_means if row["representation_kind"] == "combined"]
    sparse = [row for row in summary_means if row["representation_kind"] == "sparse"]

    tolerance = config.id_relative_tolerance
    baseline_pass = bool(baseline) and all(
        float(row["rank_2_relative_error_to_true_id"]) <= tolerance for row in baseline
    )
    isometry_pass = bool(residual) and max(
        float(row["max_isometry_distance_error"]) for row in residual
    ) <= 1e-12
    mild = [row for row in combined if float(row["feature_strength"]) <= 1.0]
    mild_pass = bool(mild) and all(
        float(row["rank_2_relative_error_to_true_id"]) <= tolerance for row in mild
    )

    errors = np.asarray([float(row["rank_2_relative_error_to_true_id"]) for row in combined])
    radii = np.asarray([float(row["mean_latent_1nn_chord"]) for row in combined])
    condition_numbers = np.asarray(
        [float(row["mean_local_condition_number"]) for row in combined]
    )
    volume_variation = np.asarray([float(row["local_log_volume_std"]) for row in combined])
    support_change = np.asarray([1.0 - float(row["mean_1nn_active_jaccard"]) for row in combined])

    profile_means = _aggregate(
        profile_rows,
        (
            "representation_kind",
            "intrinsic_dim",
            "sample_count",
            "expected_active",
            "feature_strength",
            "rank",
        ),
    )
    known_profiles = [
        row
        for row in profile_means
        if row["representation_kind"] in {"sphere", "residual", "combined"}
    ]
    accurate_profile_fraction = float(
        np.mean(
            [
                float(row["relative_error_to_true_id"]) <= tolerance
                for row in known_profiles
                if np.isfinite(float(row["relative_error_to_true_id"]))
            ]
        )
    )

    lines = [
        "# Boundaryless sparse-feature geometry audit",
        "",
        "The sphere and combined representations have population intrinsic dimension exactly d. "
        "The sparse-only branch is diagnostic and is not scored against d because its ReLU map can be locally rank-deficient or atomic.",
        "",
        f"- Uniform sphere rank-2 calibration: **{'PASS' if baseline_pass else 'NOT ESTABLISHED'}** "
        f"(all mean estimates within {tolerance:.0%} of d).",
        f"- Isometric residual implementation: **{'PASS' if isometry_pass else 'NOT ESTABLISHED'}** "
        "(maximum sampled distance error <= 1e-12).",
        f"- Combined map at feature strength gamma <= 1: **{'PASS' if mild_pass else 'NOT ESTABLISHED'}** "
        f"(all mean rank-2 estimates within {tolerance:.0%} of d).",
        f"- Across all known-ID representations and saved ranks, {accurate_profile_fraction:.1%} of mean profile cells are within tolerance.",
        "",
        "## Boundaryless baseline",
        "",
        "| d | N | mean rank-2 ID | ID / d | mean latent 1-NN chord |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in baseline:
        lines.append(
            f"| {int(row['intrinsic_dim'])} | {int(row['sample_count'])} | "
            f"{float(row['local_gride_id_rank_2']):.3f} | "
            f"{float(row['local_gride_id_rank_2']) / float(row['intrinsic_dim']):.3f} | "
            f"{float(row['mean_latent_1nn_chord']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Combined-map recovery grouped by feature strength",
            "",
            "| gamma | accurate cells | median ID / d | mean condition number | mean log-volume std | mean 1-NN Jaccard |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strength in sorted({float(row["feature_strength"]) for row in combined}):
        group = [row for row in combined if float(row["feature_strength"]) == strength]
        accurate = np.mean(
            [float(row["rank_2_relative_error_to_true_id"]) <= tolerance for row in group]
        )
        ratios = [float(row["local_gride_id_rank_2"]) / float(row["intrinsic_dim"]) for row in group]
        lines.append(
            f"| {strength:g} | {accurate:.1%} | {np.median(ratios):.3f} | "
            f"{np.mean([float(row['mean_local_condition_number']) for row in group]):.3f} | "
            f"{np.mean([float(row['local_log_volume_std']) for row in group]):.3f} | "
            f"{np.mean([float(row['mean_1nn_active_jaccard']) for row in group]):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Mechanism diagnostics across combined conditions",
            "",
            f"- Spearman(relative error, latent 1-NN chord): {_safe_spearman(errors, radii):.3f}.",
            f"- Spearman(relative error, mean local condition number): {_safe_spearman(errors, condition_numbers):.3f}.",
            f"- Spearman(relative error, local log-volume variation): {_safe_spearman(errors, volume_variation):.3f}.",
            f"- Spearman(relative error, 1 - active-set Jaccard): {_safe_spearman(errors, support_change):.3f}.",
            "",
            "These correlations are descriptive. Causal evidence comes from the crossed N, d, activity, and gamma controls, not from a correlation alone.",
            "",
            "## Sparse-only warning control",
            "",
            "| d | expected active | mean active | zero-active fraction | full-rank fraction | mean rank-2 GRIDE ID |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    sparse_grouped = _aggregate(sparse, ("intrinsic_dim", "expected_active"))
    for row in sorted(sparse_grouped, key=lambda value: (int(value["intrinsic_dim"]), int(value["expected_active"]))):
        lines.append(
            f"| {int(row['intrinsic_dim'])} | {int(row['expected_active'])} | "
            f"{float(row['realized_mean_active']):.2f} | {float(row['zero_active_fraction']):.2%} | "
            f"{float(row['local_full_rank_fraction']):.2%} | "
            f"{float(row['local_gride_id_rank_2']):.3f} |"
        )

    conclusion = baseline_pass and isometry_pass and mild_pass
    lines.extend(
        [
            "",
            f"**Overall controlled conclusion: {'PASS' if conclusion else 'NOT ESTABLISHED'}.** "
            "A PASS says GRIDE is calibrated for the boundaryless baseline and remains calibrated under the tested mild sparse deformation. "
            "It does not validate arbitrary LLM activation geometry.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_manifold_geometry_experiment(
    config: ManifoldGeometryConfig, output_dir: Path, *, show_progress: bool = True
) -> ManifoldArtifacts:
    """Run the crossed geometry experiment and save full profiles and diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    profile_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    conditions_per_repeat = sum(
        len(config.sample_counts)
        * (2 + len(config.expected_activities(d)) * (1 + len(config.feature_strengths)))
        for d in config.intrinsic_dims
    )
    total = config.repeats * conditions_per_repeat
    progress = tqdm(
        total=total,
        desc="manifold-geometry: complete measurements",
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
            print(f"[{completed}/{total}] manifold-geometry complete: {label}", flush=True)

    try:
        for repeat in range(config.repeats):
            for intrinsic_dim in config.intrinsic_dims:
                map_rng = np.random.default_rng(
                    np.random.SeedSequence([config.seed, repeat, intrinsic_dim, 101])
                )
                point_rng = np.random.default_rng(
                    np.random.SeedSequence([config.seed, repeat, intrinsic_dim, 103])
                )
                mapping = make_residual_sparse_map(
                    config.representation_dim, config.feature_count, intrinsic_dim, map_rng
                )
                all_points = sample_unit_sphere(intrinsic_dim, max(config.sample_counts), point_rng)
                all_residual = all_points @ mapping.residual_matrix.T
                distance_pairs = min(512, len(all_points) - 1)
                latent_pair_distances = np.linalg.norm(
                    all_points[:distance_pairs] - all_points[1 : distance_pairs + 1], axis=1
                )
                residual_pair_distances = np.linalg.norm(
                    all_residual[:distance_pairs] - all_residual[1 : distance_pairs + 1], axis=1
                )
                isometry_error = float(
                    np.max(np.abs(latent_pair_distances - residual_pair_distances))
                )

                for sample_count in config.sample_counts:
                    points = all_points[:sample_count]
                    sphere_profile, status = _measure_gride_safely(
                        points, config.gride_range_max, config.gride_n_jobs
                    )
                    _append_measurement(
                        profile_rows,
                        summary_rows,
                        repeat=repeat,
                        representation_kind="sphere",
                        intrinsic_dim=intrinsic_dim,
                        sample_count=sample_count,
                        expected_active=0,
                        feature_strength=0.0,
                        threshold=float("nan"),
                        representations=points,
                        latent_points=points,
                        active_sets=None,
                        feature_rms=float("nan"),
                        geometry=_baseline_geometry(),
                        profile=sphere_profile,
                        measurement_status=status,
                        population_id_known=True,
                        isometry_error=0.0,
                    )
                    mark_complete(_condition_label("sphere", intrinsic_dim, sample_count, 0, 0.0))

                    # Q is an exact isometry, so its GRIDE profile and neighbors are
                    # mathematically identical. Reuse the measured distances after an
                    # explicit sampled-distance check instead of recomputing kNN.
                    _append_measurement(
                        profile_rows,
                        summary_rows,
                        repeat=repeat,
                        representation_kind="residual",
                        intrinsic_dim=intrinsic_dim,
                        sample_count=sample_count,
                        expected_active=0,
                        feature_strength=0.0,
                        threshold=float("nan"),
                        representations=all_residual[:sample_count],
                        latent_points=points,
                        active_sets=None,
                        feature_rms=float("nan"),
                        geometry=_baseline_geometry(),
                        profile=sphere_profile,
                        measurement_status="exact_isometry_reuse",
                        population_id_known=True,
                        isometry_error=isometry_error,
                    )
                    mark_complete(_condition_label("residual", intrinsic_dim, sample_count, 0, 0.0))

                for expected_active in config.expected_activities(intrinsic_dim):
                    threshold = threshold_for_expected_activity(
                        intrinsic_dim, config.feature_count, expected_active
                    )
                    all_codes, all_active = sparse_features(
                        all_points, mapping.feature_directions, threshold
                    )
                    all_feature, feature_rms = feature_branch(
                        all_codes, mapping.feature_dictionary
                    )
                    diagnostic_count = min(config.jacobian_samples, len(all_points))
                    sparse_singular_values = sparse_tangent_singular_values(
                        all_points[:diagnostic_count],
                        all_active[:diagnostic_count],
                        mapping,
                        feature_rms,
                    )
                    sparse_geometry = local_geometry_from_singular_values(
                        sparse_singular_values, 1.0, residual=False
                    )

                    for sample_count in config.sample_counts:
                        points = all_points[:sample_count]
                        active = all_active[:sample_count]
                        sparse_representation = all_feature[:sample_count]
                        sparse_profile, sparse_status = _measure_gride_safely(
                            sparse_representation, config.gride_range_max, config.gride_n_jobs
                        )
                        _append_measurement(
                            profile_rows,
                            summary_rows,
                            repeat=repeat,
                            representation_kind="sparse",
                            intrinsic_dim=intrinsic_dim,
                            sample_count=sample_count,
                            expected_active=expected_active,
                            feature_strength=1.0,
                            threshold=threshold,
                            representations=sparse_representation,
                            latent_points=points,
                            active_sets=active,
                            feature_rms=feature_rms,
                            geometry=sparse_geometry,
                            profile=sparse_profile,
                            measurement_status=sparse_status,
                            population_id_known=False,
                            isometry_error=float("nan"),
                        )
                        mark_complete(
                            _condition_label(
                                "sparse", intrinsic_dim, sample_count, expected_active, 1.0
                            )
                        )

                        for feature_strength in config.feature_strengths:
                            combined = combined_representation(
                                points,
                                mapping,
                                sparse_representation,
                                feature_strength,
                            )
                            combined_geometry = local_geometry_from_singular_values(
                                sparse_singular_values, feature_strength, residual=True
                            )
                            combined_profile, combined_status = _measure_gride_safely(
                                combined, config.gride_range_max, config.gride_n_jobs
                            )
                            _append_measurement(
                                profile_rows,
                                summary_rows,
                                repeat=repeat,
                                representation_kind="combined",
                                intrinsic_dim=intrinsic_dim,
                                sample_count=sample_count,
                                expected_active=expected_active,
                                feature_strength=feature_strength,
                                threshold=threshold,
                                representations=combined,
                                latent_points=points,
                                active_sets=active,
                                feature_rms=feature_rms,
                                geometry=combined_geometry,
                                profile=combined_profile,
                                measurement_status=combined_status,
                                population_id_known=True,
                                isometry_error=float("nan"),
                            )
                            mark_complete(
                                _condition_label(
                                    "combined",
                                    intrinsic_dim,
                                    sample_count,
                                    expected_active,
                                    feature_strength,
                                )
                            )
    finally:
        progress.close()

    profile_path = output_dir / "gride_profiles.csv"
    summary_path = output_dir / "summary.csv"
    verdict_path = output_dir / "interpretation.md"
    _write_csv(profile_path, profile_rows)
    _write_csv(summary_path, summary_rows)
    verdict_path.write_text(
        _interpret(config, summary_rows, profile_rows), encoding="utf-8"
    )
    return ManifoldArtifacts(output_dir, profile_path, summary_path, verdict_path)

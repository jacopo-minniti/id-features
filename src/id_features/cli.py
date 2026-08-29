"""Command-line entry point for the first three controlled hypotheses."""

from __future__ import annotations

import argparse
from pathlib import Path

from .bid_experiment import run_bid_audit_experiment
from .config import BidAuditConfig, ExperimentConfig, ManifoldGeometryConfig
from .experiments import run_experiment
from .manifold_experiment import run_manifold_geometry_experiment
from .plotting import plot_results
from .structured_experiment import StructuredExperimentConfig, run_structured_experiment


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _comma_separated_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item)


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True, help="new or empty results directory")
    parser.add_argument("--dim", type=int, default=32, help="representation dimension D")
    parser.add_argument("--features", type=int, default=128, help="latent feature count m")
    parser.add_argument("--id-samples", type=int, default=4_000)
    parser.add_argument("--train-samples", type=int, default=4_000)
    parser.add_argument("--test-samples", type=int, default=4_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gride-range-max", type=int, default=64)
    parser.add_argument(
        "--gride-jobs", type=int, default=1, help="CPU workers used inside each GRIDE call"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true", help="disable terminal progress display")


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    common = dict(
        representation_dim=args.dim,
        feature_count=args.features,
        n_id_samples=args.id_samples,
        n_train=args.train_samples,
        n_test=args.test_samples,
        repeats=args.repeats,
        gride_range_max=args.gride_range_max,
        gride_n_jobs=args.gride_jobs,
        seed=args.seed,
    )
    if args.command == "load-capacity":
        return ExperimentConfig(
            experiment="load-capacity",
            active_features=_comma_separated_ints(args.k_values),
            capacity_values=_comma_separated_ints(args.m_values),
            **common,
        )
    if args.command == "load-ratio":
        return ExperimentConfig(
            experiment="load-ratio",
            active_features=_comma_separated_ints(args.k_values),
            representation_dims=_comma_separated_ints(args.d_values),
            **common,
        )
    if args.command == "support-pool":
        return ExperimentConfig(
            experiment="support-pool",
            active_features=_comma_separated_ints(args.k_values),
            support_pool_sizes=_comma_separated_ints(args.b_values),
            id_sample_values=_comma_separated_ints(args.n_values),
            **common,
        )
    return ExperimentConfig(
        experiment="geometry-control",
        fixed_k=args.k,
        correlations=_comma_separated_floats(args.rho_values),
        **common,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled sparse-ID experiments for hypotheses 1--3")
    subparsers = parser.add_subparsers(dest="command", required=True)
    capacity = subparsers.add_parser(
        "load-capacity", help="test local ID against k while independently sweeping capacity m"
    )
    _add_common_run_arguments(capacity)
    capacity.add_argument("--k-values", default="1,2,4,8,16")
    capacity.add_argument("--m-values", default="64,128,256")
    ratio = subparsers.add_parser(
        "load-ratio", help="sweep k and D to test accessibility versus local ID / D"
    )
    _add_common_run_arguments(ratio)
    ratio.add_argument("--k-values", default="1,2,4,8,16")
    ratio.add_argument("--d-values", default="16,24,32,48,64")
    support_pool = subparsers.add_parser(
        "support-pool",
        help="cross support-pool size B and sample count N while retaining GRIDE ranks",
    )
    _add_common_run_arguments(support_pool)
    support_pool.add_argument("--k-values", default="1,2,4,8,16")
    support_pool.add_argument("--b-values", default="1,4,16,64")
    support_pool.add_argument("--n-values", default="1024,2048,4096")
    geometry = subparsers.add_parser("geometry-control", help="hold k fixed and sweep correlation rho")
    _add_common_run_arguments(geometry)
    geometry.add_argument("--k", type=int, default=8)
    geometry.add_argument("--rho-values", default="0,0.25,0.5,0.75,0.9")
    manifold = subparsers.add_parser(
        "manifold-geometry",
        help="audit GRIDE on a boundaryless known-ID manifold with sparse ReLU features",
    )
    manifold.add_argument("--output", type=Path, required=True)
    manifold.add_argument("--dim", type=int, default=64, help="representation dimension D")
    manifold.add_argument("--features", type=int, default=256, help="sparse feature count m")
    manifold.add_argument("--d-values", default="4,8,16", help="ground-truth intrinsic dimensions")
    manifold.add_argument("--n-values", default="4096,16384", help="nested sample counts")
    manifold.add_argument(
        "--activity-multipliers",
        default="0.5,1,2",
        help="expected active features as multiples of d",
    )
    manifold.add_argument(
        "--feature-strengths",
        default="0.25,1,4",
        help="gamma values for the normalized sparse branch",
    )
    manifold.add_argument("--repeats", type=int, default=3)
    manifold.add_argument("--gride-range-max", type=int, default=64)
    manifold.add_argument("--gride-jobs", type=int, default=1)
    manifold.add_argument("--jacobian-samples", type=int, default=128)
    manifold.add_argument("--seed", type=int, default=0)
    manifold.add_argument("--no-progress", action="store_true")
    bid = subparsers.add_parser(
        "bid-audit",
        help="compare DADApy BID encodings with known continuous k and GRIDE",
    )
    bid.add_argument("--output", type=Path, required=True)
    bid.add_argument("--dim", type=int, default=64, help="representation dimension D")
    bid.add_argument("--features", type=int, default=256, help="latent feature count m")
    bid.add_argument("--k-values", default="2,4,8,16")
    bid.add_argument("--b-values", default="1,64")
    bid.add_argument("--samples", type=int, default=2_560)
    bid.add_argument("--repeats", type=int, default=3)
    bid.add_argument("--alpha-max-values", default="0.1,0.2,0.3,0.5,1")
    bid.add_argument("--primary-alpha-max", type=float, default=0.2)
    bid.add_argument("--bid-steps", type=int, default=100_000)
    bid.add_argument("--bid-delta", type=float, default=5e-3)
    bid.add_argument("--gride-range-max", type=int, default=64)
    bid.add_argument("--gride-jobs", type=int, default=1)
    bid.add_argument("--seed", type=int, default=0)
    bid.add_argument("--no-progress", action="store_true")
    structured = subparsers.add_parser(
        "part2-structured",
        help="run one controlled suite of LLM-motivated structured sparse features",
    )
    structured.add_argument(
        "--suite",
        choices=("support", "amplitude", "geometry", "frequency", "noise", "combined"),
        required=True,
    )
    structured.add_argument("--output", type=Path, required=True)
    structured.add_argument("--dim", type=int, default=32)
    structured.add_argument("--features", type=int, default=256)
    structured.add_argument("--modules", type=int, default=8)
    structured.add_argument("--k-values", default="2,4,8")
    structured.add_argument("--n-values", default="4096,16384")
    structured.add_argument("--repeats", type=int, default=5)
    structured.add_argument("--gride-range-max", type=int, default=64)
    structured.add_argument("--gride-jobs", type=int, default=1)
    structured.add_argument("--amplitude-log-scale", type=float, default=0.35)
    structured.add_argument("--geometry-diagnostic-samples", type=int, default=256)
    structured.add_argument("--seed", type=int, default=0)
    structured.add_argument("--no-progress", action="store_true")
    plot = subparsers.add_parser("plot", help="plot a completed result directory")
    plot.add_argument("results_dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plot":
        print(plot_results(args.results_dir))
        return
    if args.command == "part2-structured":
        config = StructuredExperimentConfig(
            suite=args.suite,
            representation_dim=args.dim,
            feature_count=args.features,
            module_count=args.modules,
            active_features=_comma_separated_ints(args.k_values),
            sample_counts=_comma_separated_ints(args.n_values),
            repeats=args.repeats,
            gride_range_max=args.gride_range_max,
            gride_n_jobs=args.gride_jobs,
            amplitude_log_scale=args.amplitude_log_scale,
            geometry_diagnostic_samples=args.geometry_diagnostic_samples,
            seed=args.seed,
        )
        artifacts = run_structured_experiment(
            config, args.output, show_progress=not args.no_progress
        )
        print(f"Wrote results to {artifacts.output_dir}")
        print(f"Wrote {artifacts.figure_path}")
        print(
            f"Read {artifacts.interpretation_path} after inspecting "
            f"{artifacts.profile_path.name}."
        )
        return
    if args.command == "manifold-geometry":
        config = ManifoldGeometryConfig(
            representation_dim=args.dim,
            feature_count=args.features,
            intrinsic_dims=_comma_separated_ints(args.d_values),
            sample_counts=_comma_separated_ints(args.n_values),
            activity_multipliers=_comma_separated_floats(args.activity_multipliers),
            feature_strengths=_comma_separated_floats(args.feature_strengths),
            repeats=args.repeats,
            gride_range_max=args.gride_range_max,
            gride_n_jobs=args.gride_jobs,
            jacobian_samples=args.jacobian_samples,
            seed=args.seed,
        )
        artifacts = run_manifold_geometry_experiment(
            config, args.output, show_progress=not args.no_progress
        )
        figure_path = plot_results(artifacts.output_dir)
        print(f"Wrote results to {artifacts.output_dir}")
        print(f"Wrote {figure_path}")
        print(f"Read {artifacts.verdict_path} after inspecting {artifacts.profile_path.name}.")
        return
    if args.command == "bid-audit":
        config = BidAuditConfig(
            representation_dim=args.dim,
            feature_count=args.features,
            active_features=_comma_separated_ints(args.k_values),
            support_pool_sizes=_comma_separated_ints(args.b_values),
            sample_count=args.samples,
            repeats=args.repeats,
            alpha_max_values=_comma_separated_floats(args.alpha_max_values),
            primary_alpha_max=args.primary_alpha_max,
            bid_steps=args.bid_steps,
            bid_delta=args.bid_delta,
            gride_range_max=args.gride_range_max,
            gride_n_jobs=args.gride_jobs,
            seed=args.seed,
        )
        artifacts = run_bid_audit_experiment(
            config, args.output, show_progress=not args.no_progress
        )
        figure_path = plot_results(artifacts.output_dir)
        print(f"Wrote results to {artifacts.output_dir}")
        print(f"Wrote {figure_path}")
        print(
            f"Read {artifacts.verdict_path} after inspecting "
            f"{artifacts.fit_path.name} and {artifacts.histogram_path.name}."
        )
        return
    artifacts = run_experiment(_config_from_args(args), args.output, show_progress=not args.no_progress)
    figure_path = plot_results(artifacts.output_dir)
    print(f"Wrote results to {artifacts.output_dir}")
    print(f"Wrote {figure_path}")
    print(f"Read {artifacts.verdict_path} after inspecting {artifacts.profile_path.name}.")


if __name__ == "__main__":
    main()

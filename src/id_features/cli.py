"""Command-line entry point for the first three controlled hypotheses."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .experiments import run_experiment
from .plotting import plot_results


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
    geometry = subparsers.add_parser("geometry-control", help="hold k fixed and sweep correlation rho")
    _add_common_run_arguments(geometry)
    geometry.add_argument("--k", type=int, default=8)
    geometry.add_argument("--rho-values", default="0,0.25,0.5,0.75,0.9")
    plot = subparsers.add_parser("plot", help="plot a completed result directory")
    plot.add_argument("results_dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plot":
        print(plot_results(args.results_dir))
        return
    artifacts = run_experiment(_config_from_args(args), args.output, show_progress=not args.no_progress)
    figure_path = plot_results(artifacts.output_dir)
    print(f"Wrote results to {artifacts.output_dir}")
    print(f"Wrote {figure_path}")
    print(f"Read {artifacts.verdict_path} after inspecting {artifacts.profile_path.name}.")


if __name__ == "__main__":
    main()

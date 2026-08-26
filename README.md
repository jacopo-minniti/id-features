# Intrinsic dimension and sparse features

This repository is a deliberately small, **untrained** test bed for the first three hypotheses about sparse representations. It generates exact latent-feature labels, encodes them as `h = Wz`, estimates a full multi-scale intrinsic-dimension profile with DADApy's GRIDE estimator, and trains held-out linear probes to recover every latent feature.

It does not assume that intrinsic dimension explains linear accessibility. Each command has pre-specified PASS / NOT ESTABLISHED criteria, so a result only supports the narrow toy-model hypothesis when the intended controls actually held.

## The three hypothesis tests

`load-capacity` tests hypothesis 1. It sweeps both active-feature count `k` and total capacity `m` at fixed `D`. Within each replicate, smaller capacities are literal prefixes of the same largest dictionary. A PASS requires: local ID rises with `k`, stays within 25% of `k`, and changes by no more than 10% across `m` at fixed `k`.

`load-ratio` tests hypothesis 2. It sweeps both `k` and `D` at fixed `m`, measures held-out AUROC and normalized geometric margin, and assesses them against measured local `d_ID / D`. A PASS requires negative monotonic associations for both accessibility measures and comparable AUROC for repeated nominal `k / D` values.

`geometry-control` tests hypothesis 3. It holds `k`, `D`, and `m` fixed while sweeping `rho`, the common-direction strength. A PASS requires at least one nonzero-`rho` condition with every saved-rank GRIDE estimate within 10% of the isotropic profile and a replicated (95%-lower-bound) AUROC decrease of at least 0.02. This is a genuine toy-model counterexample to ID sufficiency, rather than merely a changing-geometry correlation.

## Install

This project uses [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --group dev
```

## Run

Use a small smoke run first:

```bash
uv run id-features load-capacity \
  --output results/smoke-load-capacity \
  --dim 12 --features 32 --k-values 1,2,4 \
  --m-values 16,24,32 \
  --id-samples 300 --train-samples 300 --test-samples 300 \
  --repeats 1 --gride-range-max 16

uv run id-features load-ratio \
  --output results/smoke-load-ratio \
  --features 32 --k-values 1,2,4 --d-values 8,12,16 \
  --id-samples 300 --train-samples 300 --test-samples 300 \
  --repeats 1 --gride-range-max 16

uv run id-features geometry-control \
  --output results/smoke-geometry \
  --dim 12 --features 32 --k 4 --rho-values 0,0.5,0.9 \
  --id-samples 300 --train-samples 300 --test-samples 300 \
  --repeats 1 --gride-range-max 16
```

The default commands use five independent replicates. Run all three before making the combined conclusion:

```bash
uv run id-features load-capacity --output results/hypothesis-1
uv run id-features load-ratio --output results/hypothesis-2
uv run id-features geometry-control --output results/hypothesis-3
```

Interactive runs display `completed measurements / total measurements`, elapsed time, an ETA, and the condition currently being measured. In a Slurm log, the same default becomes one durable completion line per measurement. `load-capacity`'s default is 75 measurements (`3 m × 5 k × 5 repeats`). One measurement consists of one GRIDE profile plus all held-out feature probes, so progress advances only after that complete unit. Use `--no-progress` only when a batch log should remain quiet.

## Slurm: hypothesis 1

Submit the CPU-only job from the repository root:

```bash
sbatch scripts/run_hypothesis_1.sh
```

It follows the local Slurm convention used by the L2D launchers: `.logs/slurm-id-h1-capacity-<job-id>.out` contains a durable `[completed/75]` line after every measurement, and results go to `results/hypothesis-1-<job-id>/`. Monitor it with:

```bash
squeue -j <job-id>
tail -f .logs/slurm-id-h1-capacity-<job-id>.out
```

The script requests four CPUs and no GPU, and passes all four explicitly to each GRIDE nearest-neighbour calculation. The 75 outer measurements and the per-feature probes are still sequential. Direct local invocations default to `--gride-jobs 1`, avoiding accidental login-node saturation. Override a safe runtime control at submission, for example `REPEATS=1 ID_SAMPLES=500 TRAIN_SAMPLES=500 TEST_SAMPLES=500 sbatch scripts/run_hypothesis_1.sh` for a calibration job. Do not treat that reduced job as evidence for the hypothesis.

Plot any completed result directory:

```bash
uv run id-features plot results/hypothesis-1
```

## Outputs and interpretation

Each run writes only inspectable artifacts:

- `config.json` — every random-generation and measurement setting.
- `gride_profiles.csv` — one GRIDE estimate per replicate, condition, and neighbourhood rank. This is the primary ID result; do not collapse it before inspecting scale dependence.
- `summary.csv` — local rank-2 GRIDE ID, `d_ID / D`, held-out AUROC, balanced accuracy, normalized signed margin, and realized mean feature alignment.
- `interpretation.md` — the prespecified decision report. It explicitly says PASS or NOT ESTABLISHED; it does not turn a correlation into a general mechanistic claim.
- `overview.png` — multi-scale profiles and local-ID versus accessibility scatter plot.

Run the tests with:

```bash
uv run pytest
```

See [docs/](docs/) for the theory, hypothesis boundaries, and implementation decisions.

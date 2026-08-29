# Intrinsic dimension and sparse features

This repository is a deliberately small, **untrained** test bed for the first three hypotheses about sparse representations. It generates exact latent-feature labels, encodes them as `h = Wz`, estimates a full multi-scale intrinsic-dimension profile with DADApy's GRIDE estimator, and trains held-out linear probes to recover every latent feature.

It does not assume that intrinsic dimension explains linear accessibility. Each command has pre-specified PASS / NOT ESTABLISHED criteria, so a result only supports the narrow toy-model hypothesis when the intended controls actually held.

## The three hypothesis tests

`load-capacity` tests hypothesis 1. It sweeps both active-feature count `k` and total capacity `m` at fixed `D`. Within each replicate, smaller capacities are literal prefixes of the same largest dictionary. A PASS requires: local ID rises with `k`, stays within 25% of `k`, and changes by no more than 10% across `m` at fixed `k`.

`support-pool` is the controlled follow-up for hypothesis 1. It fixes `D` and `m`, crosses the number of exact-k supports `B` with total sample count `N`, and saves every GRIDE neighborhood rank `r`. Supports are nested across `B`, samples per support are nested across `N`, and every profile row records both mean same-support neighbor fraction and the fraction of fully support-pure neighborhoods. A PASS supports only the conditional claim that local GRIDE recovers `k` when its neighborhoods remain on one support.

`load-ratio` tests hypothesis 2. It sweeps both `k` and `D` at fixed `m`, measures held-out AUROC and normalized geometric margin, and assesses them against measured local `d_ID / D`. A PASS requires negative monotonic associations for both accessibility measures and comparable AUROC for repeated nominal `k / D` values.

`geometry-control` tests hypothesis 3. It holds `k`, `D`, and `m` fixed while sweeping `rho`, the common-direction strength. A PASS requires at least one nonzero-`rho` condition with every saved-rank GRIDE estimate within 10% of the isotropic profile and a replicated (95%-lower-bound) AUROC decrease of at least 0.02. This is a genuine toy-model counterexample to ID sufficiency, rather than merely a changing-geometry correlation.

`manifold-geometry` is an estimator audit before making the generator more LLM-like. It starts from uniform points on a boundaryless sphere with known intrinsic dimension `d`, then adds continuously varying sparse ReLU features in a subspace orthogonal to an isometric residual stream. The combined representation retains population ID exactly `d` at every tested feature strength. The crossed controls vary `d`, sample count `N`, expected active count, and sparse-feature strength. Saved diagnostics expose finite neighborhood radius, local Jacobian conditioning, local volume variation, and nearest-neighbor active-set overlap.

`bid-audit` asks whether the Binary Intrinsic Dimension (BID) estimator of Acevedo, Rodriguez, and Laio can recover `k` more accurately. It follows the DADApy protocol: spins are `-1/+1`, BID is fit to the complete pairwise Hamming-distance histogram, the local fit uses the package default `alpha_max=0.2`, the fit quantile is swept through `1.0`, and empirical/model histograms are saved. Independent `k`-bit and sign-binarized latent-amplitude controls must recover `k`. These are compared with population-centered activation signs, the paper's two-bit bins at `(-sigma, 0, sigma)`, exact support masks, and GRIDE on the identical continuous samples. The report does not assume that binary ID must equal continuous manifold ID.

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

uv run id-features support-pool \
  --output results/smoke-support-pool \
  --dim 12 --features 32 --k-values 1,2,4 \
  --b-values 1,4 --n-values 128,256 \
  --repeats 1 --gride-range-max 16

uv run id-features manifold-geometry \
  --output results/smoke-manifold-geometry \
  --dim 24 --features 64 --d-values 4 \
  --n-values 512 --activity-multipliers 1 \
  --feature-strengths 0.5 --repeats 1 --gride-range-max 16

uv run id-features bid-audit \
  --output results/smoke-bid-audit \
  --dim 24 --features 64 --k-values 2,8 \
  --b-values 1,64 --samples 256 --repeats 1 \
  --alpha-max-values 0.1,0.2,0.5,1 --primary-alpha-max 0.2 \
  --bid-steps 10000 --gride-range-max 16
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

The support-controlled follow-up also has a CPU-only launcher:

```bash
sbatch scripts/run_support_pool.sh
```

Its default is `5 k × 4 B × 3 N × 5 repeats = 300` complete GRIDE measurements, with ranks 2 through 64 retained inside each measurement. Results go to `results/support-pool-<job-id>/`.

The boundaryless geometry audit also has a CPU-only launcher:

```bash
sbatch scripts/run_manifold_geometry.sh
```

Its default crosses `d={4,8,16}`, `N={4096,16384}`, expected activity `{d/2,d,2d}`, and feature strength `gamma={0.25,1,4}` over three independent maps. Sphere and isometric-residual baselines are saved alongside sparse-only warning controls and the known-ID combined representations. Results go to `results/manifold-geometry-<job-id>/`.

The BID estimator audit uses the paper-scale sample count and has its own CPU launcher:

```bash
sbatch scripts/run_bid_audit.sh
```

It uses `N=2560` so `B=64` remains exactly balanced, `k={2,4,8,16}`, three repeats, and the DADApy tutorial's `100000` optimizer steps with `delta=0.005`. Results go to `results/bid-audit-<job-id>/`.

Plot any completed result directory:

```bash
uv run id-features plot results/hypothesis-1
```

## Outputs and interpretation

Each run writes only inspectable artifacts:

- `config.json` — every random-generation and measurement setting.
- `gride_profiles.csv` — one GRIDE estimate per replicate, condition, and neighbourhood rank. This is the primary ID result; do not collapse it before inspecting scale dependence.
- `summary.csv` — local rank-2 GRIDE ID plus experiment-specific diagnostics. The manifold audit includes known population ID, active-set overlap, local Jacobian conditioning, volume variation, and neighborhood radius.
- `bid_fits.csv` and `bid_histograms.csv` — for `bid-audit`, every BID value, fit quantile, binary-code diagnostic, optimization diagnostic, and explicit empirical-versus-model Hamming histogram.
- `interpretation.md` — the prespecified decision report. It explicitly says PASS or NOT ESTABLISHED; it does not turn a correlation into a general mechanistic claim.
- `overview.png` — multi-scale profiles and local-ID versus accessibility scatter plot.

Run the tests with:

```bash
uv run pytest
```

Start with the compact [main results document](docs/main.md). The remaining files under [docs/](docs/) contain detailed theory, controls, implementation decisions, the [BID audit](docs/04-bid-results.md), the full [Part 1 proof](docs/05-when-id-counts-active-features.md), and the expanded [structured-model results](docs/07-part2-results.md).

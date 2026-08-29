#!/usr/bin/env bash
#SBATCH --job-name=id-part2-structured
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=.logs/slurm-%x-%A_%a.out
#SBATCH --error=.logs/slurm-%x-%A_%a.out
#SBATCH --open-mode=append

# Part 2: six paired suites for a structured sparse-superposition model.
# Submit from the repository root: sbatch scripts/run_part2_structured.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repository root}"
mkdir -p .logs results

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER:-user}-id-features-mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

SUITES=(support amplitude geometry frequency noise combined)
SUITE="${SUITES[${SLURM_ARRAY_TASK_ID}]}"
RESULT_DIR="results/part2-${SUITE}-${SLURM_ARRAY_JOB_ID}"

echo "Starting Part 2 suite '$SUITE' in $PWD"
echo "Results: $RESULT_DIR"
echo "Controls: D=32 m=256 modules=8 k={2,4,8} N={4096,16384} repeats=5 ranks<=64"

srun uv run id-features part2-structured \
  --suite "$SUITE" \
  --output "$RESULT_DIR" \
  --dim 32 \
  --features 256 \
  --modules 8 \
  --k-values 2,4,8 \
  --n-values 4096,16384 \
  --repeats 5 \
  --gride-range-max 64 \
  --gride-jobs "${SLURM_CPUS_PER_TASK:-4}" \
  --amplitude-log-scale 0.35 \
  --geometry-diagnostic-samples 256

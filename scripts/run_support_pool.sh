#!/usr/bin/env bash
#SBATCH --job-name=id-h1-support-pool
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=.logs/slurm-%x-%j.out
#SBATCH --error=.logs/slurm-%x-%j.out
#SBATCH --open-mode=append

# Hypothesis 1b: recover k only where B/N/r keeps neighborhoods support-pure.
# Submit from the repository root: sbatch scripts/run_support_pool.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repository root}"
mkdir -p .logs results

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER:-user}-id-features-mpl-${SLURM_JOB_ID}}"
mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

RESULT_DIR="${RESULT_DIR:-results/support-pool-${SLURM_JOB_ID}}"
REPEATS="${REPEATS:-5}"
DIM="${DIM:-32}"
FEATURES="${FEATURES:-256}"
K_VALUES="${K_VALUES:-1,2,4,8,16}"
B_VALUES="${B_VALUES:-1,4,16,64}"
N_VALUES="${N_VALUES:-1024,2048,4096}"
GRIDE_RANGE_MAX="${GRIDE_RANGE_MAX:-64}"

echo "Starting support-pool experiment in $PWD"
echo "Results: $RESULT_DIR"
echo "Controls: D=$DIM m=$FEATURES k=$K_VALUES B=$B_VALUES N=$N_VALUES repeats=$REPEATS ranks<=${GRIDE_RANGE_MAX}"

srun uv run id-features support-pool \
  --output "$RESULT_DIR" \
  --dim "$DIM" \
  --features "$FEATURES" \
  --k-values "$K_VALUES" \
  --b-values "$B_VALUES" \
  --n-values "$N_VALUES" \
  --repeats "$REPEATS" \
  --gride-range-max "$GRIDE_RANGE_MAX" \
  --gride-jobs "${SLURM_CPUS_PER_TASK:-4}"

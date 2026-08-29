#!/usr/bin/env bash
#SBATCH --job-name=id-part1-gride
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=.logs/slurm-%x-%j.out
#SBATCH --error=.logs/slurm-%x-%j.out
#SBATCH --open-mode=append

# Part 1: fixed-support validation of the analytical d_ID = k result.
# Submit from the repository root: sbatch scripts/run_part1_gride.sh
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

RESULT_DIR="results/part1-fixed-support-${SLURM_JOB_ID}"

echo "Starting Part 1 fixed-support GRIDE validation in $PWD"
echo "Results: $RESULT_DIR"
echo "Controls: D=32 m=256 k={1,2,4,8,16} B=1 N={1024,4096,16384} repeats=5 ranks<=64"

srun uv run id-features support-pool \
  --output "$RESULT_DIR" \
  --dim 32 \
  --features 256 \
  --k-values 1,2,4,8,16 \
  --b-values 1 \
  --n-values 1024,4096,16384 \
  --repeats 5 \
  --gride-range-max 64 \
  --gride-jobs "${SLURM_CPUS_PER_TASK:-4}"

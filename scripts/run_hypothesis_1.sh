#!/usr/bin/env bash
#SBATCH --job-name=id-h1-capacity
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=.logs/slurm-%x-%j.out
#SBATCH --error=.logs/slurm-%x-%j.out
#SBATCH --open-mode=append

# Hypothesis 1: local ID tracks concurrent k and is stable as capacity m changes.
# Submit from the repository root: sbatch scripts/run_hypothesis_1.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repository root}"
mkdir -p .logs results

# Keep uv and Matplotlib writes out of the login-node home-cache locations.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER:-user}-id-features-mpl-${SLURM_JOB_ID}}"
mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR"

# The implementation uses CPU NumPy/SciPy/scikit-learn/DADApy; do not request a GPU.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

RESULT_DIR="${RESULT_DIR:-results/hypothesis-1-${SLURM_JOB_ID}}"
REPEATS="${REPEATS:-5}"
ID_SAMPLES="${ID_SAMPLES:-4000}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-4000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"

echo "Starting hypothesis 1 in $PWD"
echo "Results: $RESULT_DIR"
echo "Controls: repeats=$REPEATS id_samples=$ID_SAMPLES train_samples=$TRAIN_SAMPLES test_samples=$TEST_SAMPLES"

srun uv run id-features load-capacity \
  --output "$RESULT_DIR" \
  --repeats "$REPEATS" \
  --id-samples "$ID_SAMPLES" \
  --train-samples "$TRAIN_SAMPLES" \
  --test-samples "$TEST_SAMPLES" \
  --gride-jobs "${SLURM_CPUS_PER_TASK:-4}"

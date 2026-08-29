#!/usr/bin/env bash
#SBATCH --job-name=id-part3-pool
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=.logs/slurm-%x-%A_%a.out
#SBATCH --error=.logs/slurm-%x-%A_%a.out
#SBATCH --open-mode=append

# Targeted structured support-pool control. Submit from the repository root.
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

KS=(4 4 8 8)
RHOS=(0.9 1.0 0.9 1.0)
K="${KS[${SLURM_ARRAY_TASK_ID}]}"
RHO="${RHOS[${SLURM_ARRAY_TASK_ID}]}"
RESULT_ROOT="results/part3-support-pool-${SLURM_ARRAY_JOB_ID}"
RESULT_DIR="${RESULT_ROOT}/task-${SLURM_ARRAY_TASK_ID}"

echo "Starting structured support-pool task: k=$K rho=$RHO"
echo "B={1,16,256,4096} N={4096,16384} repeats=3 ranks<=64"

srun uv run python scripts/part3_support_pool.py run \
  --output "$RESULT_DIR" \
  --k "$K" \
  --rho "$RHO" \
  --gride-jobs "${SLURM_CPUS_PER_TASK:-4}"

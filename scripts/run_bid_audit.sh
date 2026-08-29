#!/usr/bin/env bash
#SBATCH --job-name=id-bid-audit
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=.logs/slurm-%x-%j.out
#SBATCH --error=.logs/slurm-%x-%j.out
#SBATCH --open-mode=append

# DADApy BID audit: fixed versus pooled supports and matched GRIDE profiles.
# Submit from the repository root: sbatch scripts/run_bid_audit.sh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repository root}"
mkdir -p .logs results

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER:-user}-id-features-mpl-${SLURM_JOB_ID}}"
mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR"

export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

RESULT_DIR="${RESULT_DIR:-results/bid-audit-${SLURM_JOB_ID}}"
REPEATS="${REPEATS:-3}"
DIM="${DIM:-64}"
FEATURES="${FEATURES:-256}"
K_VALUES="${K_VALUES:-2,4,8,16}"
B_VALUES="${B_VALUES:-1,64}"
SAMPLES="${SAMPLES:-2560}"
ALPHA_MAX_VALUES="${ALPHA_MAX_VALUES:-0.1,0.2,0.3,0.5,1}"
PRIMARY_ALPHA_MAX="${PRIMARY_ALPHA_MAX:-0.2}"
BID_STEPS="${BID_STEPS:-100000}"
GRIDE_RANGE_MAX="${GRIDE_RANGE_MAX:-64}"

echo "Starting BID audit in $PWD"
echo "Results: $RESULT_DIR"
echo "Controls: D=$DIM m=$FEATURES k=$K_VALUES B=$B_VALUES N=$SAMPLES repeats=$REPEATS alpha=$ALPHA_MAX_VALUES local-alpha=$PRIMARY_ALPHA_MAX BID-steps=$BID_STEPS"

srun uv run id-features bid-audit \
  --output "$RESULT_DIR" \
  --dim "$DIM" \
  --features "$FEATURES" \
  --k-values "$K_VALUES" \
  --b-values "$B_VALUES" \
  --samples "$SAMPLES" \
  --repeats "$REPEATS" \
  --alpha-max-values "$ALPHA_MAX_VALUES" \
  --primary-alpha-max "$PRIMARY_ALPHA_MAX" \
  --bid-steps "$BID_STEPS" \
  --gride-range-max "$GRIDE_RANGE_MAX" \
  --gride-jobs "${SLURM_CPUS_PER_TASK:-4}"

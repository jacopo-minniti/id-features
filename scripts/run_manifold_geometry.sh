#!/usr/bin/env bash
#SBATCH --job-name=id-manifold-geometry
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=.logs/slurm-%x-%j.out
#SBATCH --error=.logs/slurm-%x-%j.out
#SBATCH --open-mode=append

# Known-ID audit: boundaryless sphere plus orthogonal residual/sparse branches.
# Submit from the repository root: sbatch scripts/run_manifold_geometry.sh
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

RESULT_DIR="${RESULT_DIR:-results/manifold-geometry-${SLURM_JOB_ID}}"
REPEATS="${REPEATS:-3}"
DIM="${DIM:-64}"
FEATURES="${FEATURES:-256}"
D_VALUES="${D_VALUES:-4,8,16}"
N_VALUES="${N_VALUES:-4096,16384}"
ACTIVITY_MULTIPLIERS="${ACTIVITY_MULTIPLIERS:-0.5,1,2}"
FEATURE_STRENGTHS="${FEATURE_STRENGTHS:-0.25,1,4}"
GRIDE_RANGE_MAX="${GRIDE_RANGE_MAX:-64}"
JACOBIAN_SAMPLES="${JACOBIAN_SAMPLES:-128}"

# Narrow predeclared extension used to test the slow high-dimension density trend.
if [[ "${DENSITY_FOLLOWUP:-0}" == "1" ]]; then
  RESULT_DIR="results/manifold-geometry-density-${SLURM_JOB_ID}"
  D_VALUES="16"
  N_VALUES="4096,16384,65536"
  ACTIVITY_MULTIPLIERS="1"
  FEATURE_STRENGTHS="1"
  REPEATS="3"
fi

echo "Starting manifold-geometry experiment in $PWD"
echo "Results: $RESULT_DIR"
echo "Controls: D=$DIM m=$FEATURES d=$D_VALUES N=$N_VALUES activity=$ACTIVITY_MULTIPLIERS gamma=$FEATURE_STRENGTHS repeats=$REPEATS ranks<=${GRIDE_RANGE_MAX}"

srun uv run id-features manifold-geometry \
  --output "$RESULT_DIR" \
  --dim "$DIM" \
  --features "$FEATURES" \
  --d-values "$D_VALUES" \
  --n-values "$N_VALUES" \
  --activity-multipliers "$ACTIVITY_MULTIPLIERS" \
  --feature-strengths "$FEATURE_STRENGTHS" \
  --repeats "$REPEATS" \
  --gride-range-max "$GRIDE_RANGE_MAX" \
  --gride-jobs "${SLURM_CPUS_PER_TASK:-4}" \
  --jacobian-samples "$JACOBIAN_SAMPLES"

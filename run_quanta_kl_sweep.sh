#!/usr/bin/env bash
#SBATCH --job-name=q-kl-sweep
#SBATCH --partition=a3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:35:00
#SBATCH --array=0-5%3
#SBATCH --output=/tmp/q-kl-sweep-%A_%a.out

set -euo pipefail

weights=(0p0 0p2 0p3 0p5 0p7 0p8)
weight="${weights[${SLURM_ARRAY_TASK_ID}]}"
repo=/home/jacopominniti_sakana_ai/quanta
run_dir="$repo/.experiments/quanta_discovery/number_naming/quanta_discovery_v0/exact_gd/extreme_generalization_v2-layers3-d32-heads1-steps5000-seed0-6571b3eb47b4"

cd "$repo"
PYTHONPATH=. "$repo/.venv/bin/python" scripts/train_quanta_model.py "$run_dir" \
  --priority-dir "$run_dir/factorization" \
  --config "/home/jacopominniti_sakana_ai/id-features/.quanta_kl_sweep_${weight}.yaml" \
  --output-dir "$run_dir/qmodel_ablation_direct_attention_l0target3_kl${weight}_20260827" \
  --evaluation-batch-size 300 \
  --greedy-max-examples 300 \
  --seed 0 \
  --device cuda

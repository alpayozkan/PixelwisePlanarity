#!/bin/bash
# Aggregation step for the fast partitioned GT evaluation. Runs after all partition
# array tasks succeed (sbatch --dependency=afterok). Merges the per-partition
# results_shard_*.csv into the standard results.csv / results_per_scene.csv /
# results_dataset.csv via evaluate_all_baselines.py --aggregate-only (which calls
# _merge_shards internally).
#
# Usage (via sbatch):
#   run_gt_aggregate_job.sh <EVAL_ROOT>
set -eu

EVAL_ROOT="$1"

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
EVAL_SCRIPT="$PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py"
CONDA_SH="/cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="planamono"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

echo "[INFO] Host: $(hostname)  Aggregating shards in: $EVAL_ROOT"

python "$EVAL_SCRIPT" \
    --methods gt \
    --eval-root "$EVAL_ROOT" \
    --output-dir "$EVAL_ROOT" \
    --aggregate-only

echo "[SUCCESS] Aggregation done -> $EVAL_ROOT/gt_v6/results.csv (+ per_scene, dataset)"

#!/bin/bash
# Re-run eval-only for moge_hires_ep3_v5origparams_relative_seg on ScanNet++.
# Inference H5 already exists; stale shard file has been deleted.
#
# Usage:
#   bash submit_eval_v5origparams_rel_scannetpp.sh

set -euo pipefail

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/v5orig_rel_full"
mkdir -p "$LOG_DIR"

JOB_ID=$(sbatch --parsable \
    --job-name="eval_v5orig_rel" \
    --time=12:00:00 \
    --cpus-per-task=16 \
    --mem-per-cpu=4G \
    --output="$LOG_DIR/eval_rerun_%j.out" \
    --error="$LOG_DIR/eval_rerun_%j.err" \
    --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
python $PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py \
    --methods moge_hires_ep3_v5origparams_relative_seg
'")

echo "Submitted: $JOB_ID"
echo "Log: $LOG_DIR/eval_rerun_${JOB_ID}.out"
echo "Monitor: squeue -u \$USER"

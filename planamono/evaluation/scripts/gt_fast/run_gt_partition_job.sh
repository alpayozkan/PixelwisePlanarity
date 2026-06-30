#!/bin/bash
# One scene-partition of the GT evaluation. Submitted as a SLURM array task by
# submit_gt_seed42_fast.sh. Computes its scene range from $SLURM_ARRAY_TASK_ID and
# writes results_shard_<scene_start>.csv into $EVAL_ROOT/gt_v6/.
#
# Real bash script (NOT sbatch --wrap, which runs under /bin/sh and breaks
# `source`/`conda activate`).
#
# Usage (via sbatch --array):
#   run_gt_partition_job.sh <SEED> <EVAL_ROOT> <SCENES_PER_JOB> <TOTAL_SCENES> [NUM_WORKERS]
set -eu

SEED="$1"
EVAL_ROOT="$2"
SCENES_PER_JOB="$3"
TOTAL_SCENES="$4"
NUM_WORKERS="${5:-8}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SCENE_START=$(( TASK_ID * SCENES_PER_JOB ))
SCENE_END=$(( SCENE_START + SCENES_PER_JOB ))
if [ "$SCENE_END" -gt "$TOTAL_SCENES" ]; then SCENE_END="$TOTAL_SCENES"; fi

if [ "$SCENE_START" -ge "$TOTAL_SCENES" ]; then
    echo "[INFO] Partition $TASK_ID: scene_start=$SCENE_START >= total=$TOTAL_SCENES -> empty, nothing to do."
    exit 0
fi

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
EVAL_SCRIPT="$PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py"
CONDA_SH="/cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="planamono"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

echo "[INFO] Host: $(hostname)  Partition: $TASK_ID  Scenes [$SCENE_START:$SCENE_END)  Seed: $SEED  Eval-root: $EVAL_ROOT"

python "$EVAL_SCRIPT" \
    --methods gt \
    --ransac-seed "$SEED" \
    --eval-root "$EVAL_ROOT" \
    --scene-start "$SCENE_START" \
    --scene-end "$SCENE_END" \
    --num-workers "$NUM_WORKERS"

echo "[SUCCESS] Partition $TASK_ID done -> $EVAL_ROOT/gt_v6/results_shard_${SCENE_START}.csv"

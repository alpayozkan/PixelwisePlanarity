#!/bin/bash
# Single ZEROPLANE-evaluation job (one seed, one run dir). Submitted by
# submit_zeroplane_seed_repro.sh as a real bash script so `source`/`conda activate`
# work (sbatch --wrap runs under /bin/sh, which breaks both).
#
# ZeroPlane uses label 20 for non-planar; evaluate_all_baselines.py auto-remaps
# 20 -> 0 via the method's nonplanar_label config (nothing to do here).
#
# Usage (normally invoked via sbatch):
#   run_zeroplane_eval_job.sh <SEED> <EVAL_ROOT> [NUM_WORKERS]
set -eu

SEED="$1"
EVAL_ROOT="$2"
NUM_WORKERS="${3:-8}"

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
EVAL_SCRIPT="$PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py"
CONDA_SH="/cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="planamono"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

echo "[INFO] Host: $(hostname)  Seed: $SEED  Eval-root: $EVAL_ROOT  Python: $(which python)"

python "$EVAL_SCRIPT" \
    --methods zeroplane \
    --ransac-seed "$SEED" \
    --eval-root "$EVAL_ROOT" \
    --output-dir "$EVAL_ROOT" \
    --num-workers "$NUM_WORKERS"

echo "[SUCCESS] ZEROPLANE eval done for seed $SEED -> $EVAL_ROOT"

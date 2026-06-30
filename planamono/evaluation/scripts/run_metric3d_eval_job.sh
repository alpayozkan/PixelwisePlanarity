#!/bin/bash
# Single METRIC3D-evaluation job (one seed, one run dir). Submitted by
# submit_metric3d_seed_repro.sh as a real bash script so `source`/`conda activate`
# work (sbatch --wrap runs under /bin/sh, which breaks both).
#
# Uses the `metric3d` entry in the METHODS dict of evaluate_all_baselines.py,
# which reads the predicted `planes` from per-scene rendered_v2.h5 (0 = non-planar)
# and evaluates them with GT depth + dataset K -- the same treatment ours/zeroplane
# get, so the seed sweep is cross-method consistent.
#
# Usage (normally invoked via sbatch):
#   run_metric3d_eval_job.sh <SEED> <EVAL_ROOT> [NUM_WORKERS]
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
    --methods metric3d \
    --ransac-seed "$SEED" \
    --eval-root "$EVAL_ROOT" \
    --output-dir "$EVAL_ROOT" \
    --num-workers "$NUM_WORKERS"

echo "[SUCCESS] METRIC3D eval done for seed $SEED -> $EVAL_ROOT"

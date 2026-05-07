#!/bin/bash
# ============================================================
# Segmentation Configs Ablation (5 configs: loose → strict)
#
# Assumes raw MoGe H5 already exists at RAW_ROOT.
# Submits 5 parallel eval jobs (one per config) + 1 aggregation job.
#
# Usage:
#   bash submit_configs_ablation.sh
# ============================================================

set -euo pipefail

# ── Config ──
N_SCENES=10
SPLIT="test"

# ── Paths ──
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
RAW_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference_raw/moge_hires_4ds_ep2_sensitivity_raw"
RESULTS_DIR="/cluster/scratch/aoezkan/planeseg/scannetpp/eval/configs_ablation"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/configs_ablation"

SCRIPT="$PROJECT_ROOT/evaluation/quantitative/segmentation_configs_ablation.py"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Segmentation Configs Ablation (5 configs)"
echo "============================================================"
echo "Raw root:    $RAW_ROOT"
echo "N scenes:    $N_SCENES"
echo "Results:     $RESULTS_DIR"
echo "============================================================"

# ── Eval jobs (one per config, parallel) ──
CONFIG_NAMES=("config1_loose" "config2_relaxed" "config3_default" "config4_moderate" "config5_strict")
EVAL_JOBS=()

for CFG in "${CONFIG_NAMES[@]}"; do
    JOB=$(sbatch --parsable \
        --time=4:00:00 \
        --cpus-per-task=4 \
        --mem-per-cpu=8G \
        --gpus=rtx_3090:1 \
        --output="$LOG_DIR/${CFG}_%j.out" \
        --error="$LOG_DIR/${CFG}_%j.err" \
        --job-name="cfg_${CFG}" \
        --wrap="
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg
python $SCRIPT \
    --raw_root $RAW_ROOT \
    --output_dir $RESULTS_DIR \
    --configs $CFG \
    --max_scenes $N_SCENES \
    --split $SPLIT
")
    EVAL_JOBS+=($JOB)
    echo "[Eval] $CFG: job $JOB"
done

# Build dependency string
EVAL_DEP=$(IFS=:; echo "${EVAL_JOBS[*]}")

# ── Aggregation job (CPU only, depends on all eval jobs) ──
JOB_AGG=$(sbatch --parsable \
    --dependency=afterok:$EVAL_DEP \
    --time=0:10:00 \
    --cpus-per-task=2 \
    --mem-per-cpu=4G \
    --output="$LOG_DIR/aggregate_%j.out" \
    --error="$LOG_DIR/aggregate_%j.err" \
    --job-name="cfg_agg" \
    --wrap="
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg
python $SCRIPT \
    --raw_root $RAW_ROOT \
    --output_dir $RESULTS_DIR \
    --aggregate-only
")
echo "[Agg]  Aggregate: job $JOB_AGG (depends on ${EVAL_JOBS[*]})"

echo ""
echo "============================================================"
echo "Job chain submitted:"
echo "  Eval jobs:    ${EVAL_JOBS[*]}"
echo "  Aggregation:  $JOB_AGG"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: $RESULTS_DIR"
echo "============================================================"

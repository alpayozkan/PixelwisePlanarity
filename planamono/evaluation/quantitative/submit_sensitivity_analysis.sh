#!/bin/bash
# ============================================================
# Sensitivity Analysis: MoGe 4DS ep2 + v5_relative
#
# Submits 6 SLURM jobs:
#   1. save_moge_raw.py   → raw H5 (planarity, depth, normals) for 10 scenes [GPU]
#   2. sensitivity sweep   → 4 jobs (one per parameter, parallel)             [GPU, for segmentation]
#   3. aggregate tables    → combine results into tables                      [CPU]
#
# Usage:
#   bash submit_sensitivity_analysis.sh
# ============================================================

set -euo pipefail

# ── Config ──
MODEL_PATH="/cluster/scratch/ayavuz/moge_HIRES_4datasets/model_epoch2.pt"
N_SCENES=10
SPLIT="test"

# ── Paths ──
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
RAW_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference_raw/moge_hires_4ds_ep2_sensitivity_raw"
RESULTS_DIR="/cluster/scratch/aoezkan/planeseg/scannetpp/eval/sensitivity_v5rel_4ds_ep2"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/sensitivity"

DATASET_DIR="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
RGB_ROOT="/cluster/project/cvg/Shared_datasets/scannet++/data"

SCRIPT_RAW="$PROJECT_ROOT/inference/planarity/save_moge_raw.py"
SCRIPT_SWEEP="$PROJECT_ROOT/evaluation/quantitative/sensitivity_analysis_v5rel.py"
SCRIPT_AGG="$PROJECT_ROOT/evaluation/quantitative/aggregate_sensitivity.py"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Sensitivity Analysis: MoGe 4DS ep2 + v5_relative"
echo "============================================================"
echo "Model:       $MODEL_PATH"
echo "N scenes:    $N_SCENES"
echo "Raw output:  $RAW_ROOT"
echo "Results:     $RESULTS_DIR"
echo "============================================================"

# ── Step 1: Save raw MoGe outputs (GPU) ──
JOB_RAW=$(sbatch --parsable \
    --time=4:00:00 \
    --cpus-per-task=4 \
    --mem-per-cpu=8G \
    --gpus=rtx_3090:1 \
    --output="$LOG_DIR/raw_inference_%j.out" \
    --error="$LOG_DIR/raw_inference_%j.err" \
    --job-name="sens_raw" \
    --wrap="
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg
python $SCRIPT_RAW \
    --model_path $MODEL_PATH \
    --output_root $RAW_ROOT \
    --dataset_dir $DATASET_DIR \
    --rgb_root $RGB_ROOT \
    --split $SPLIT \
    --max_scenes $N_SCENES \
    --batch_size 16 \
    --num_tokens 1600 \
    --metric_depth
")
echo "[Step 1] Raw inference: job $JOB_RAW"

# ── Step 2: Sweep jobs (one per parameter, depends on raw) ──
PARAMS=("threshold_planarity" "normal_threshold_deg" "depth_threshold" "neighbor_match_count_thresh")
SWEEP_JOBS=()

for PARAM in "${PARAMS[@]}"; do
    JOB_SWEEP=$(sbatch --parsable \
        --dependency=afterok:$JOB_RAW \
        --time=8:00:00 \
        --cpus-per-task=4 \
        --mem-per-cpu=8G \
        --gpus=rtx_3090:1 \
        --output="$LOG_DIR/sweep_${PARAM}_%j.out" \
        --error="$LOG_DIR/sweep_${PARAM}_%j.err" \
        --job-name="sens_${PARAM}" \
        --wrap="
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg
python $SCRIPT_SWEEP \
    --param $PARAM \
    --raw_root $RAW_ROOT \
    --output_dir $RESULTS_DIR \
    --max_scenes $N_SCENES \
    --split $SPLIT
")
    SWEEP_JOBS+=($JOB_SWEEP)
    echo "[Step 2] Sweep $PARAM: job $JOB_SWEEP (depends on $JOB_RAW)"
done

# Build dependency string for aggregation
SWEEP_DEP=$(IFS=:; echo "${SWEEP_JOBS[*]}")

# ── Step 3: Aggregate results (CPU only, depends on all sweeps) ──
JOB_AGG=$(sbatch --parsable \
    --dependency=afterok:$SWEEP_DEP \
    --time=0:10:00 \
    --cpus-per-task=2 \
    --mem-per-cpu=4G \
    --output="$LOG_DIR/aggregate_%j.out" \
    --error="$LOG_DIR/aggregate_%j.err" \
    --job-name="sens_agg" \
    --wrap="
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg
python $SCRIPT_AGG \
    --results_dir $RESULTS_DIR \
    --latex
")
echo "[Step 3] Aggregate: job $JOB_AGG (depends on ${SWEEP_JOBS[*]})"

echo ""
echo "============================================================"
echo "Job chain submitted:"
echo "  Raw inference:  $JOB_RAW"
echo "  Sweeps:         ${SWEEP_JOBS[*]}"
echo "  Aggregation:    $JOB_AGG"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: $RESULTS_DIR"
echo "============================================================"

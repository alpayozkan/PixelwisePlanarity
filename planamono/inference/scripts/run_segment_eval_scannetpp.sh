#!/bin/bash
# Two-stage pipeline: GPU segmentation → distributed CPU evaluation (ScanNet++)
# Usage: bash run_segment_eval_scannetpp.sh [NUM_EVAL_JOBS]
#   NUM_EVAL_JOBS: number of parallel eval jobs (default: 5)

set -euo pipefail

NUM_EVAL_JOBS="${1:-5}"
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/inference"
mkdir -p "$LOG_DIR"

# ---- Input: raw MoGe outputs from stage 1 ----
RAW_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference_raw/moge_hires_ep3_raw"

# ---- Output: must land under H5_ROOT so eval finds it ----
METHOD_NAME="moge_hires_ep3_v11seg_metric"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference/${METHOD_NAME}_h5"

# ---- Segmentation parameters ----
SEG_VERSION="v11"
THRESHOLD_PLANARITY=0.2
NORMAL_THRESHOLD_DEG=5.0
DEPTH_THRESHOLD=0.025
NEIGHBOR_MATCH_COUNT_THRESH=18
ADAPTIVE_FRAC=0.75
MIN_VALID_NEIGHBORS=3
MIN_SEGMENT_PIXELS=50
MERGE_ENABLED=""  # set to "--merge_enabled" to enable

echo "=== Submitting ScanNet++ pipeline ==="
echo "Raw input:  $RAW_ROOT"
echo "Output:     $OUTPUT_ROOT"
echo "Seg:        $SEG_VERSION, planarity=$THRESHOLD_PLANARITY"
echo "Eval jobs:  $NUM_EVAL_JOBS"

# --- Job 1: GPU segmentation ---
SEG_JOB=$(sbatch --parsable \
    --time=4:00:00 \
    --cpus-per-task=8 \
    --mem-per-cpu=8G \
    --gpus=rtx_3090:1 \
    --output="$LOG_DIR/seg_scannetpp_%j.out" \
    --error="$LOG_DIR/seg_scannetpp_%j.err" \
    --wrap="
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
mkdir -p $OUTPUT_ROOT
python $PROJECT_ROOT/inference/planarity/segment_from_raw.py \
    --raw_root $RAW_ROOT \
    --output_root $OUTPUT_ROOT \
    --dataset scannetpp \
    --seg_version $SEG_VERSION \
    --threshold_planarity $THRESHOLD_PLANARITY \
    --normal_threshold_deg $NORMAL_THRESHOLD_DEG \
    --depth_threshold $DEPTH_THRESHOLD \
    --neighbor_match_count_thresh $NEIGHBOR_MATCH_COUNT_THRESH \
    --adaptive_frac $ADAPTIVE_FRAC \
    --min_valid_neighbors $MIN_VALID_NEIGHBORS \
    --min_segment_pixels $MIN_SEGMENT_PIXELS \
    $MERGE_ENABLED
")
echo "[SUBMITTED] Segmentation job: $SEG_JOB"

# --- Job 2: CPU evaluation (array job, depends on segmentation) ---
# ScanNet++ test split has 152 scenes. Divide among NUM_EVAL_JOBS.
TOTAL_SCENES=42
SCENES_PER_JOB=$(( (TOTAL_SCENES + NUM_EVAL_JOBS - 1) / NUM_EVAL_JOBS ))

EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:$SEG_JOB \
    --array=0-$((NUM_EVAL_JOBS - 1)) \
    --time=8:00:00 \
    --cpus-per-task=16 \
    --mem-per-cpu=4G \
    --output="$LOG_DIR/eval_scannetpp_%A_%a.out" \
    --error="$LOG_DIR/eval_scannetpp_%A_%a.err" \
    --wrap="
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
SCENE_START=\$(( SLURM_ARRAY_TASK_ID * $SCENES_PER_JOB ))
SCENE_END=\$(( SCENE_START + $SCENES_PER_JOB ))
if [ \$SCENE_END -gt $TOTAL_SCENES ]; then SCENE_END=$TOTAL_SCENES; fi
echo \"Eval scenes [\$SCENE_START:\$SCENE_END]\"
python $PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py \
    --methods $METHOD_NAME \
    --scene-start \$SCENE_START \
    --scene-end \$SCENE_END
")
echo "[SUBMITTED] Evaluation array job: $EVAL_JOB (${NUM_EVAL_JOBS} tasks, depends on $SEG_JOB)"

# --- Job 3: Aggregate results (depends on all eval jobs) ---
AGG_JOB=$(sbatch --parsable \
    --dependency=afterok:$EVAL_JOB \
    --time=0:10:00 \
    --cpus-per-task=2 \
    --mem-per-cpu=4G \
    --output="$LOG_DIR/agg_scannetpp_%j.out" \
    --error="$LOG_DIR/agg_scannetpp_%j.err" \
    --wrap="
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
python $PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py \
    --methods $METHOD_NAME \
    --aggregate-only
")
echo "[SUBMITTED] Aggregation job: $AGG_JOB (depends on $EVAL_JOB)"

echo ""
echo "Pipeline: $SEG_JOB (seg) → $EVAL_JOB (eval x${NUM_EVAL_JOBS}) → $AGG_JOB (agg)"

#!/bin/bash
# Full pipeline: inference_to_h5 (GPU inference + segmentation) → Eval → Agg
# for v5orig_rel (Sobel + relative depth) with original v5 parameters on ALL 42 test scenes.
#
# Uses inference_to_h5.py (single-stage) — no raw H5 saved, saves ~540GB.
#
# Usage:
#   bash submit_v5orig_rel_full.sh

set -euo pipefail

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/v5orig_rel_full"
mkdir -p "$LOG_DIR"

# ---- Model ----
MODEL_PATH="/cluster/scratch/ayavuz/moge_mixed_output_476644_fixed_cosLR_singlePhase_mixed_HiRes/model_epoch3.pt"
DATASET_DIR="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
RGB_ROOT="/cluster/project/cvg/Shared_datasets/scannet++/data"
NUM_TOKENS=1600
BATCH_SIZE=16

# ---- Segmentation (v5_relative + origparams) ----
H5_BASE="/cluster/scratch/aoezkan/planeseg/scannetpp/inference"
SEG_VERSION="v5_relative"
THRESHOLD_PLANARITY=0.6
NORMAL_THRESHOLD_DEG=10.0
DEPTH_THRESHOLD_REL=0.025
NEIGHBOR_MATCH_COUNT_THRESH=24
EVAL_METHOD="moge_hires_ep3_v5origparams_relative_seg"
OUTPUT_ROOT="${H5_BASE}/${EVAL_METHOD}_h5"

echo "============================================================"
echo "Full pipeline: v5orig_rel (Sobel+rel) on ALL 42 test scenes"
echo "============================================================"
echo "Model:         $MODEL_PATH"
echo "Output:        $OUTPUT_ROOT"
echo "Seg version:   $SEG_VERSION"
echo "Planarity θ:   $THRESHOLD_PLANARITY"
echo "Normal θ:      ${NORMAL_THRESHOLD_DEG}°"
echo "Depth rel θ:   $DEPTH_THRESHOLD_REL"
echo "Match thresh:  $NEIGHBOR_MATCH_COUNT_THRESH"
echo "num_tokens:    $NUM_TOKENS"
echo "============================================================"
echo ""

# --- Job 1: Inference + Segmentation (GPU) ---
INFER_JOB=$(sbatch --parsable \
    --job-name="infer_v5orig_rel" \
    --time=12:00:00 \
    --cpus-per-task=8 \
    --mem-per-cpu=8G \
    --gpus=rtx_3090:1 \
    --output="$LOG_DIR/infer_%j.out" \
    --error="$LOG_DIR/infer_%j.err" \
    --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
python $PROJECT_ROOT/inference/planarity/inference_to_h5.py \
    --model_path $MODEL_PATH \
    --output_root $OUTPUT_ROOT \
    --dataset_dir $DATASET_DIR \
    --rgb_root $RGB_ROOT \
    --split test \
    --batch_size $BATCH_SIZE \
    --num_tokens $NUM_TOKENS \
    --metric_depth \
    --seg_version $SEG_VERSION \
    --threshold_planarity $THRESHOLD_PLANARITY \
    --normal_threshold_deg $NORMAL_THRESHOLD_DEG \
    --depth_threshold $DEPTH_THRESHOLD_REL \
    --neighbor_match_count_thresh $NEIGHBOR_MATCH_COUNT_THRESH
'")
echo "[INFER] Job: $INFER_JOB (GPU inference + segmentation, all 42 scenes)"

# --- Job 2: Evaluation + Aggregation (depends on inference) ---
EVAL_JOB=$(sbatch --parsable \
    --job-name="eval_v5orig_rel" \
    --dependency=afterok:$INFER_JOB \
    --time=12:00:00 \
    --cpus-per-task=16 \
    --mem-per-cpu=4G \
    --output="$LOG_DIR/eval_%j.out" \
    --error="$LOG_DIR/eval_%j.err" \
    --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
python $PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py \
    --methods $EVAL_METHOD
'")
echo "[EVAL] Job: $EVAL_JOB (eval + aggregate, depends on $INFER_JOB)"

echo ""
echo "============================================================"
echo "Pipeline: $INFER_JOB → $EVAL_JOB"
echo "============================================================"
echo "Monitor: squeue -u \$USER"
echo "Logs:    $LOG_DIR"
echo "============================================================"

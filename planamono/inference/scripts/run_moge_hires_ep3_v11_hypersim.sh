#!/bin/bash
# MoGe HiRes ep3 + v11 segmentation — Hypersim Inference → Distributed Eval
# Usage: bash run_moge_hires_ep3_v11_hypersim.sh [NUM_EVAL_JOBS]

set -euo pipefail

NUM_EVAL_JOBS="${1:-5}"
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/inference"
mkdir -p "$LOG_DIR"

MODEL_PATH="/cluster/scratch/ayavuz/moge_mixed_output_476644_fixed_cosLR_singlePhase_mixed_HiRes/model_epoch3.pt"
SPLIT="test"
BATCH_SIZE=8
NUM_TOKENS=1600
METHOD_NAME="moge_hires_ep3_v11seg_metric"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/hypersim/inference/${METHOD_NAME}_h5"

# v11 segmentation parameters
THRESHOLD_PLANARITY=0.2
NORMAL_THRESHOLD_DEG=5.0
DEPTH_THRESHOLD=0.025
NEIGHBOR_MATCH_COUNT_THRESH=18
SEG_VERSION="v11"
ADAPTIVE_FRAC=0.75
MIN_VALID_NEIGHBORS=3
MIN_SEGMENT_PIXELS=50

HYPERSIM_ROOT="/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
PLANE_LABEL_ROOT="/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
PARAMS_ROOT="/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"

echo "=== Submitting Hypersim v11 pipeline ==="
echo "Output:     $OUTPUT_ROOT"
echo "Seg:        $SEG_VERSION, planarity=$THRESHOLD_PLANARITY, merge=False"
echo "Eval jobs:  $NUM_EVAL_JOBS"

# --- Job 1: GPU inference + segmentation ---
INFER_JOB=$(sbatch --parsable \
    --time=24:00:00 \
    --cpus-per-task=8 \
    --mem-per-cpu=16G \
    --gpus=rtx_3090:1 \
    --output="$LOG_DIR/moge_hires_ep3_v11seg_hypersim_infer_%j.out" \
    --error="$LOG_DIR/moge_hires_ep3_v11seg_hypersim_infer_%j.err" \
    --wrap="
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh || true
conda activate planamono
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p $OUTPUT_ROOT
python $PROJECT_ROOT/inference/planarity/inference_to_h5_hypersim.py \
    --model_path $MODEL_PATH \
    --output_root $OUTPUT_ROOT \
    --hypersim_root $HYPERSIM_ROOT \
    --plane_label_root $PLANE_LABEL_ROOT \
    --params_root $PARAMS_ROOT \
    --split $SPLIT \
    --batch_size $BATCH_SIZE \
    --num_tokens $NUM_TOKENS \
    --threshold_planarity $THRESHOLD_PLANARITY \
    --normal_threshold_deg $NORMAL_THRESHOLD_DEG \
    --depth_threshold $DEPTH_THRESHOLD \
    --neighbor_match_count_thresh $NEIGHBOR_MATCH_COUNT_THRESH \
    --seg_version $SEG_VERSION \
    --adaptive_frac $ADAPTIVE_FRAC \
    --min_valid_neighbors $MIN_VALID_NEIGHBORS \
    --min_segment_pixels $MIN_SEGMENT_PIXELS
")
echo "[SUBMITTED] Inference job: $INFER_JOB"

# --- Job 2: CPU evaluation (array job, depends on inference) ---
TOTAL_SCENES=70
SCENES_PER_JOB=$(( (TOTAL_SCENES + NUM_EVAL_JOBS - 1) / NUM_EVAL_JOBS ))

EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:$INFER_JOB \
    --array=0-$((NUM_EVAL_JOBS - 1)) \
    --time=8:00:00 \
    --cpus-per-task=16 \
    --mem-per-cpu=4G \
    --output="$LOG_DIR/eval_v11_hypersim_%A_%a.out" \
    --error="$LOG_DIR/eval_v11_hypersim_%A_%a.err" \
    --wrap="
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh || true
conda activate planamono
export HDF5_USE_FILE_LOCKING=FALSE
SCENE_START=\$(( SLURM_ARRAY_TASK_ID * $SCENES_PER_JOB ))
SCENE_END=\$(( SCENE_START + $SCENES_PER_JOB ))
if [ \$SCENE_END -gt $TOTAL_SCENES ]; then SCENE_END=$TOTAL_SCENES; fi
echo \"Eval scenes [\$SCENE_START:\$SCENE_END]\"
python $PROJECT_ROOT/evaluation/quantitative/evaluate_hypersim_all_baselines.py \
    --methods $METHOD_NAME \
    --scene-start \$SCENE_START \
    --scene-end \$SCENE_END
")
echo "[SUBMITTED] Evaluation array job: $EVAL_JOB (${NUM_EVAL_JOBS} tasks, depends on $INFER_JOB)"

# --- Job 3: Aggregate results ---
AGG_JOB=$(sbatch --parsable \
    --dependency=afterok:$EVAL_JOB \
    --time=0:10:00 \
    --cpus-per-task=2 \
    --mem-per-cpu=4G \
    --output="$LOG_DIR/agg_v11_hypersim_%j.out" \
    --error="$LOG_DIR/agg_v11_hypersim_%j.err" \
    --wrap="
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh || true
conda activate planamono
export HDF5_USE_FILE_LOCKING=FALSE
python $PROJECT_ROOT/evaluation/quantitative/evaluate_hypersim_all_baselines.py \
    --methods $METHOD_NAME \
    --aggregate-only
")
echo "[SUBMITTED] Aggregation job: $AGG_JOB (depends on $EVAL_JOB)"

echo ""
echo "Pipeline: $INFER_JOB (infer) → $EVAL_JOB (eval x${NUM_EVAL_JOBS}) → $AGG_JOB (agg)"

#!/bin/bash
# Submit loss ablation (Focal, DICE, Mixed) inference + eval on ScanNet++.
# All use epoch 2 checkpoints with v5_relative segmentation params matching
# the BCE baseline: plan=0.3, norm=5°, match=8, depth_rel=0.025.
#
# Usage:
#   bash submit_loss_ablation_4ds_scannetpp.sh

set -euo pipefail

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/loss_ablation_4ds_scannetpp"
mkdir -p "$LOG_DIR"

# ---- Shared segmentation parameters (same as BCE baseline) ----
SEG_VERSION="v5_relative"
THRESHOLD_PLANARITY=0.3
NORMAL_THRESHOLD_DEG=5.0
DEPTH_THRESHOLD=0.025
NEIGHBOR_MATCH_COUNT_THRESH=8

# ---- ScanNet++ dataset config ----
DATASET_DIR="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
RGB_ROOT="/cluster/project/cvg/Shared_datasets/scannet++/data"
OUTPUT_BASE="/cluster/scratch/aoezkan/planeseg/scannetpp/inference"
SPLIT="test"
BATCH_SIZE=8
NUM_TOKENS=1600

# ---- Ablation configs: name:model_path:method_key ----
declare -a ABLATIONS=(
    "focal:/cluster/scratch/ayavuz/moge_HIRES_4datasets_focal/model_epoch2.pt:moge_hires_4ds_focal_ep2_v5_relative_seg"
    "dice:/cluster/scratch/ayavuz/moge_HIRES_4datasets_dice/model_epoch2.pt:moge_hires_4ds_dice_ep2_v5_relative_seg"
    "mixed:/cluster/scratch/ayavuz/moge_HIRES_4datasets_mixed/model_epoch2.pt:moge_hires_4ds_mixed_ep2_v5_relative_seg"
)

echo "============================================================"
echo "Loss Ablation — 4ds ep2 — ScanNet++"
echo "============================================================"
echo "Seg params: plan=${THRESHOLD_PLANARITY}, norm=${NORMAL_THRESHOLD_DEG}°, match=${NEIGHBOR_MATCH_COUNT_THRESH}, depth_rel=${DEPTH_THRESHOLD}"
echo "Ablations:  ${#ABLATIONS[@]} (Focal, DICE, Mixed)"
echo "============================================================"
echo ""

for ablation_spec in "${ABLATIONS[@]}"; do
    IFS=':' read -r LOSS_NAME MODEL_PATH METHOD <<< "$ablation_spec"
    OUTPUT_ROOT="${OUTPUT_BASE}/${METHOD}_h5"

    echo "==== ${LOSS_NAME^^} ===="
    echo "  model:  $MODEL_PATH"
    echo "  output: $OUTPUT_ROOT"
    echo "  method: $METHOD"

    # ── Inference (GPU) ──
    JOB_INFER=$(sbatch --parsable \
        --job-name="infer_4ds_${LOSS_NAME}" \
        --time=12:00:00 \
        --cpus-per-task=8 \
        --mem-per-cpu=16G \
        --gpus=rtx_3090:1 \
        --output="$LOG_DIR/infer_${LOSS_NAME}_%j.out" \
        --error="$LOG_DIR/infer_${LOSS_NAME}_%j.err" \
        --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
mkdir -p $OUTPUT_ROOT
python $PROJECT_ROOT/inference/planarity/inference_to_h5.py \
    --model_path $MODEL_PATH \
    --output_root $OUTPUT_ROOT \
    --dataset_dir $DATASET_DIR \
    --rgb_root $RGB_ROOT \
    --split $SPLIT \
    --batch_size $BATCH_SIZE \
    --num_tokens $NUM_TOKENS \
    --seg_version $SEG_VERSION \
    --threshold_planarity $THRESHOLD_PLANARITY \
    --normal_threshold_deg $NORMAL_THRESHOLD_DEG \
    --depth_threshold $DEPTH_THRESHOLD \
    --neighbor_match_count_thresh $NEIGHBOR_MATCH_COUNT_THRESH
'")
    echo "  [INFER] $JOB_INFER"

    # ── Evaluation (CPU, depends on inference) ──
    JOB_EVAL=$(sbatch --parsable \
        --job-name="eval_4ds_${LOSS_NAME}" \
        --dependency=afterok:$JOB_INFER \
        --time=12:00:00 \
        --cpus-per-task=16 \
        --mem-per-cpu=4G \
        --output="$LOG_DIR/eval_${LOSS_NAME}_%j.out" \
        --error="$LOG_DIR/eval_${LOSS_NAME}_%j.err" \
        --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
python $PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py --methods $METHOD
'")
    echo "  [EVAL]  $JOB_EVAL (after $JOB_INFER)"
    echo "  Chain:  $JOB_INFER → $JOB_EVAL"
    echo ""
done

echo "============================================================"
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: $LOG_DIR"
echo "============================================================"

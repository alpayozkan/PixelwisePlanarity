#!/bin/bash
# MoGe HiRes 4DS ep2 — 7-Scenes Inference (v5_relative segmentation)
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_4ds_ep2_sevenscenes_infer_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_4ds_ep2_sevenscenes_infer_%j.err

set -euo pipefail

MODEL_PATH="/cluster/scratch/ayavuz/moge_HIRES_4datasets/model_epoch2.pt"
SPLIT="val"
BATCH_SIZE=16
NUM_TOKENS=1600

# v5_relative segmentation params (matches submit_v5_relative_4ds_all_datasets.sh
# and segmentation_configs_ablation.py:config3_default)
SEG_VERSION="v5_relative"
THRESHOLD_PLANARITY=0.3
NORMAL_THRESHOLD_DEG=5.0
DEPTH_THRESHOLD=0.025  # relative fraction of center depth
NEIGHBOR_MATCH_COUNT_THRESH=8

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/sevenscenes/inference/moge_hires_4ds_ep2_v5_relative_seg_h5"
DATA_ROOT="/cluster/scratch/aoezkan/planeseg/dataset/sevenscenes_plane"

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "7-Scenes inference -> $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

python "$PROJECT_ROOT/inference/planarity/inference_to_h5_sevenscenes.py" \
    --model_path "$MODEL_PATH" \
    --output_root "$OUTPUT_ROOT" \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --num_tokens "$NUM_TOKENS" \
    --seg_version "$SEG_VERSION" \
    --threshold_planarity "$THRESHOLD_PLANARITY" \
    --normal_threshold_deg "$NORMAL_THRESHOLD_DEG" \
    --depth_threshold "$DEPTH_THRESHOLD" \
    --neighbor_match_count_thresh "$NEIGHBOR_MATCH_COUNT_THRESH"

echo "[DONE] 7-Scenes inference"

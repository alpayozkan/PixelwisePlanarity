#!/bin/bash
# Synthia Inference: MoGe Planarity + Segmentation -> H5
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/synthia_infer_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/synthia_infer_%j.err

set -euo pipefail

MODEL_PATH="/cluster/scratch/ayavuz/moge_mixed_output_bce_476644_fixed/model_epoch6.pt"
SPLIT="test"
BATCH_SIZE=16
NUM_TOKENS=1024
THRESHOLD_PLANARITY=0.6
NORMAL_THRESHOLD_DEG=10.0
DEPTH_THRESHOLD=0.05
NEIGHBOR_MATCH_COUNT_THRESH=24
SEG_VERSION="v5"

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/synthia/inference/moge_ours_h5"
DATA_ROOT="/cluster/scratch/ayavuz/dataset/synthia_planes"

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Synthia inference -> $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

python "$PROJECT_ROOT/inference/planarity/inference_to_h5_synthia.py" \
    --model_path "$MODEL_PATH" \
    --output_root "$OUTPUT_ROOT" \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --num_tokens "$NUM_TOKENS" \
    --threshold_planarity "$THRESHOLD_PLANARITY" \
    --normal_threshold_deg "$NORMAL_THRESHOLD_DEG" \
    --depth_threshold "$DEPTH_THRESHOLD" \
    --neighbor_match_count_thresh "$NEIGHBOR_MATCH_COUNT_THRESH" \
    --seg_version "$SEG_VERSION"

echo "[DONE] Synthia inference"

#!/bin/bash
# MoGe Mixed BCE 476644 (epoch 6, v6 seg) — Hypersim Inference only
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_mixed_bce_476644_ep6_v6seg_hypersim_infer_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_mixed_bce_476644_ep6_v6seg_hypersim_infer_%j.err

set -euo pipefail

MODEL_PATH="/cluster/scratch/ayavuz/moge_mixed_output_bce_476644_fixed/model_epoch6.pt"
SPLIT="test"
BATCH_SIZE=8
NUM_TOKENS=1024
THRESHOLD_PLANARITY=0.6
NORMAL_THRESHOLD_DEG=10.0
DEPTH_THRESHOLD=0.02
NEIGHBOR_MATCH_COUNT_THRESH=18
SEG_VERSION="v6"

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/hypersim/inference/moge_mixed_bce_476644_ep6_v6seg_h5"
HYPERSIM_ROOT="/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
PLANE_LABEL_ROOT="/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
PARAMS_ROOT="/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Hypersim inference (v6 seg) → $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

python "$PROJECT_ROOT/inference/planarity/inference_to_h5_hypersim.py" \
    --model_path "$MODEL_PATH" \
    --output_root "$OUTPUT_ROOT" \
    --hypersim_root "$HYPERSIM_ROOT" \
    --plane_label_root "$PLANE_LABEL_ROOT" \
    --params_root "$PARAMS_ROOT" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --num_tokens "$NUM_TOKENS" \
    --threshold_planarity "$THRESHOLD_PLANARITY" \
    --normal_threshold_deg "$NORMAL_THRESHOLD_DEG" \
    --depth_threshold "$DEPTH_THRESHOLD" \
    --neighbor_match_count_thresh "$NEIGHBOR_MATCH_COUNT_THRESH" \
    --seg_version "$SEG_VERSION"

echo "[DONE] Hypersim inference (v6 seg)"

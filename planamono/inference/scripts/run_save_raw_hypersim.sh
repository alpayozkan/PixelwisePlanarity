#!/bin/bash
# Stage 1: MoGe Inference → Raw H5 (Hypersim)
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/save_raw_hypersim_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/save_raw_hypersim_%j.err

set -euo pipefail

MODEL_PATH="/cluster/scratch/ayavuz/moge_mixed_output_476644_fixed_cosLR_singlePhase_mixed_HiRes/model_epoch3.pt"
SPLIT="test"
BATCH_SIZE=8
NUM_TOKENS=1600

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/hypersim/inference_raw/moge_hires_ep3_raw"
HYPERSIM_ROOT="/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
PLANE_LABEL_ROOT="/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
PARAMS_ROOT="/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Stage 1: Hypersim MoGe inference → $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

python "$PROJECT_ROOT/inference/planarity/save_moge_raw_hypersim.py" \
    --model_path "$MODEL_PATH" \
    --output_root "$OUTPUT_ROOT" \
    --hypersim_root "$HYPERSIM_ROOT" \
    --plane_label_root "$PLANE_LABEL_ROOT" \
    --params_root "$PARAMS_ROOT" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --num_tokens "$NUM_TOKENS"

echo "[DONE] Stage 1 Hypersim raw H5 saved"

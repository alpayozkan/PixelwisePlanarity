#!/bin/bash
# Stage 1: MoGe Inference → Raw H5 (ScanNet++)
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/save_raw_scannetpp_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/save_raw_scannetpp_%j.err

set -euo pipefail

MODEL_PATH="/cluster/scratch/ayavuz/moge_HIRES_4datasets/model_epoch2.pt"
SPLIT="test"
BATCH_SIZE=16
NUM_TOKENS=1600
MAX_SCENES=10

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference_raw/moge_hires_4ds_ep2_raw"
DATASET_DIR="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
RGB_ROOT="/cluster/project/cvg/Shared_datasets/scannet++/data"

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Stage 1: ScanNet++ MoGe inference → $OUTPUT_ROOT (max $MAX_SCENES scenes)"
mkdir -p "$OUTPUT_ROOT"

python "$PROJECT_ROOT/inference/planarity/save_moge_raw.py" \
    --model_path "$MODEL_PATH" \
    --output_root "$OUTPUT_ROOT" \
    --dataset_dir "$DATASET_DIR" \
    --rgb_root "$RGB_ROOT" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --num_tokens "$NUM_TOKENS" \
    --max_scenes "$MAX_SCENES" \
    --metric_depth

echo "[DONE] Stage 1 ScanNet++ raw H5 saved"

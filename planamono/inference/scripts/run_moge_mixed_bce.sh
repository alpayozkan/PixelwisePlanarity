#!/bin/bash
# MoGe Mixed BCE Inference → H5
# Run: ./run_moge_mixed_bce.sh

# SLURM options (uncomment for cluster submission with sbatch)
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_mixed_bce_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_mixed_bce_%j.err

# Configuration
# MODEL_PATH="/cluster/scratch/ayavuz/moge_mixed_output_bce_fixed/last_model.pt"
# MODEL_PATH="/cluster/scratch/aoezkan/moge_runs/backup2/moge_scannetpp_4heads_v3/final_planarity_4heads_model.pt"
MODEL_PATH="/cluster/scratch/ayavuz/moge_mixed_output_bce_fixed/last_model.pt"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference/moge_mixed_bce_h5"
SPLIT="val"
BATCH_SIZE=8

# Project paths
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
DATASET_DIR="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
RGB_ROOT="/cluster/project/cvg/Shared_datasets/scannet++/data"

# Segmentation parameters
THRESHOLD_PLANARITY=0.6
NORMAL_THRESHOLD_DEG=10.0
DEPTH_THRESHOLD=0.05
NEIGHBOR_MATCH_COUNT_THRESH=24
NUM_TOKENS=1024

# Activate conda
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "=========================================="
echo "MoGe Mixed BCE Inference"
echo "=========================================="
echo "Model:      $MODEL_PATH"
echo "Output:     $OUTPUT_ROOT"
echo "Split:      $SPLIT"
echo "Batch size: $BATCH_SIZE"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# Run inference
python "$PROJECT_ROOT/inference/planarity/inference_to_h5.py" \
    --model_path "$MODEL_PATH" \
    --output_root "$OUTPUT_ROOT" \
    --dataset_dir "$DATASET_DIR" \
    --rgb_root "$RGB_ROOT" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --num_tokens "$NUM_TOKENS" \
    --threshold_planarity "$THRESHOLD_PLANARITY" \
    --normal_threshold_deg "$NORMAL_THRESHOLD_DEG" \
    --depth_threshold "$DEPTH_THRESHOLD" \
    --neighbor_match_count_thresh "$NEIGHBOR_MATCH_COUNT_THRESH"

echo "[DONE] Results saved to $OUTPUT_ROOT"

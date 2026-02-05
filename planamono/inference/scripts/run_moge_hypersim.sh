#!/bin/bash
# MoGe Hypersim Inference → H5
# Run: ./run_moge_hypersim.sh

# SLURM options (uncomment for cluster submission with sbatch)
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hypersim_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hypersim_%j.err

# Configuration
MODEL_PATH="/cluster/scratch/aoezkan/moge_runs/backup2/moge_scannetpp_4heads_v3/final_planarity_4heads_model.pt"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/hypersim/inference/moge_ours_h5"
SPLIT="val"
BATCH_SIZE=8

# Dataset paths
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
HYPERSIM_ROOT="/cluster/scratch/ayavuz/dataset/Hypersim_merged"
PLANE_LABEL_ROOT="/cluster/scratch/ayavuz/dataset/Hypersim_rendered"
PARAMS_ROOT="/cluster/scratch/ayavuz/dataset/Hypersim_params"

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
echo "MoGe Hypersim Inference"
echo "=========================================="
echo "Model:      $MODEL_PATH"
echo "Output:     $OUTPUT_ROOT"
echo "Split:      $SPLIT"
echo "Batch size: $BATCH_SIZE"
echo "=========================================="

# Create output directory and logs
mkdir -p "$OUTPUT_ROOT"
mkdir -p /cluster/scratch/aoezkan/planeseg/logs/inference

# Run inference
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
    --neighbor_match_count_thresh "$NEIGHBOR_MATCH_COUNT_THRESH"

echo "[DONE] Results saved to $OUTPUT_ROOT"

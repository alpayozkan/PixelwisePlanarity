#!/bin/bash
# Evaluation Runner Script
# Runs planarity/segmentation evaluation on ScanNet++

# SLURM options (uncomment for cluster use)
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=logs/evaluation_%j.out
#SBATCH --error=logs/evaluation_%j.err

# Configuration - Original cluster paths as defaults
METHOD="${1:-moge}"
MODEL_PATH="${2:-/cluster/scratch/aoezkan/MoGe/checkpoints/final_planarity_4heads_model.pt}"
RGB_ROOT="${3:-/cluster/project/cvg/Shared_datasets/scannet++/data}"
DATASET_ROOT="${4:-/cluster/scratch/aoezkan/dataset/scannetpp}"
SAVE_DIR="${5:-/cluster/scratch/aoezkan/dataset/scannetpp/results/metrics}"
MAX_SCENES="${6:-5}"
MODEL_SIZE="${7:-large}"
CACHE_DIR="${8:-/cluster/scratch/aoezkan/MoGe/checkpoints}"

# Export environment variables
export MOGE_CACHE_DIR="$CACHE_DIR"

echo "[INFO] Starting evaluation on: $(hostname)"
echo "[INFO] Method: $METHOD"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] RGB root: $RGB_ROOT"
echo "[INFO] Dataset root: $DATASET_ROOT"
echo "[INFO] Save dir: $SAVE_DIR"
echo "[INFO] Max scenes: $MAX_SCENES"

# Create output directory
mkdir -p "$SAVE_DIR"

# Run evaluation
python ../run_evaluation.py \
    --method "$METHOD" \
    --model_path "$MODEL_PATH" \
    --model_size "$MODEL_SIZE" \
    --rgb_root "$RGB_ROOT" \
    --dataset_root "$DATASET_ROOT" \
    --save_dir "$SAVE_DIR" \
    --max_scenes "$MAX_SCENES" \
    --cache_dir "$CACHE_DIR"

if [[ $? -eq 0 ]]; then
    echo "[SUCCESS] Evaluation completed"
else
    echo "[ERROR] Evaluation failed"
    exit 1
fi

echo "[INFO] Results saved to: $SAVE_DIR"

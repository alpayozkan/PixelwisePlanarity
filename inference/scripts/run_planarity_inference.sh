#!/bin/bash
# MoGe Planarity Inference Script
# Runs planarity prediction on images using trained MoGe model

# SLURM options (uncomment for cluster use)
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=logs/planarity_inference_%j.out
#SBATCH --error=logs/planarity_inference_%j.err

# Configuration - Original cluster paths as defaults
MODEL_PATH="${1:-/cluster/scratch/aoezkan/MoGe/checkpoints/final_planarity_4heads_model.pt}"
INPUT_DIR="${2:-/path/to/input/images}"
OUTPUT_DIR="${3:-/path/to/output}"
MODEL_SIZE="${4:-large}"
CACHE_DIR="${5:-/cluster/scratch/aoezkan/MoGe/checkpoints}"

# Validate inputs
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ERROR] Model checkpoint not found: $MODEL_PATH"
    echo "Usage: $0 <model_path> <input_dir> <output_dir> [model_size] [cache_dir]"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "[ERROR] Input directory not found: $INPUT_DIR"
    exit 1
fi

# Export environment variables for the Python script
export MOGE_CACHE_DIR="$CACHE_DIR"

echo "[INFO] Starting MoGe planarity inference on: $(hostname)"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Model size: $MODEL_SIZE"
echo "[INFO] Input: $INPUT_DIR"
echo "[INFO] Output: $OUTPUT_DIR"
echo "[INFO] Cache dir: $CACHE_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
python ../planarity/run_inference.py \
    --model_path "$MODEL_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_size "$MODEL_SIZE" \
    --save_raw \
    --save_visualization

if [[ $? -eq 0 ]]; then
    echo "[SUCCESS] Inference completed"
else
    echo "[ERROR] Inference failed"
    exit 1
fi

echo "[INFO] Results saved to: $OUTPUT_DIR"

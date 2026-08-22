#!/bin/bash
# Segmentation Prediction Script
# Runs full pipeline: RGB → MoGe → plane segmentation

# SLURM options (uncomment for cluster use)
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=logs/segmentation_%j.out
#SBATCH --error=logs/segmentation_%j.err

# Get script directory (works from any location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Configuration - checkpoint default resolves from paths.py
MODEL_PATH="${1:-$(python -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from pxwplanar.paths import planarity_model_path; print(planarity_model_path)")}"
INPUT_ROOT="${2:?usage: run_segmentation.sh <model_path> <input_root> <output_root> [frame_skip]}"
OUTPUT_ROOT="${3:?output_root required}"
FRAME_SKIP="${4:-50}"

# Validate inputs
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ERROR] Model checkpoint not found: $MODEL_PATH"
    echo "Usage: $0 <model_path> <input_root> <output_root> [frame_skip]"
    exit 1
fi

if [[ ! -d "$INPUT_ROOT" ]]; then
    echo "[ERROR] Input directory not found: $INPUT_ROOT"
    exit 1
fi

echo "[INFO] Starting segmentation prediction on: $(hostname)"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Input: $INPUT_ROOT"
echo "[INFO] Output: $OUTPUT_ROOT"
echo "[INFO] Frame skip: $FRAME_SKIP"

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# Run prediction
python "$SCRIPT_DIR/../segmentation/predict.py" \
    --model_path "$MODEL_PATH" \
    --input_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --frame_skip "$FRAME_SKIP" \
    --save_visualization

if [[ $? -eq 0 ]]; then
    echo "[SUCCESS] Segmentation completed"
else
    echo "[ERROR] Segmentation failed"
    exit 1
fi

echo "[INFO] Results saved to: $OUTPUT_ROOT"

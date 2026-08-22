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

# Get script directory (works from any location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Configuration - checkpoint default resolves from paths.py
MODEL_PATH="${1:-$(python -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from pxwplanar.paths import planarity_model_path; print(planarity_model_path)")}"
INPUT_DIR="${2:?usage: run_planarity_inference.sh <model_path> <input_dir> <output_dir>}"
OUTPUT_DIR="${3:?output_dir required}"

# Validate inputs
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ERROR] Model checkpoint not found: $MODEL_PATH"
    echo "Usage: $0 <model_path> <input_dir> <output_dir>"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "[ERROR] Input directory not found: $INPUT_DIR"
    exit 1
fi

echo "[INFO] Starting MoGe planarity inference on: $(hostname)"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Input: $INPUT_DIR"
echo "[INFO] Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
python "$SCRIPT_DIR/../planarity/run_inference.py" \
    --model_path "$MODEL_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --save_raw \
    --save_binary \
    --save_visualization

if [[ $? -eq 0 ]]; then
    echo "[SUCCESS] Inference completed"
else
    echo "[ERROR] Inference failed"
    exit 1
fi

echo "[INFO] Results saved to: $OUTPUT_DIR"

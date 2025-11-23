#!/bin/bash
# Evaluation Script for Plane Segmentation Methods
# Runs evaluation on ScanNet++ test split

# SLURM options (uncomment for cluster use)
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1

# Configuration
METHOD="${1:-moge}"           # moge, planercnn, gt, monoplane
SPLIT="${2:-test}"            # train, val, test
OUTPUT_DIR="${3:-./results}"

# Model paths (update these)
MOGE_MODEL_PATH="${MOGE_MODEL_PATH:-/path/to/moge/model.pth}"
MOGE_MODEL_SIZE="${MOGE_MODEL_SIZE:-large}"

echo "[INFO] Starting evaluation on: $(hostname)"
echo "[INFO] Method: $METHOD"
echo "[INFO] Split: $SPLIT"
echo "[INFO] Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
python ../run_evaluation.py \
    --method "$METHOD" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --moge_model_path "$MOGE_MODEL_PATH" \
    --moge_model_size "$MOGE_MODEL_SIZE"

if [[ $? -eq 0 ]]; then
    echo "[SUCCESS] Evaluation completed"
    echo "[INFO] Results saved to: $OUTPUT_DIR"
else
    echo "[ERROR] Evaluation failed"
    exit 1
fi

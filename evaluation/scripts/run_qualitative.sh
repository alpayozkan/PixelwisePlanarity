#!/bin/bash
# Qualitative Comparison Video Generation Script
# Generates side-by-side comparison videos

# SLURM options (uncomment for cluster use)
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/qualitative_%j.out
#SBATCH --error=logs/qualitative_%j.err

# Get script directory (works from any location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration - Original cluster paths as defaults
RGB_ROOT="${1:?usage: run_qualitative.sh <rgb_root> <results_root> <gt_root> <output_root> [frame_skip] [max_scenes]}"
RESULTS_ROOT="${2:?results_root required}"
GT_ROOT="${3:?gt_root required}"
OUTPUT_ROOT="${4:?output_root required}"
FRAME_SKIP="${5:-50}"
MAX_SCENES="${6:-}"

echo "[INFO] Starting qualitative comparison on: $(hostname)"
echo "[INFO] RGB root: $RGB_ROOT"
echo "[INFO] Results root: $RESULTS_ROOT"
echo "[INFO] GT root: $GT_ROOT"
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] Frame skip: $FRAME_SKIP"

# Build command
CMD="python \"$SCRIPT_DIR/../qualitative/visualize_comparison.py\" \
    --rgb_root \"$RGB_ROOT\" \
    --results_root \"$RESULTS_ROOT\" \
    --gt_root \"$GT_ROOT\" \
    --output_root \"$OUTPUT_ROOT\" \
    --frame_skip $FRAME_SKIP"

if [[ -n "$MAX_SCENES" ]]; then
    CMD="$CMD --max_scenes $MAX_SCENES"
fi

# Run
eval $CMD

if [[ $? -eq 0 ]]; then
    echo "[SUCCESS] Video generation completed"
else
    echo "[ERROR] Video generation failed"
    exit 1
fi

echo "[INFO] Videos saved to: $OUTPUT_ROOT/comparison_videos"

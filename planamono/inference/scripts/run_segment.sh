#!/bin/bash
# Stage 2: Raw H5 → Segmented Labels H5
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/segment_from_raw_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/segment_from_raw_%j.err

set -euo pipefail

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"

# ---- Dataset selection (uncomment one) ----
DATASET="scannetpp"
RAW_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference_raw/moge_hires_ep3_raw"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference/moge_hires_ep3_raw_v10seg_h5"

# DATASET="hypersim"
# RAW_ROOT="/cluster/scratch/aoezkan/planeseg/hypersim/inference_raw/moge_hires_ep3_raw"
# OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/hypersim/inference/moge_hires_ep3_raw_v10seg_h5"

# ---- Segmentation parameters ----
SEG_VERSION="v10"
THRESHOLD_PLANARITY=0.3
NORMAL_THRESHOLD_DEG=5.0
DEPTH_THRESHOLD=0.025
NEIGHBOR_MATCH_COUNT_THRESH=18
ADAPTIVE_FRAC=0.75
MIN_VALID_NEIGHBORS=3
MIN_SEGMENT_PIXELS=50

# ---- Grid search (set to empty string to disable) ----
GRID_CONFIG=""
# GRID_CONFIG="$PROJECT_ROOT/evaluation/quantitative/grid_search_v10_config.yaml"

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Stage 2: Segmentation → $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

if [ -n "$GRID_CONFIG" ]; then
    echo "Running grid search with config: $GRID_CONFIG"
    python "$PROJECT_ROOT/inference/planarity/segment_from_raw.py" \
        --raw_root "$RAW_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --dataset "$DATASET" \
        --grid_config "$GRID_CONFIG"
else
    echo "Running single config: $SEG_VERSION, planarity=$THRESHOLD_PLANARITY"
    python "$PROJECT_ROOT/inference/planarity/segment_from_raw.py" \
        --raw_root "$RAW_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --dataset "$DATASET" \
        --seg_version "$SEG_VERSION" \
        --threshold_planarity "$THRESHOLD_PLANARITY" \
        --normal_threshold_deg "$NORMAL_THRESHOLD_DEG" \
        --depth_threshold "$DEPTH_THRESHOLD" \
        --neighbor_match_count_thresh "$NEIGHBOR_MATCH_COUNT_THRESH" \
        --adaptive_frac "$ADAPTIVE_FRAC" \
        --min_valid_neighbors "$MIN_VALID_NEIGHBORS" \
        --min_segment_pixels "$MIN_SEGMENT_PIXELS"
fi

echo "[DONE] Stage 2 segmentation complete"

#!/bin/bash
#SBATCH --job-name=hypersim_export
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=logs/hypersim_export_%j.out
#SBATCH --error=logs/hypersim_export_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module load stack/2024-05 gcc/13.2.0 eth_proxy

# Set paths
REPO_ROOT="/cluster/home/ayavuz/PixelwisePlanarity"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Args
SPLIT="${1:-val}"
CHECKPOINT="${2:?Usage: $0 <split> <checkpoint> <output_dir> [data_root] [gt_root] [metadata_csv]}"
OUTPUT_DIR="${3:?Usage: $0 <split> <checkpoint> <output_dir> [data_root] [gt_root] [metadata_csv]}"
DATA_ROOT="${4:-/cluster/scratch/ayavuz/dataset/HP_all/Hypersim}"
GT_ROOT="${5:-$DATA_ROOT}"
METADATA_CSV="${6:-$REPO_ROOT/metadata_camera_parameters.csv}"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Hypersim Inference Export ($SPLIT)"
echo "=============================================="
echo "Start time:   $(date)"
echo "Node:         $(hostname)"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Checkpoint:   $CHECKPOINT"
echo "Output dir:   $OUTPUT_DIR"
echo "Data root:    $DATA_ROOT"
echo "GT root:      $GT_ROOT"
echo "Metadata CSV: $METADATA_CSV"
echo "=============================================="

python $REPO_ROOT/planamono/inference/export_hypersim_val.py \
    --split        "$SPLIT" \
    --checkpoint   "$CHECKPOINT" \
    --output_dir   "$OUTPUT_DIR" \
    --data_root    "$DATA_ROOT" \
    --gt_root      "$GT_ROOT" \
    --metadata_csv "$METADATA_CSV"

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="

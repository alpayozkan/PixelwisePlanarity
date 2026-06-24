#!/bin/bash
#SBATCH --job-name=scannetpp_export
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=logs/scannetpp_export_%j.out
#SBATCH --error=logs/scannetpp_export_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module load stack/2024-05 gcc/13.2.0 eth_proxy

# Set paths
REPO_ROOT="/cluster/home/ayavuz/PixelwisePlanarity"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Args
SPLIT="${1:-val}"
CHECKPOINT="${2:?Usage: $0 <split> <checkpoint> <output_dir> [rgb_root] [gt_root] [split_dir]}"
OUTPUT_DIR="${3:?Usage: $0 <split> <checkpoint> <output_dir> [rgb_root] [gt_root] [split_dir]}"
RGB_ROOT="${4:-/cluster/project/cvg/Shared_datasets/scannet++/data}"
GT_ROOT="${5:-/cluster/scratch/ayavuz/SCANNETPP_BACKUP}"
SPLIT_DIR="${6:-$REPO_ROOT/planamono/splits/scannetpp}"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "ScanNet++ Inference Export ($SPLIT)"
echo "=============================================="
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Checkpoint:  $CHECKPOINT"
echo "Output dir:  $OUTPUT_DIR"
echo "RGB root:    $RGB_ROOT"
echo "GT root:     $GT_ROOT"
echo "Split dir:   $SPLIT_DIR"
echo "=============================================="

python $REPO_ROOT/planamono/inference/export_scannetpp_val.py \
    --split        "$SPLIT" \
    --checkpoint   "$CHECKPOINT" \
    --output_dir   "$OUTPUT_DIR" \
    --rgb_root     "$RGB_ROOT" \
    --gt_root      "$GT_ROOT" \
    --split_dir    "$SPLIT_DIR"

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="

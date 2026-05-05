#!/bin/bash
# Export MoGe inference + GT (planes, sem, depth, intrinsics, pose) on ScanNet++.
#
# Usage:
#   sbatch planamono/inference/scripts/export_scannetpp.sh \
#       <split> <checkpoint> <output_dir> [rgb_root] [gt_root] [split_dir]
#
# Defaults match the standard cluster layout for ayavuz.

#SBATCH --job-name=scannetpp_export
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --output=/cluster/home/ayavuz/PixelwisePlanarity/logs/scannetpp_export_%j.out
#SBATCH --error=/cluster/home/ayavuz/PixelwisePlanarity/logs/scannetpp_export_%j.err

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

SPLIT="${1:-val}"
CHECKPOINT="${2:?Usage: $0 <split> <checkpoint> <output_dir> [rgb_root] [gt_root] [split_dir]}"
OUTPUT_DIR="${3:?Usage: $0 <split> <checkpoint> <output_dir> [rgb_root] [gt_root] [split_dir]}"
RGB_ROOT="${4:-/cluster/project/cvg/Shared_datasets/scannet++/data}"
GT_ROOT="${5:-/cluster/scratch/ayavuz/dataset/scannetpp}"
SPLIT_DIR="${6:-$REPO_ROOT/planamono/splits/scannetpp}"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs"

echo "[INFO] ScanNet++ export on: $(hostname)"
echo "[INFO] Split:       $SPLIT"
echo "[INFO] Checkpoint:  $CHECKPOINT"
echo "[INFO] Output:      $OUTPUT_DIR"
echo "[INFO] RGB root:    $RGB_ROOT"
echo "[INFO] GT root:     $GT_ROOT"
echo "[INFO] Split dir:   $SPLIT_DIR"

python "$REPO_ROOT/planamono/inference/export_scannetpp_val.py" \
    --split        "$SPLIT" \
    --checkpoint   "$CHECKPOINT" \
    --output_dir   "$OUTPUT_DIR" \
    --rgb_root     "$RGB_ROOT" \
    --gt_root      "$GT_ROOT" \
    --split_dir    "$SPLIT_DIR"

echo "[INFO] Done."

#!/bin/bash
# Export MoGe inference (depth, normals, planarity, mask) on Hypersim,
# along with GT planes, GT depth (Z-converted), and per-frame intrinsics.
#
# Usage:
#   sbatch planamono/inference/scripts/export_hypersim.sh \
#       <split> <checkpoint> <output_dir> [data_root] [gt_root] [metadata_csv]
#
# Defaults below match the standard cluster layout for ayavuz.

#SBATCH --job-name=hypersim_export
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --output=/cluster/home/ayavuz/PixelwisePlanarity/logs/hypersim_export_%j.out
#SBATCH --error=/cluster/home/ayavuz/PixelwisePlanarity/logs/hypersim_export_%j.err

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

SPLIT="${1:-val}"
CHECKPOINT="${2:?Usage: $0 <split> <checkpoint> <output_dir> [data_root] [gt_root] [metadata_csv]}"
OUTPUT_DIR="${3:?Usage: $0 <split> <checkpoint> <output_dir> [data_root] [gt_root] [metadata_csv]}"
DATA_ROOT="${4:-/cluster/scratch/ayavuz/dataset/HP_all/Hypersim}"
GT_ROOT="${5:-$DATA_ROOT}"
METADATA_CSV="${6:-$REPO_ROOT/metadata_camera_parameters.csv}"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs"

echo "[INFO] Hypersim export on: $(hostname)"
echo "[INFO] Split:       $SPLIT"
echo "[INFO] Checkpoint:  $CHECKPOINT"
echo "[INFO] Output:      $OUTPUT_DIR"
echo "[INFO] Data root:   $DATA_ROOT"
echo "[INFO] GT root:     $GT_ROOT"
echo "[INFO] Metadata:    $METADATA_CSV"

python "$REPO_ROOT/planamono/inference/export_hypersim_val.py" \
    --split        "$SPLIT" \
    --checkpoint   "$CHECKPOINT" \
    --output_dir   "$OUTPUT_DIR" \
    --data_root    "$DATA_ROOT" \
    --gt_root      "$GT_ROOT" \
    --metadata_csv "$METADATA_CSV"

echo "[INFO] Done."

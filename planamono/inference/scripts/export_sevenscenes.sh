#!/bin/bash
#SBATCH --job-name=sevenscenes_export
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --output=logs/sevenscenes_export_%j.out
#SBATCH --error=logs/sevenscenes_export_%j.err

mkdir -p logs

module load stack/2024-05 gcc/13.2.0 eth_proxy

REPO_ROOT="/cluster/home/ayavuz/PixelwisePlanarity"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CHECKPOINT="${1:?Usage: $0 <checkpoint> <output_dir> [json_file] [npz_root]}"
OUTPUT_DIR="${2:?Usage: $0 <checkpoint> <output_dir> [json_file] [npz_root]}"
JSON_FILE="${3:-/cluster/scratch/ayavuz/dataset/sevenscenes_plane/sevenscenes_plane_len758_val.json}"
NPZ_ROOT="${4:-/cluster/scratch/ayavuz/dataset/sevenscenes_plane}"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "MoGe SevenScenes Export"
echo "=============================================="
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Checkpoint:  $CHECKPOINT"
echo "Output:      $OUTPUT_DIR"
echo "JSON:        $JSON_FILE"
echo "NPZ root:    $NPZ_ROOT"
echo "=============================================="

python $REPO_ROOT/planamono/inference/export_sevenscenes_val.py \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --json_file  "$JSON_FILE" \
    --npz_root   "$NPZ_ROOT"

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="

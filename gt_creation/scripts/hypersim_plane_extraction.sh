#!/bin/bash
# Hypersim Plane Extraction Script
# Extracts planes from Hypersim scenes

# SLURM options (uncomment for cluster use)
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/hypersim_plane_extraction_%j.out
#SBATCH --error=logs/hypersim_plane_extraction_%j.err

# Configuration
SCENE_LIST="${1:-/cluster/home/ayavuz/PixelwisePlanarity/splits/hypersim/all_scenes.txt}"
CONFIG="${2:-/cluster/home/ayavuz/PixelwisePlanarity/gt_creation/configs/hypersim_default.yml}"
INPUT_ROOT="${3:-/cluster/scratch/ayavuz/dataset/Hypersim}"
OUTPUT_ROOT="${4:-/cluster/scratch/ayavuz/dataset/Hypersim_GT}"

# Validate inputs
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Scene list file not found: $SCENE_LIST"
    echo "Usage: $0 <scene_list.txt> [config.yml] [input_root] [output_root]"
    exit 1
fi

echo "[INFO] Starting Hypersim plane extraction on: $(hostname)"
echo "[INFO] Scene list: $SCENE_LIST"
echo "[INFO] Config: $CONFIG"

# Process each scene
while IFS= read -r scene_id; do
    [[ -z "$scene_id" ]] && continue
    [[ "$scene_id" =~ ^#.*$ ]] && continue

    echo "================================================================"
    echo "[INFO] Processing scene: $scene_id"
    python /cluster/home/ayavuz/PixelwisePlanarity/gt_creation/hypersim/scene_runner.py "$scene_id" \
        --config "$CONFIG" \
        --input_root "$INPUT_ROOT" \
        --output_root "$OUTPUT_ROOT"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Completed: $scene_id"
    else
        echo "[ERROR] Failed: $scene_id"
    fi
done < "$SCENE_LIST"

echo "================================================================"
echo "[INFO] Hypersim plane extraction completed."

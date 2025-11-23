#!/bin/bash
# ScanNet++ Plane Raycasting Script
# Raycasts extracted planes to 2D images

# SLURM options (uncomment for cluster use)
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

# Configuration
SCENE_LIST="${1:-scene_list.txt}"
INPUT_ROOT="${2:-/path/to/scannetpp/data}"
PLANE_ROOT="${3:-/path/to/plane/output}"
OUTPUT_ROOT="${4:-/path/to/rendered/output}"

# Validate inputs
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Scene list file not found: $SCENE_LIST"
    echo "Usage: $0 <scene_list.txt> [input_root] [plane_root] [output_root]"
    exit 1
fi

echo "[INFO] Starting plane raycasting on: $(hostname)"
echo "[INFO] Scene list: $SCENE_LIST"

# Process each scene
while IFS= read -r scene_id; do
    [[ -z "$scene_id" ]] && continue
    [[ "$scene_id" =~ ^#.*$ ]] && continue

    echo "================================================================"
    echo "[INFO] Raycasting planes for scene: $scene_id"
    python ../scannetpp/rendering.py "$scene_id" \
        --input_root "$INPUT_ROOT" \
        --plane_root "$PLANE_ROOT" \
        --output_root "$OUTPUT_ROOT"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Completed: $scene_id"
    else
        echo "[ERROR] Failed: $scene_id"
    fi
done < "$SCENE_LIST"

echo "================================================================"
echo "[INFO] Raycasting job completed."

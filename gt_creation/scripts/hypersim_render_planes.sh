#!/bin/bash
# Hypersim Plane Rendering Script
# Renders planes to HDF5 for each camera

# SLURM options (uncomment for cluster use)
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/hypersim_render_%j.out
#SBATCH --error=logs/hypersim_render_%j.err

# Get script directory (works even when called from different locations)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration
SCENE_LIST="${1:-/cluster/home/ayavuz/PixelwisePlanarity/splits/hypersim/all_scenes.txt}"
INPUT_ROOT="${2:-/cluster/scratch/ayavuz/dataset/Hypersim_params}"
PLANE_ROOT="${3:-/cluster/scratch/ayavuz/dataset/Hypersim_ours}"
OUTPUT_ROOT="${4:-/cluster/scratch/ayavuz/dataset/Hypersim_rendered}"
FRAME_SKIP="${5:-1}"

# Validate inputs
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Scene list file not found: $SCENE_LIST"
    echo "Usage: $0 <scene_list.txt> [input_root] [plane_root] [output_root] [frame_skip]"
    exit 1
fi

echo "[INFO] Starting Hypersim plane rendering on: $(hostname)"
echo "[INFO] Scene list: $SCENE_LIST"
echo "[INFO] Input root: $INPUT_ROOT"
echo "[INFO] Plane root: $PLANE_ROOT"
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] Frame skip: $FRAME_SKIP"

# Process each scene
while IFS= read -r scene_id; do
    [[ -z "$scene_id" ]] && continue
    [[ "$scene_id" =~ ^#.*$ ]] && continue

    echo "================================================================"
    echo "[INFO] Rendering planes for scene: $scene_id"
    python "$SCRIPT_DIR/../hypersim/rendering.py" "$scene_id" \
        --input_root "$INPUT_ROOT" \
        --plane_root "$PLANE_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --frame_skip "$FRAME_SKIP"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Completed: $scene_id"
    else
        echo "[ERROR] Failed: $scene_id"
    fi
done < "$SCENE_LIST"

echo "================================================================"
echo "[INFO] Rendering job completed."

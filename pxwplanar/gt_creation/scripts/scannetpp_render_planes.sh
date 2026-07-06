#!/bin/bash
# ScanNet++ Plane Raycasting Script
# Raycasts extracted planes to 2D images

# SLURM options (uncomment for cluster use)
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/scannetpp_render_%j.out
#SBATCH --error=logs/scannetpp_render_%j.err

# Get script directory (works from any location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Configuration - defaults resolve from pxwplanar/paths.py
SCENE_LIST="${1:-scene_list.txt}"
INPUT_ROOT="${2:-$(python -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from pxwplanar.paths import scannetppv2_path; print(scannetppv2_path)")/data}"
PLANE_ROOT="${3:-$(python -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from pxwplanar.paths import scannetpp_plane_path; print(scannetpp_plane_path)")}"
OUTPUT_ROOT="${4:-$(python -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from pxwplanar.paths import scannetpp_rend_plane_path; print(scannetpp_rend_plane_path)")}"
FRAME_SKIP="${5:-25}"

# Validate inputs
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Scene list file not found: $SCENE_LIST"
    echo "Usage: $0 <scene_list.txt> [input_root] [plane_root] [output_root] [frame_skip]"
    exit 1
fi

echo "[INFO] Starting plane raycasting on: $(hostname)"
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
    echo "[INFO] Raycasting planes for scene: $scene_id"
    python "$SCRIPT_DIR/../scannetpp/render_scene.py" "$scene_id" \
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
echo "[INFO] Raycasting job completed."

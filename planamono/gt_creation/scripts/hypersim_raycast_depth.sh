#!/bin/bash
# Hypersim Raycasted Depth Worker Script
# Raycasts plane mesh to produce per-frame depth for each scene
#
# Supports two depth types (DEPTH_TYPE, arg $8):
#   zdepth (default) - z-depth, saved to *_raycast/
#   euclidean        - Euclidean ray distance, saved to *_raycast_euc/

# SLURM options (uncomment for cluster use)
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/hypersim_raycast_depth_%j.out
#SBATCH --error=logs/hypersim_raycast_depth_%j.err

# Configuration
SCENE_LIST="${1:-scene_list.txt}"
PARAMS_ROOT="${2:-/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params}"
PLANE_ROOT="${3:-/cluster/scratch/ayavuz/dataset/hypersim_mesh_ours}"
OUTPUT_ROOT="${4:-/cluster/scratch/aoezkan/planeseg/dataset/hypersim}"
FRAME_SKIP="${5:-1}"
PYTHON_SCRIPT="${6:-/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/gt_creation/hypersim/raycast_depth.py}"
METADATA_CSV="${7:-/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/shared/datasets/metadata_camera_parameters.csv}"
DEPTH_TYPE="${8:-zdepth}"

# Activate conda environment
source activate planeseg 2>/dev/null || conda activate planeseg 2>/dev/null || true

# Validate inputs
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Scene list file not found: $SCENE_LIST"
    echo "Usage: $0 <scene_list.txt> [params_root] [plane_root] [output_root] [frame_skip] [python_script] [metadata_csv] [depth_type]"
    exit 1
fi

echo "[INFO] Starting Hypersim raycasted depth on: $(hostname)"
echo "[INFO] Scene list: $SCENE_LIST"
echo "[INFO] Params root: $PARAMS_ROOT"
echo "[INFO] Plane root: $PLANE_ROOT"
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] Frame skip: $FRAME_SKIP"
echo "[INFO] Depth type: $DEPTH_TYPE"

# Process each scene
while IFS= read -r scene_id; do
    [[ -z "$scene_id" ]] && continue
    [[ "$scene_id" =~ ^#.*$ ]] && continue

    echo "================================================================"
    echo "[INFO] Raycasting depth for scene: $scene_id"
    python "$PYTHON_SCRIPT" "$scene_id" \
        --params_root "$PARAMS_ROOT" \
        --plane_root "$PLANE_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --frame_skip "$FRAME_SKIP" \
        --metadata_csv "$METADATA_CSV" \
        --depth_type "$DEPTH_TYPE"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Completed: $scene_id"
    else
        echo "[ERROR] Failed: $scene_id"
    fi
done < "$SCENE_LIST"

echo "================================================================"
echo "[INFO] Raycasted depth job completed."

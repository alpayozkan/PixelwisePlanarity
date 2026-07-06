#!/bin/bash
# Hypersim Plane Rendering Script
# Renders planes to HDF5 for each camera

# SLURM options (uncomment for cluster use)
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/hypersim_render_%j.out
#SBATCH --error=logs/hypersim_render_%j.err

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

SCENE_LIST="${1:-scene_list.txt}"
# INPUT_ROOT="${2:-/cluster/scratch/ayavuz/dataset/Hypersim_params}"
# PLANE_ROOT="${3:-/cluster/scratch/ayavuz/dataset/Hypersim_ours}"
# PYTHON_SCRIPT="${6:-/cluster/home/ayavuz/PixelwisePlanarity/clean_structure/gt_creation/hypersim/rendering.py}"
INPUT_ROOT="${2:-/cluster/scratch/aoezkan/planeseg/hypersim/Hypersim_params}"
PLANE_ROOT="${3:-/cluster/scratch/aoezkan/planeseg/hypersim/hypersim_mesh_ours}"
# OUTPUT_ROOT="${4:-/cluster/scratch/ayavuz/dataset/Hypersim_rendered}"
OUTPUT_ROOT="${4:-/cluster/scratch/aoezkan/planeseg/dataset/hypersim/plane_rendered}"
FRAME_SKIP="${5:-1}"
PYTHON_SCRIPT="${6:-$SCRIPT_DIR/../hypersim/rendering.py}"
METADATA_CSV="${7:-$REPO_ROOT/shared/datasets/metadata_camera_parameters.csv}"

# Activate conda environment
source activate planeseg 2>/dev/null || conda activate planeseg 2>/dev/null || true

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
    python "$PYTHON_SCRIPT" "$scene_id" \
        --input_root "$INPUT_ROOT" \
        --plane_root "$PLANE_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --frame_skip "$FRAME_SKIP" \
        --metadata_csv "$METADATA_CSV"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Completed: $scene_id"
    else
        echo "[ERROR] Failed: $scene_id"
    fi
done < "$SCENE_LIST"

echo "================================================================"
echo "[INFO] Rendering job completed."

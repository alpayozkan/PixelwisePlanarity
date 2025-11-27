#!/bin/bash
# ScanNet++ Video Generation Script
# Generates visualization videos from rendered plane HDF5 files

# SLURM options (uncomment for cluster use)
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/scannetpp_video_%j.out
#SBATCH --error=logs/scannetpp_video_%j.err

# Get script directory (works from any location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration - Original cluster paths as defaults
SCENE_LIST="${1:-scene_list.txt}"
H5_ROOT="${2:-/cluster/scratch/aoezkan/dataset/scannetpp/plane_ours_gt}"
RGB_ROOT="${3:-/cluster/project/cvg/Shared_datasets/scannet++/data}"
OUTPUT_ROOT="${4:-/cluster/scratch/aoezkan/dataset/scannetpp/visual}"
FPS="${5:-5}"

# Validate inputs
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Scene list file not found: $SCENE_LIST"
    echo "Usage: $0 <scene_list.txt> [h5_root] [rgb_root] [output_root] [fps]"
    exit 1
fi

echo "[INFO] Starting video generation on: $(hostname)"
echo "[INFO] Scene list: $SCENE_LIST"
echo "[INFO] H5 root: $H5_ROOT"
echo "[INFO] RGB root: $RGB_ROOT"
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] FPS: $FPS"

# Process each scene
while IFS= read -r scene_id; do
    [[ -z "$scene_id" ]] && continue
    [[ "$scene_id" =~ ^#.*$ ]] && continue

    echo "================================================================"
    echo "[INFO] Generating video for scene: $scene_id"
    python "$SCRIPT_DIR/../scannetpp/video_gen.py" "$scene_id" \
        --h5_root "$H5_ROOT" \
        --rgb_root "$RGB_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --fps "$FPS"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Completed: $scene_id"
    else
        echo "[ERROR] Failed: $scene_id"
    fi
done < "$SCENE_LIST"

echo "================================================================"
echo "[INFO] Video generation completed."

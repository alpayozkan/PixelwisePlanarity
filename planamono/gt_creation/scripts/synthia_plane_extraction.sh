#!/bin/bash
# SYNTHIA-AL Plane Extraction Script
# Extracts planes from depth + semantic segmentation for all scenes

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/synthia_extract_%j.out
#SBATCH --error=logs/synthia_extract_%j.err

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Configuration
SCENE_LIST="${1:-$SCRIPT_DIR/../../splits/synthia/scene_list.txt}"
CONFIG="${2:-$SCRIPT_DIR/../configs/synthia_default.yml}"
DATA_ROOT="${3:-/cluster/project/cvg/Shared_datasets/SYNTHIA-AL/test}"

echo "[INFO] Starting SYNTHIA plane extraction on: $(hostname)"
echo "[INFO] Scene list: $SCENE_LIST"
echo "[INFO] Config: $CONFIG"
echo "[INFO] Data root: $DATA_ROOT"

# Process each scene
while IFS= read -r scene_name; do
    [[ -z "$scene_name" ]] && continue
    [[ "$scene_name" =~ ^#.*$ ]] && continue

    echo "================================================================"
    echo "[INFO] Processing: $scene_name"

    python -m planamono.gt_creation.synthia.scene_runner \
        --scene_dir "$DATA_ROOT/$scene_name" \
        --config "$CONFIG"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Completed: $scene_name"
    else
        echo "[ERROR] Failed: $scene_name"
    fi
done < "$SCENE_LIST"

echo "================================================================"
echo "[INFO] SYNTHIA plane extraction completed."

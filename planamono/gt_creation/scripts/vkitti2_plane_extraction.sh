#!/bin/bash
# VKITTI2 Plane Extraction Script
# Extracts planes from depth + semantic segmentation for all scene/variant pairs

#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/vkitti2_extract_%j.out
#SBATCH --error=logs/vkitti2_extract_%j.err

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Configuration
SCENE_LIST="${1:-$SCRIPT_DIR/../../splits/vkitti2/scene_list.txt}"
CONFIG="${2:-$SCRIPT_DIR/../configs/vkitti2_default.yml}"

echo "[INFO] Starting VKITTI2 plane extraction on: $(hostname)"
echo "[INFO] Scene list: $SCENE_LIST"
echo "[INFO] Config: $CONFIG"

# Process each scene/variant
while IFS= read -r scene_variant; do
    [[ -z "$scene_variant" ]] && continue
    [[ "$scene_variant" =~ ^#.*$ ]] && continue

    SCENE=$(echo "$scene_variant" | cut -d'/' -f1)
    VARIANT=$(echo "$scene_variant" | cut -d'/' -f2)

    echo "================================================================"
    echo "[INFO] Processing: $SCENE / $VARIANT"

    python -m planamono.gt_creation.vkitti2.scene_runner \
        --scene "$SCENE" \
        --variant "$VARIANT" \
        --config "$CONFIG"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Completed: $SCENE/$VARIANT"
    else
        echo "[ERROR] Failed: $SCENE/$VARIANT"
    fi
done < "$SCENE_LIST"

echo "================================================================"
echo "[INFO] VKITTI2 plane extraction completed."

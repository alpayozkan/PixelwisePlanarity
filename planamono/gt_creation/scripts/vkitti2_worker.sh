#!/bin/bash
# VKITTI2 worker script - processes scene/variant pairs from a list file
# Called by vkitti2_submit_parallel.sh

#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

SCENE_LIST="$1"
CONFIG="$2"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "[INFO] Worker started on: $(hostname)"
echo "[INFO] Scene list: $SCENE_LIST"
echo "[INFO] Config: $CONFIG"

while IFS=$'\t' read -r scene variant; do
    [ -z "$scene" ] && continue

    echo "----------------------------------------------------------------"
    echo "[INFO] Processing: $scene / $variant"

    python -m planamono.gt_creation.vkitti2.scene_runner \
        --scene "$scene" \
        --variant "$variant" \
        --config "$CONFIG"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Completed: $scene/$variant"
    else
        echo "[ERROR] Failed: $scene/$variant"
    fi
done < "$SCENE_LIST"

echo "[INFO] Worker finished."

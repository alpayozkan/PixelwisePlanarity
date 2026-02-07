#!/bin/bash
# SYNTHIA worker script - processes a list of scenes from a file
# Called by synthia_submit_parallel.sh

#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

SCENE_LIST="$1"
CONFIG="$2"
OUTPUT_ROOT="$3"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "[INFO] Worker started on: $(hostname)"
echo "[INFO] Scene list: $SCENE_LIST"
echo "[INFO] Config: $CONFIG"
echo "[INFO] Output root: $OUTPUT_ROOT"

while IFS=$'\t' read -r split scene_dir; do
    [ -z "$scene_dir" ] && continue
    scene_name=$(basename "$scene_dir")

    echo "----------------------------------------------------------------"
    echo "[INFO] Processing: $scene_name ($split)"

    python -m planamono.gt_creation.synthia.scene_runner \
        --scene_dir "$scene_dir" \
        --config "$CONFIG" \
        --output_root_override "$OUTPUT_ROOT/$split"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Completed: $scene_name"
    else
        echo "[ERROR] Failed: $scene_name"
    fi
done < "$SCENE_LIST"

echo "[INFO] Worker finished."

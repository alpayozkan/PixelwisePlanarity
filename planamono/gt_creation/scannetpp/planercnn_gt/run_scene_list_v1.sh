#!/bin/bash
# Process a list of ScanNet++ scenes through PlaneRCNN GT v1 pipeline (fit + render)
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

SCENE_LIST="$(realpath "${1:?Usage: $0 <scene_list.txt> [config_path]}")"
CONFIG="${2:-}"

source activate planeseg 2>/dev/null || conda activate planeseg 2>/dev/null || true

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(realpath "$SCRIPT_DIR/../../../..")"

# Default config if not provided; always resolve to absolute path
if [[ -z "$CONFIG" ]]; then
    CONFIG="$SCRIPT_DIR/planercnn_v1.yml"
else
    CONFIG="$(realpath "$CONFIG")"
fi

if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Scene list not found: $SCENE_LIST"
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config not found: $CONFIG"
    exit 1
fi

TOTAL=$(wc -l < "$SCENE_LIST")
echo "[INFO] Starting PlaneRCNN GT v1 on $(hostname)"
echo "[INFO] Scenes: $TOTAL from $SCENE_LIST"
echo "[INFO] Config: $CONFIG"

i=0
while IFS= read -r scene_id; do
    [[ -z "$scene_id" ]] && continue
    i=$((i + 1))
    echo "================================================================"
    echo "[$i/$TOTAL] Processing scene: $scene_id"

    # Read output_root from config to check for existing outputs
    OUTPUT_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_root'])")
    if [[ -f "$OUTPUT_ROOT/$scene_id/rendered.h5" ]]; then
        echo "[SKIP] rendered.h5 already exists"
        continue
    fi

    cd "$REPO_ROOT"
    python -m planamono.gt_creation.scannetpp.planercnn_gt.run_scene \
        "$scene_id" --config "$CONFIG"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] $scene_id"
    else
        echo "[ERROR] run_scene failed for $scene_id"
    fi
done < "$SCENE_LIST"

echo "================================================================"
echo "[DONE] Processed $i scenes."

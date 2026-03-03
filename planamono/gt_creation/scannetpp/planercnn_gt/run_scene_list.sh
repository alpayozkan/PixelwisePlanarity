#!/bin/bash
# Process a list of ScanNet++ scenes through PlaneRCNN GT pipeline (fit + render)
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

SCENE_LIST="${1:?Usage: $0 <scene_list.txt>}"
MESH_ROOT="${2:-/cluster/scratch/aoezkan/planeseg/dataset_mesh/scannetpp_planercnn}"
OUTPUT_ROOT="${3:-/cluster/scratch/aoezkan/planeseg/dataset/scannetpp_planercnn}"

source activate planeseg 2>/dev/null || conda activate planeseg 2>/dev/null || true

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(realpath "$SCRIPT_DIR/../../../..")"

if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Scene list not found: $SCENE_LIST"
    exit 1
fi

TOTAL=$(wc -l < "$SCENE_LIST")
echo "[INFO] Starting PlaneRCNN GT on $(hostname)"
echo "[INFO] Scenes: $TOTAL from $SCENE_LIST"
echo "[INFO] Mesh root: $MESH_ROOT"
echo "[INFO] Output root: $OUTPUT_ROOT"

i=0
while IFS= read -r scene_id; do
    [[ -z "$scene_id" ]] && continue
    i=$((i + 1))
    echo "================================================================"
    echo "[$i/$TOTAL] Processing scene: $scene_id"

    # Skip if already done
    if [[ -f "$OUTPUT_ROOT/$scene_id/rendered.h5" ]]; then
        echo "[SKIP] rendered.h5 already exists"
        continue
    fi

    cd "$REPO_ROOT"
    python -m planamono.gt_creation.scannetpp.planercnn_gt.fit_planes \
        "$scene_id" --output_root "$MESH_ROOT"

    if [[ $? -ne 0 ]]; then
        echo "[ERROR] fit_planes failed for $scene_id"
        continue
    fi

    python -m planamono.gt_creation.scannetpp.planercnn_gt.render_planes \
        "$scene_id" --mesh_root "$MESH_ROOT" --output_root "$OUTPUT_ROOT"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] $scene_id"
    else
        echo "[ERROR] render_planes failed for $scene_id"
    fi
done < "$SCENE_LIST"

echo "================================================================"
echo "[DONE] Processed $i scenes."

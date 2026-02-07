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
CONFIG="${1:-$SCRIPT_DIR/../configs/synthia_default.yml}"
SYNTHIA_TRAIN="${2:-/cluster/scratch/ayavuz/dataset/synthia/train}"
SYNTHIA_TEST="${3:-/cluster/scratch/ayavuz/dataset/synthia/test}"
TEST_RUN="${4:-false}"  # pass "true" as 4th arg for test run (2 scenes per split)

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "[INFO] Starting SYNTHIA plane extraction on: $(hostname)"
echo "[INFO] Config: $CONFIG"
echo "[INFO] Train root: $SYNTHIA_TRAIN"
echo "[INFO] Test root: $SYNTHIA_TEST"
echo "[INFO] Test run: $TEST_RUN"

# Process all scenes in both train and test
for DATA_ROOT in "$SYNTHIA_TRAIN" "$SYNTHIA_TEST"; do
    SPLIT_NAME=$(basename "$DATA_ROOT")
    echo "================================================================"
    echo "[INFO] Processing $SPLIT_NAME split from: $DATA_ROOT"
    echo "================================================================"

    COUNT=0
    for scene_dir in "$DATA_ROOT"/test5_*; do
        [ ! -d "$scene_dir" ] && continue
        scene_name=$(basename "$scene_dir")

        if [[ "$TEST_RUN" == "true" && $COUNT -ge 2 ]]; then
            echo "[INFO] Test run: skipping remaining scenes in $SPLIT_NAME"
            break
        fi

        echo "----------------------------------------------------------------"
        echo "[INFO] Processing: $scene_name ($SPLIT_NAME)"

        EXTRA_ARGS=""
        if [[ "$TEST_RUN" == "true" ]]; then
            EXTRA_ARGS="--max_frames 5"
        fi

        python -m planamono.gt_creation.synthia.scene_runner \
            --scene_dir "$scene_dir" \
            --config "$CONFIG" \
            $EXTRA_ARGS

        if [[ $? -eq 0 ]]; then
            echo "[SUCCESS] Completed: $scene_name"
        else
            echo "[ERROR] Failed: $scene_name"
        fi
        COUNT=$((COUNT + 1))
    done
done

echo "================================================================"
echo "[INFO] SYNTHIA plane extraction completed."

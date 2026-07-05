#!/bin/bash
# VKITTI2 Plane Extraction Script (single job)
# Extracts planes from depth + semantic segmentation for all scene/variant pairs

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/vkitti2_extract_%j.out
#SBATCH --error=logs/vkitti2_extract_%j.err

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Defaults
CONFIG="$SCRIPT_DIR/../configs/vkitti2_default.yml"
TEST_RUN="false"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --test_run) TEST_RUN="true"; shift ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Read paths from config
DATA_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('data_root') or c.get('rgb_root'))")
OUTPUT_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_root'])")

echo "[INFO] Starting VKITTI2 plane extraction on: $(hostname)"
echo "[INFO] Config: $CONFIG"
echo "[INFO] Data root: $DATA_ROOT"
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] Test run: $TEST_RUN"

SCENES="Scene01 Scene02 Scene06 Scene18 Scene20"
VARIANTS="clone fog morning overcast rain sunset 15-deg-left 15-deg-right 30-deg-left 30-deg-right"

COUNT=0
for SCENE in $SCENES; do
    for VARIANT in $VARIANTS; do
        # Check if scene/variant dir exists
        if [ ! -d "$DATA_ROOT/$SCENE/$VARIANT" ]; then
            continue
        fi

        if [[ "$TEST_RUN" == "true" && $COUNT -ge 2 ]]; then
            echo "[INFO] Test run: stopping after 2 scene/variant pairs"
            exit 0
        fi

        EXTRA_ARGS=""
        if [[ "$TEST_RUN" == "true" ]]; then
            EXTRA_ARGS="--max_frames 5"
        fi

        echo "================================================================"
        echo "[INFO] Processing: $SCENE / $VARIANT"

        python -m gt_creation.vkitti2.scene_runner \
            --scene "$SCENE" \
            --variant "$VARIANT" \
            --config "$CONFIG" \
            $EXTRA_ARGS

        if [[ $? -eq 0 ]]; then
            echo "[SUCCESS] Completed: $SCENE/$VARIANT"
        else
            echo "[ERROR] Failed: $SCENE/$VARIANT"
        fi
        COUNT=$((COUNT + 1))
    done
done

echo "================================================================"
echo "[INFO] VKITTI2 plane extraction completed. Processed $COUNT scene/variant pairs."

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

# Defaults — raw SYNTHIA root from paths.py unless SYNTHIA_RAW is exported
SYNTHIA_RAW="${SYNTHIA_RAW:-$(python -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from pxwplanar.paths import synthia_raw_path; print(synthia_raw_path)")}"
CONFIG="$SCRIPT_DIR/../configs/synthia_default.yml"
SYNTHIA_TRAIN="$SYNTHIA_RAW/train"
SYNTHIA_TEST="$SYNTHIA_RAW/test"
TEST_RUN="false"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --train_root) SYNTHIA_TRAIN="$2"; shift 2 ;;
        --test_root) SYNTHIA_TEST="$2"; shift 2 ;;
        --test_run) TEST_RUN="true"; shift ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

# Fail fast instead of silently matching zero scenes
if [[ ! -d "$SYNTHIA_TRAIN" && ! -d "$SYNTHIA_TEST" ]]; then
    echo "[ERROR] Neither $SYNTHIA_TRAIN nor $SYNTHIA_TEST exists."
    echo "        Set --train_root/--test_root, export SYNTHIA_RAW, or configure paths.synthia_raw_path."
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Read output_root from config
OUTPUT_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_root'])")

echo "[INFO] Starting SYNTHIA plane extraction on: $(hostname)"
echo "[INFO] Config: $CONFIG"
echo "[INFO] Train root: $SYNTHIA_TRAIN"
echo "[INFO] Test root: $SYNTHIA_TEST"
echo "[INFO] Output root: $OUTPUT_ROOT"
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

        python -m pxwplanar.gt_creation.synthia.scene_runner \
            --scene_dir "$scene_dir" \
            --config "$CONFIG" \
            --output_root_override "$OUTPUT_ROOT/$SPLIT_NAME" \
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

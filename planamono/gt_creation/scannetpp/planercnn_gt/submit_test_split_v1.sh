#!/bin/bash
# Submit 10 parallel SLURM jobs for PlaneRCNN GT v1 on ScanNet++ test split
#
# Usage:
#   bash submit_test_split_v1.sh
#   bash submit_test_split_v1.sh /path/to/custom_config.yml

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NUM_JOBS=10

CONFIG="${1:-$SCRIPT_DIR/planercnn_v1.yml}"
SCENE_LIST="$(realpath "$SCRIPT_DIR/../../../splits/scannetpp/test.txt")"
BASH_SCRIPT="$(realpath "$SCRIPT_DIR/run_scene_list_v1.sh")"

if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Test split not found: $SCENE_LIST"
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config not found: $CONFIG"
    exit 1
fi

# Create split + log directories
SPLIT_DIR="$SCRIPT_DIR/splits_v1_tmp"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$SPLIT_DIR" "$LOG_DIR"

TOTAL=$(wc -l < "$SCENE_LIST")
PER_SPLIT=$(( (TOTAL + NUM_JOBS - 1) / NUM_JOBS ))

echo "[INFO] Config: $CONFIG"
echo "[INFO] Test scenes: $TOTAL"
echo "[INFO] Jobs: $NUM_JOBS (~$PER_SPLIT scenes each)"

# Split scene list
split -l "$PER_SPLIT" -d -a 2 "$SCENE_LIST" "$SPLIT_DIR/test_"

# Submit jobs
i=0
for split_file in "$SPLIT_DIR"/test_*; do
    i=$((i + 1))
    count=$(wc -l < "$split_file")
    echo "[INFO] Job $i: $count scenes ($(head -1 "$split_file") .. $(tail -1 "$split_file"))"

    sbatch --job-name="prcnn_v1_$i" \
           --time=24:00:00 \
           --cpus-per-task=4 \
           --mem-per-cpu=8G \
           --output="$LOG_DIR/planercnn_v1_${i}_%j.out" \
           --error="$LOG_DIR/planercnn_v1_${i}_%j.err" \
           "$BASH_SCRIPT" \
           "$split_file" \
           "$CONFIG"
done

echo "[INFO] Submitted $i jobs"
echo "[INFO] Splits: $SPLIT_DIR"
echo "[INFO] Logs: $LOG_DIR"

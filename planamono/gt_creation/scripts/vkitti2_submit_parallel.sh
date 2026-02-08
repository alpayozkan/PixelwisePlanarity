#!/bin/bash
# Submit parallel SLURM jobs for VKITTI2 plane extraction
#
# Usage:
#   bash vkitti2_submit_parallel.sh                    # default 10 jobs
#   bash vkitti2_submit_parallel.sh --num_jobs 20

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Defaults
CONFIG="$SCRIPT_DIR/../configs/vkitti2_default.yml"
NUM_JOBS=10

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --num_jobs) NUM_JOBS="$2"; shift 2 ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

# Read data root from config
DATA_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('data_root') or c.get('rgb_root'))")

echo "=============================================="
echo "VKITTI2 Parallel Plane Extraction"
echo "=============================================="
echo "Config:    $CONFIG"
echo "Data root: $DATA_ROOT"
echo "Num jobs:  $NUM_JOBS"

# Create temp dir for scene lists
LISTS_DIR="$REPO_ROOT/planamono/gt_creation/scripts/vkitti2_job_lists"
mkdir -p "$LISTS_DIR"
mkdir -p "$REPO_ROOT/logs"

# Collect all scene/variant pairs
ALL_SCENES="$LISTS_DIR/all_scenes.txt"
> "$ALL_SCENES"

SCENES="Scene01 Scene02 Scene06 Scene18 Scene20"
VARIANTS="clone fog morning overcast rain sunset 15-deg-left 15-deg-right 30-deg-left 30-deg-right"

for SCENE in $SCENES; do
    for VARIANT in $VARIANTS; do
        if [ -d "$DATA_ROOT/$SCENE/$VARIANT" ]; then
            echo -e "${SCENE}\t${VARIANT}" >> "$ALL_SCENES"
        fi
    done
done

TOTAL=$(wc -l < "$ALL_SCENES" | tr -d ' ')
echo "Total scene/variant pairs: $TOTAL"

# Split into N job files
PAIRS_PER_JOB=$(( (TOTAL + NUM_JOBS - 1) / NUM_JOBS ))
echo "Pairs per job: ~$PAIRS_PER_JOB"

split -l "$PAIRS_PER_JOB" -d -a 3 "$ALL_SCENES" "$LISTS_DIR/job_"

# Submit each job
JOB_COUNT=0
for job_file in "$LISTS_DIR"/job_*; do
    N_PAIRS=$(wc -l < "$job_file" | tr -d ' ')
    [ "$N_PAIRS" -eq 0 ] && continue

    echo "Submitting job $JOB_COUNT: $N_PAIRS pairs ($job_file)"
    sbatch --job-name="vkitti2_${JOB_COUNT}" \
           --output="$REPO_ROOT/logs/vkitti2_job${JOB_COUNT}_%j.out" \
           --error="$REPO_ROOT/logs/vkitti2_job${JOB_COUNT}_%j.err" \
           "$SCRIPT_DIR/vkitti2_worker.sh" "$job_file" "$CONFIG"

    JOB_COUNT=$((JOB_COUNT + 1))
done

echo "=============================================="
echo "Submitted $JOB_COUNT jobs. Monitor with: squeue -u \$USER"
echo "=============================================="

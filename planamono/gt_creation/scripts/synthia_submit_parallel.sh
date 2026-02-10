#!/bin/bash
# Submit parallel SLURM jobs for SYNTHIA plane extraction
#
# Usage:
#   bash synthia_submit_parallel.sh                    # default 10 jobs
#   bash synthia_submit_parallel.sh --num_jobs 20
#   bash synthia_submit_parallel.sh --test_run         # 2 scenes total, 1 job

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Defaults
CONFIG="$SCRIPT_DIR/../configs/synthia_default.yml"
SYNTHIA_TRAIN="/cluster/scratch/ayavuz/dataset/synthia/train"
SYNTHIA_TEST="/cluster/scratch/ayavuz/dataset/synthia/test"
NUM_JOBS=10

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --train_root) SYNTHIA_TRAIN="$2"; shift 2 ;;
        --test_root) SYNTHIA_TEST="$2"; shift 2 ;;
        --num_jobs) NUM_JOBS="$2"; shift 2 ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

# Read output_root from config
OUTPUT_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_root'])")

echo "=============================================="
echo "SYNTHIA Parallel Plane Extraction"
echo "=============================================="
echo "Config:      $CONFIG"
echo "Train root:  $SYNTHIA_TRAIN"
echo "Test root:   $SYNTHIA_TEST"
echo "Output root: $OUTPUT_ROOT"
echo "Num jobs:    $NUM_JOBS"

# Create temp dir for scene lists
LISTS_DIR="$REPO_ROOT/planamono/gt_creation/scripts/synthia_job_lists"
mkdir -p "$LISTS_DIR"
mkdir -p "$REPO_ROOT/logs"

# Collect all scenes with their split label
ALL_SCENES="$LISTS_DIR/all_scenes.txt"
> "$ALL_SCENES"

for DATA_ROOT in "$SYNTHIA_TRAIN" "$SYNTHIA_TEST"; do
    SPLIT_NAME=$(basename "$DATA_ROOT")
    for scene_dir in "$DATA_ROOT"/test5_*; do
        [ ! -d "$scene_dir" ] && continue
        echo -e "${SPLIT_NAME}\t${scene_dir}" >> "$ALL_SCENES"
    done
done

TOTAL=$(wc -l < "$ALL_SCENES" | tr -d ' ')
echo "Total scenes: $TOTAL"

# Clean old job files
rm -f "$LISTS_DIR"/job_*

# Split into N job files
SCENES_PER_JOB=$(( (TOTAL + NUM_JOBS - 1) / NUM_JOBS ))
echo "Scenes per job: ~$SCENES_PER_JOB"

split -l "$SCENES_PER_JOB" -d -a 3 "$ALL_SCENES" "$LISTS_DIR/job_"

# Submit each job
JOB_COUNT=0
for job_file in "$LISTS_DIR"/job_*; do
    N_SCENES=$(wc -l < "$job_file" | tr -d ' ')
    [ "$N_SCENES" -eq 0 ] && continue

    echo "Submitting job $JOB_COUNT: $N_SCENES scenes ($job_file)"
    sbatch --job-name="synthia_${JOB_COUNT}" \
           --output="$REPO_ROOT/logs/synthia_job${JOB_COUNT}_%j.out" \
           --error="$REPO_ROOT/logs/synthia_job${JOB_COUNT}_%j.err" \
           "$SCRIPT_DIR/synthia_worker.sh" "$job_file" "$CONFIG" "$OUTPUT_ROOT"

    JOB_COUNT=$((JOB_COUNT + 1))
done

echo "=============================================="
echo "Submitted $JOB_COUNT jobs. Monitor with: squeue -u \$USER"
echo "=============================================="

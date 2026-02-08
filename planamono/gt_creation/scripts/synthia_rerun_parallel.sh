#!/bin/bash
# Submit parallel SLURM jobs for SYNTHIA scenes that need rerunning
#
# Usage:
#   python -m planamono.gt_creation.synthia.check_missing_scenes   # generates scenes_to_rerun.txt
#   bash synthia_rerun_parallel.sh                                  # submits jobs
#   bash synthia_rerun_parallel.sh --num_jobs 5

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Defaults
CONFIG="$SCRIPT_DIR/../configs/synthia_default.yml"
RERUN_FILE="/cluster/scratch/ayavuz/dataset/synthia_planes/scenes_to_rerun.txt"
NUM_JOBS=10

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --rerun_file) RERUN_FILE="$2"; shift 2 ;;
        --num_jobs) NUM_JOBS="$2"; shift 2 ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

if [ ! -f "$RERUN_FILE" ]; then
    echo "[ERROR] Rerun file not found: $RERUN_FILE"
    echo "Run: python -m planamono.gt_creation.synthia.check_missing_scenes"
    exit 1
fi

OUTPUT_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_root'])")

TOTAL=$(wc -l < "$RERUN_FILE" | tr -d ' ')
echo "=============================================="
echo "SYNTHIA Rerun Parallel Submission"
echo "=============================================="
echo "Config:      $CONFIG"
echo "Rerun file:  $RERUN_FILE"
echo "Output root: $OUTPUT_ROOT"
echo "Scenes:      $TOTAL"
echo "Num jobs:    $NUM_JOBS"

if [ "$TOTAL" -eq 0 ]; then
    echo "No scenes to rerun."
    exit 0
fi

# Create job list dir
LISTS_DIR="$REPO_ROOT/planamono/gt_creation/scripts/synthia_job_lists"
mkdir -p "$LISTS_DIR"
mkdir -p "$REPO_ROOT/logs"

# Split into N job files
SCENES_PER_JOB=$(( (TOTAL + NUM_JOBS - 1) / NUM_JOBS ))
echo "Scenes per job: ~$SCENES_PER_JOB"

split -l "$SCENES_PER_JOB" -d -a 3 "$RERUN_FILE" "$LISTS_DIR/rerun_"

# Submit each job
JOB_COUNT=0
for job_file in "$LISTS_DIR"/rerun_*; do
    N_SCENES=$(wc -l < "$job_file" | tr -d ' ')
    [ "$N_SCENES" -eq 0 ] && continue

    echo "Submitting job $JOB_COUNT: $N_SCENES scenes ($job_file)"
    sbatch --job-name="synthia_rerun_${JOB_COUNT}" \
           --output="$REPO_ROOT/logs/synthia_rerun${JOB_COUNT}_%j.out" \
           --error="$REPO_ROOT/logs/synthia_rerun${JOB_COUNT}_%j.err" \
           "$SCRIPT_DIR/synthia_worker.sh" "$job_file" "$CONFIG" "$OUTPUT_ROOT"

    JOB_COUNT=$((JOB_COUNT + 1))
done

echo "=============================================="
echo "Submitted $JOB_COUNT jobs. Monitor with: squeue -u \$USER"
echo "=============================================="

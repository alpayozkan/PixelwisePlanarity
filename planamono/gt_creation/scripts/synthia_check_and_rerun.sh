#!/bin/bash
# Step 1: Submit depth check job
# Step 2: When done, append missing scenes and submit rerun jobs
#
# Usage:
#   bash synthia_check_and_rerun.sh
#   bash synthia_check_and_rerun.sh --num_jobs 5

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

CONFIG="$SCRIPT_DIR/../configs/synthia_default.yml"
NUM_JOBS=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --num_jobs) NUM_JOBS="$2"; shift 2 ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

mkdir -p "$REPO_ROOT/logs"

echo "=============================================="
echo "Step 1: Submitting depth check job"
echo "=============================================="

CHECK_JOB=$(sbatch --parsable \
    --job-name="synthia_check" \
    --output="$REPO_ROOT/logs/synthia_check_%j.out" \
    --error="$REPO_ROOT/logs/synthia_check_%j.err" \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem-per-cpu=4G \
    --wrap="cd $REPO_ROOT && export PYTHONPATH=$REPO_ROOT:\${PYTHONPATH:-} && python -m planamono.gt_creation.synthia.check_depth_limit && python -m planamono.gt_creation.synthia.check_missing_scenes")

echo "Check job submitted: $CHECK_JOB"

echo "=============================================="
echo "Step 2: Submitting rerun jobs (depends on check job)"
echo "=============================================="

sbatch --dependency=afterok:$CHECK_JOB \
    --job-name="synthia_rerun_launcher" \
    --output="$REPO_ROOT/logs/synthia_rerun_launcher_%j.out" \
    --error="$REPO_ROOT/logs/synthia_rerun_launcher_%j.err" \
    --time=00:10:00 \
    --cpus-per-task=1 \
    --mem-per-cpu=2G \
    --wrap="cd $REPO_ROOT && bash $SCRIPT_DIR/synthia_rerun_parallel.sh --num_jobs $NUM_JOBS --config $CONFIG"

echo "=============================================="
echo "Rerun jobs will launch after check job $CHECK_JOB completes."
echo "Monitor with: squeue -u \$USER"
echo "=============================================="

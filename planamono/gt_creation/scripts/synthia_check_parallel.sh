#!/bin/bash
# Submit parallel depth check jobs, then merge results + missing scenes, then submit rerun jobs
#
# Usage:
#   bash synthia_check_parallel.sh
#   bash synthia_check_parallel.sh --num_check_jobs 10 --num_rerun_jobs 3

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

CONFIG="$SCRIPT_DIR/../configs/synthia_default.yml"
SYNTHIA_TRAIN="/cluster/scratch/ayavuz/dataset/synthia/train"
SYNTHIA_TEST="/cluster/scratch/ayavuz/dataset/synthia/test"
OUTPUT_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_root'])")
NUM_CHECK_JOBS=10
NUM_RERUN_JOBS=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_check_jobs) NUM_CHECK_JOBS="$2"; shift 2 ;;
        --num_rerun_jobs) NUM_RERUN_JOBS="$2"; shift 2 ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

LISTS_DIR="$REPO_ROOT/planamono/gt_creation/scripts/synthia_check_lists"
RESULTS_DIR="$REPO_ROOT/planamono/gt_creation/scripts/synthia_check_results"
RERUN_FILE="$OUTPUT_ROOT/scenes_to_rerun.txt"
mkdir -p "$LISTS_DIR" "$RESULTS_DIR" "$REPO_ROOT/logs"

# Collect all scenes
ALL_SCENES="$LISTS_DIR/all_scenes.txt"
> "$ALL_SCENES"
for DATA_ROOT in "$SYNTHIA_TRAIN" "$SYNTHIA_TEST"; do
    SPLIT=$(basename "$DATA_ROOT")
    for scene_dir in "$DATA_ROOT"/test5_*; do
        [ ! -d "$scene_dir" ] && continue
        echo -e "${SPLIT}\t${scene_dir}" >> "$ALL_SCENES"
    done
done

TOTAL=$(wc -l < "$ALL_SCENES" | tr -d ' ')
echo "=============================================="
echo "SYNTHIA Parallel Depth Check + Rerun"
echo "=============================================="
echo "Total scenes: $TOTAL"
echo "Check jobs:   $NUM_CHECK_JOBS"
echo "Rerun jobs:   $NUM_RERUN_JOBS"

# Split into check job files
PER_JOB=$(( (TOTAL + NUM_CHECK_JOBS - 1) / NUM_CHECK_JOBS ))
split -l "$PER_JOB" -d -a 3 "$ALL_SCENES" "$LISTS_DIR/check_"

# Submit check jobs
CHECK_JOB_IDS=""
JOB_COUNT=0
for job_file in "$LISTS_DIR"/check_*; do
    OUT_RESULT="$RESULTS_DIR/result_${JOB_COUNT}.txt"
    JID=$(sbatch --parsable \
        --job-name="synthia_chk_${JOB_COUNT}" \
        --output="$REPO_ROOT/logs/synthia_chk${JOB_COUNT}_%j.out" \
        --error="$REPO_ROOT/logs/synthia_chk${JOB_COUNT}_%j.err" \
        --time=04:00:00 \
        --cpus-per-task=2 \
        --mem-per-cpu=4G \
        --wrap="cd $REPO_ROOT && export PYTHONPATH=$REPO_ROOT:\${PYTHONPATH:-} && python -m planamono.gt_creation.synthia.check_depth_limit_worker $job_file $OUT_RESULT")
    echo "Check job $JOB_COUNT: $JID ($(wc -l < "$job_file" | tr -d ' ') scenes)"
    CHECK_JOB_IDS="${CHECK_JOB_IDS}:${JID}"
    JOB_COUNT=$((JOB_COUNT + 1))
done

# Remove leading colon
CHECK_JOB_IDS="${CHECK_JOB_IDS#:}"

# Submit merge + rerun job (waits for all checks)
sbatch --dependency=afterok:${CHECK_JOB_IDS} \
    --job-name="synthia_merge_rerun" \
    --output="$REPO_ROOT/logs/synthia_merge_rerun_%j.out" \
    --error="$REPO_ROOT/logs/synthia_merge_rerun_%j.err" \
    --time=00:30:00 \
    --cpus-per-task=1 \
    --mem-per-cpu=2G \
    --wrap="cd $REPO_ROOT && export PYTHONPATH=$REPO_ROOT:\${PYTHONPATH:-} && \
        cat $RESULTS_DIR/result_*.txt > $RERUN_FILE && \
        python -m planamono.gt_creation.synthia.check_missing_scenes && \
        bash $SCRIPT_DIR/synthia_rerun_parallel.sh --num_jobs $NUM_RERUN_JOBS --config $CONFIG"

echo "=============================================="
echo "Submitted $JOB_COUNT check jobs."
echo "Merge + rerun will launch after all checks complete."
echo "Monitor with: squeue -u \$USER"
echo "=============================================="

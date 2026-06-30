#!/bin/bash
# Fast scene-partitioned GT (seed 42) evaluation on ScanNet++.
#
# Splits the 42 test scenes across a SLURM array of partition jobs (each evaluates
# a few scenes and writes results_shard_<start>.csv), then runs a dependent
# aggregation job that merges the shards into the standard
# results.csv / results_per_scene.csv / results_dataset.csv.
#
# Fully self-contained: writes to its OWN output dir and does NOT touch the
# gt_seed_repro/ sweep tree or any existing script. Result is byte-identical to a
# monolithic seed-42 run (RANSAC is re-seeded per frame), but ready in << 1h.
#
# Usage:
#   bash submit_gt_seed42_fast.sh
#   FORCE=1 bash submit_gt_seed42_fast.sh   # re-run even if results already exist

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARTITION_SCRIPT="$SCRIPT_DIR/run_gt_partition_job.sh"
AGGREGATE_SCRIPT="$SCRIPT_DIR/run_gt_aggregate_job.sh"

SEED=42
TOTAL_SCENES=42        # ScanNet++ test split
NUM_PARTITIONS=15      # requested fan-out
NUM_WORKERS=8          # DataLoader workers per partition

# Separate output + log dirs (isolated from the gt_seed_repro sweep).
EVAL_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/eval/gt_seed42_fast"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_gt_fast"

EXP_VER="v6"           # gt exp_name = "gt_${EXP_VER}" (must match evaluate_all_baselines.py)

# SLURM resources (CPU-only).
PART_TIME="1:30:00"
PART_CPUS=16
PART_MEM_PER_CPU="4G"
AGG_TIME="0:20:00"
AGG_CPUS=4
AGG_MEM_PER_CPU="4G"

FORCE="${FORCE:-0}"

# ceil(TOTAL/NUM_PARTITIONS) scenes per job, then ceil(TOTAL/per_job) actual jobs
# (drops empty tail partitions: 42 scenes, 15 requested -> 3/job -> 14 jobs).
SCENES_PER_JOB=$(( (TOTAL_SCENES + NUM_PARTITIONS - 1) / NUM_PARTITIONS ))
ACTUAL_JOBS=$(( (TOTAL_SCENES + SCENES_PER_JOB - 1) / SCENES_PER_JOB ))

RESULT_CSV="${EVAL_ROOT}/gt_${EXP_VER}/results.csv"

mkdir -p "$LOG_DIR" "$EVAL_ROOT"

echo "=== Fast partitioned GT (seed $SEED) ==="
echo "Total scenes:    $TOTAL_SCENES"
echo "Partitions:      $NUM_PARTITIONS requested -> $SCENES_PER_JOB scenes/job -> $ACTUAL_JOBS array tasks"
echo "Output:          $EVAL_ROOT"
echo "Log dir:         $LOG_DIR"
echo "========================================"

if [[ -f "$RESULT_CSV" && "$FORCE" != "1" ]]; then
    echo "[SKIP] Merged results already exist ($RESULT_CSV). Set FORCE=1 to re-run."
    exit 0
fi

# Clear any stale shards so a re-run merges only this run's partitions.
if [[ "$FORCE" == "1" ]]; then
    rm -f "${EVAL_ROOT}/gt_${EXP_VER}"/results_shard_*.csv 2>/dev/null || true
fi

# --- Partition array job ---
ARRAY_JID=$(sbatch --parsable \
    --job-name="gt42fast_part" \
    --array=0-$(( ACTUAL_JOBS - 1 )) \
    --time="$PART_TIME" \
    --cpus-per-task="$PART_CPUS" \
    --mem-per-cpu="$PART_MEM_PER_CPU" \
    --output="$LOG_DIR/gt42fast_part_%A_%a.out" \
    --error="$LOG_DIR/gt42fast_part_%A_%a.err" \
    "$PARTITION_SCRIPT" "$SEED" "$EVAL_ROOT" "$SCENES_PER_JOB" "$TOTAL_SCENES" "$NUM_WORKERS")
echo "[SUBMITTED] partition array: $ARRAY_JID ($ACTUAL_JOBS tasks)"

# --- Aggregation job (waits for all partitions to succeed) ---
AGG_JID=$(sbatch --parsable \
    --job-name="gt42fast_agg" \
    --dependency=afterok:"$ARRAY_JID" \
    --time="$AGG_TIME" \
    --cpus-per-task="$AGG_CPUS" \
    --mem-per-cpu="$AGG_MEM_PER_CPU" \
    --output="$LOG_DIR/gt42fast_agg_%j.out" \
    --error="$LOG_DIR/gt42fast_agg_%j.err" \
    "$AGGREGATE_SCRIPT" "$EVAL_ROOT")
echo "[SUBMITTED] aggregate job: $AGG_JID (afterok:$ARRAY_JID)"

echo "========================================"
echo "[DONE] Merged result will appear at: $RESULT_CSV"
echo "Monitor:  squeue -u \$USER --name=gt42fast_part,gt42fast_agg"

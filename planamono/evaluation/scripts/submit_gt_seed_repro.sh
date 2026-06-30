#!/bin/bash
# Reproducibility sweep: evaluate the GT (upper-bound) method on ScanNet++ across
# multiple RANSAC seeds, each repeated several times.
#
# Submits ONE separate CPU sbatch job per (seed, repeat) so results never
# overwrite each other. Each job writes to its own --eval-root, giving:
#
#   ${OUTPUT_BASE}/gt_seed${SEED}_rep${IDX}/gt_${EXP_VER}/
#       results.csv              (per-frame: RI/VOI/SC + prec/rec@τ + bp_*)
#       results_per_scene.csv
#       results_dataset.csv      (dataset-level summary)
#       runtime_breakdown.csv
#       table_*_baselines.csv    (aggregated tables for this run)
#
# Only the 3D plane-fitting metrics (prec/rec@τ) depend on the seed; RI/VOI/SC
# and the binary-planarity metrics are deterministic, so repeats with the same
# seed should match bit-for-bit (a determinism check), while different seeds
# expose RANSAC variance.
#
# Usage:
#   bash submit_gt_seed_repro.sh                 # 5 seeds x 2 repeats = 10 jobs
#   FORCE=1 bash submit_gt_seed_repro.sh         # re-run even if a run dir already has results

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Per-job bash script (handles conda activate + python; see run_gt_eval_job.sh).
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
JOB_SCRIPT="$SCRIPT_DIR/run_gt_eval_job.sh"

# Base directory under which every (seed, repeat) run dir is created.
OUTPUT_BASE="/cluster/scratch/aoezkan/planeseg/scannetpp/eval/gt_seed_repro"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_gt_repro"

# Must match EXP_VER in evaluate_all_baselines.py (gt exp_name = "gt_${EXP_VER}").
EXP_VER="v6"

SEEDS=(0 13 42 71 109)
REPEATS=2

# SLURM resources (CPU-only: the gt method uses GT labels directly, no GPU).
SLURM_TIME="4:00:00"
SLURM_CPUS=16
SLURM_MEM_PER_CPU="4G"
NUM_WORKERS=8   # DataLoader workers for evaluate_all_baselines.py

FORCE="${FORCE:-0}"

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

echo "=== GT reproducibility sweep ==="
echo "Seeds:        ${SEEDS[*]}"
echo "Repeats:      $REPEATS"
echo "Total jobs:   $(( ${#SEEDS[@]} * REPEATS ))"
echo "Output base:  $OUTPUT_BASE"
echo "Log dir:      $LOG_DIR"
echo "================================"

SUBMITTED=0
SKIPPED=0

for SEED in "${SEEDS[@]}"; do
    for IDX in $(seq 1 "$REPEATS"); do
        RUN_NAME="gt_seed${SEED}_rep${IDX}"
        EVAL_ROOT="${OUTPUT_BASE}/${RUN_NAME}"
        RESULT_CSV="${EVAL_ROOT}/gt_${EXP_VER}/results_dataset.csv"

        # Never silently overwrite an existing completed run.
        if [[ -f "$RESULT_CSV" && "$FORCE" != "1" ]]; then
            echo "[SKIP] $RUN_NAME already has results ($RESULT_CSV). Set FORCE=1 to re-run."
            SKIPPED=$(( SKIPPED + 1 ))
            continue
        fi

        mkdir -p "$EVAL_ROOT"

        # Submit a real bash job script (NOT --wrap, which runs under /bin/sh
        # and breaks `source`/`conda activate`).
        JOB_ID=$(sbatch --parsable \
            --job-name="$RUN_NAME" \
            --time="$SLURM_TIME" \
            --cpus-per-task="$SLURM_CPUS" \
            --mem-per-cpu="$SLURM_MEM_PER_CPU" \
            --output="$LOG_DIR/${RUN_NAME}_%j.out" \
            --error="$LOG_DIR/${RUN_NAME}_%j.err" \
            "$JOB_SCRIPT" "$SEED" "$EVAL_ROOT" "$NUM_WORKERS")
        echo "[SUBMITTED] $RUN_NAME  ->  job $JOB_ID  (eval-root: $EVAL_ROOT)"
        SUBMITTED=$(( SUBMITTED + 1 ))
    done
done

echo "================================"
echo "[DONE] Submitted $SUBMITTED job(s), skipped $SKIPPED."
echo "Monitor:  squeue -u \$USER --name=gt_seed*  ||  tail -f $LOG_DIR/gt_seed*_*.out"

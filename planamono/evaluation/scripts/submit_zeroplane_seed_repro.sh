#!/bin/bash
# Reproducibility sweep: evaluate ZEROPLANE on ScanNet++ across multiple RANSAC
# seeds, each repeated several times.
#
# Mirror of submit_gt_seed_repro.sh, but for the "zeroplane" method, which reads
# pre-computed (already segmented) per-scene planes.h5 instead of GT labels.
# ZeroPlane uses label 20 for non-planar regions; evaluate_all_baselines.py
# auto-remaps 20 -> 0 via the method's nonplanar_label=20 config (nothing to do
# here, just noted for context).
#
# The eval script (evaluate_all_baselines.py) reads predictions from
#   ${H5_ROOT}/${H5_FOLDER}/<scene>/planes.h5
# H5_ROOT is hardcoded in the python source and has NO CLI override, so this
# script first ensures a symlink
#   ${H5_ROOT}/${H5_FOLDER}  ->  ${SRC_PRED_DIR}
# pointing at the saved predictions under ayavuz's scratch. The symlink setup
# is idempotent and refuses to overwrite a real directory.
#
# Submits ONE separate CPU sbatch job per (seed, repeat) so results never
# overwrite each other. Each job writes to its own --eval-root, giving:
#
#   ${OUTPUT_BASE}/zeroplane_seed${SEED}_rep${IDX}/zeroplane_${EXP_VER}/
#       results.csv              (per-frame: RI/VOI/SC + prec/rec@τ + bp_*)
#       results_per_scene.csv
#       results_dataset.csv      (dataset-level summary)
#       runtime_breakdown.csv
#       table_*_baselines.csv    (aggregated tables for this run)
#
# Only the 3D plane-fitting metrics (prec/rec@τ) depend on the seed; RI/VOI/SC
# and the binary-planarity metrics are deterministic (the segmentation is
# pre-computed), so repeats with the same seed should match bit-for-bit (a
# determinism check), while different seeds expose RANSAC variance.
#
# Usage:
#   bash submit_zeroplane_seed_repro.sh                 # 5 seeds x 2 repeats = 10 jobs
#   FORCE=1 bash submit_zeroplane_seed_repro.sh         # re-run even if a run dir already has results

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Per-job bash script (handles conda activate + python; see run_zeroplane_eval_job.sh).
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
JOB_SCRIPT="$SCRIPT_DIR/run_zeroplane_eval_job.sh"

# How the eval locates its predictions. These MUST match the corresponding
# entry in the METHODS dict of evaluate_all_baselines.py.
H5_FOLDER="zeroplane_h5"             # zeroplane -> H5_ROOT/zeroplane_h5/<scene>/planes.h5
EXP_NAME_PREFIX="zeroplane"          # zeroplane exp_name = "zeroplane_${EXP_VER}"

# Where evaluate_all_baselines.py looks for prediction H5s (hardcoded in source).
H5_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference"
# Saved ZeroPlane predictions (segmented planes.h5, label 20 = non-planar, 42/42 test scenes).
SRC_PRED_DIR="/cluster/scratch/ayavuz/scannetpp_backup2/planeseg/inference/scannetpp/zeroplane_h5"

# Base directory under which every (seed, repeat) run dir is created.
OUTPUT_BASE="/cluster/scratch/aoezkan/planeseg/scannetpp/eval/zeroplane_seed_repro"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_zeroplane_repro"

# Must match EXP_VER in evaluate_all_baselines.py (exp_name = "${EXP_NAME_PREFIX}_${EXP_VER}").
EXP_VER="v6"

SEEDS=(0 13 42 71 109)
REPEATS=2

# SLURM resources (CPU-only: predictions are pre-computed planes.h5, no GPU).
SLURM_TIME="4:00:00"
SLURM_CPUS=16
SLURM_MEM_PER_CPU="4G"
NUM_WORKERS=8   # DataLoader workers for evaluate_all_baselines.py

FORCE="${FORCE:-0}"

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

# ---------------------------------------------------------------------------
# Ensure H5_ROOT/<H5_FOLDER> points at the saved predictions (idempotent).
# ---------------------------------------------------------------------------
ensure_pred_symlink() {
    local link="$1" target="$2"
    if [[ ! -d "$target" ]]; then
        echo "[ERROR] Source predictions not found: $target" >&2
        exit 1
    fi
    if [[ -L "$link" ]]; then
        if [[ "$(readlink -f "$link")" == "$(readlink -f "$target")" ]]; then
            echo "[OK]   symlink already correct: $link -> $target"
            return
        fi
        echo "[ERROR] $link is a symlink to '$(readlink -f "$link")', not '$target'." >&2
        echo "        Remove it manually before re-running." >&2
        exit 1
    fi
    if [[ -e "$link" ]]; then
        echo "[ERROR] $link exists and is NOT a symlink (real dir/file). Refusing to overwrite." >&2
        exit 1
    fi
    mkdir -p "$(dirname "$link")"
    ln -s "$target" "$link"
    echo "[LINK] $link -> $target"
}

PRED_LINK="${H5_ROOT}/${H5_FOLDER}"
ensure_pred_symlink "$PRED_LINK" "$SRC_PRED_DIR"

echo "=== ZEROPLANE reproducibility sweep ==="
echo "Method:       zeroplane"
echo "Predictions:  $PRED_LINK (-> $SRC_PRED_DIR)"
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
        RUN_NAME="zeroplane_seed${SEED}_rep${IDX}"
        EVAL_ROOT="${OUTPUT_BASE}/${RUN_NAME}"
        RESULT_CSV="${EVAL_ROOT}/${EXP_NAME_PREFIX}_${EXP_VER}/results_dataset.csv"

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
echo "Monitor:  squeue -u \$USER --name=zeroplane_seed*  ||  tail -f $LOG_DIR/zeroplane_seed*_*.out"

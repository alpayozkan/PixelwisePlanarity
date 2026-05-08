#!/bin/bash
# Submit parallel SLURM shards of evaluate_gt_moge_zeroplane_v1.py.
#
# Layout per (method, dataset):
#   - N_scannetpp_shards CPU jobs for ScanNet++ (default 20)
#   - N_other_shards CPU jobs for nyuv2/sevenscenes (default 1)
# Plus one --aggregate_only job per method that depends on its shards.
#
# CPU-only: the v1 pipeline keeps `compute_vectorized_planar_segments_v5_relative`
# on `device="cpu"`, all other compute is numpy/sklearn. No GPU is needed.
#
# Usage:
#   bash submit_eval_gt_moge_zp_v1.sh --exp gt_moge_zp_v1 \
#       [--methods "gt moge zeroplane"] \
#       [--datasets "scannetpp nyuv2 sevenscenes"] \
#       [--num-shards-scannetpp 20]   # optional, default 20
#       [--num-shards-other 1]         # optional, default 1 — n=1 is enough
#                                       #   for nyuv2 / sevenscenes
#       [--scenes N] [--frames N]      # optional caps (ALL methods/datasets)
#       [--moge-root PATH] [--zp-root PATH] [--eval-root PATH] \
#       [--time 4:00:00] [--cpus 8] [--mem-per-cpu 8G] \
#       [--dry-run]
#
# Examples:
#   # Full sweep, all 3 methods × all 3 datasets
#   bash submit_eval_gt_moge_zp_v1.sh --exp gt_moge_zp_v1
#
#   # Just MoGe on ScanNet++ (20 shards + aggregate)
#   bash submit_eval_gt_moge_zp_v1.sh --exp moge_only --methods moge --datasets scannetpp
#
#   # Just GT on NYU-v2 / 7-Scenes (1 shard each + aggregate)
#   bash submit_eval_gt_moge_zp_v1.sh --exp gt_only --methods gt --datasets "nyuv2 sevenscenes"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY="$SCRIPT_DIR/evaluate_gt_moge_zeroplane_v1.py"
LOGS_ROOT="/cluster/scratch/aoezkan/planeseg/logs/eval_gt_moge_zp_v1"

# IMPORTANT: conda env is `planeseg`, not `planamono` (legacy scripts get this wrong).
ENV_ACTIVATE="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planeseg"

# ---------- defaults ----------
EXP=""
METHODS="gt moge zeroplane"
DATASETS="scannetpp nyuv2 sevenscenes"
NUM_SHARDS_SCANNETPP=20
NUM_SHARDS_OTHER=1
MOGE_ROOT="/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1"
ZP_ROOT="/cluster/scratch/aoezkan/planeseg/inference/zeroplane_default_dust3r_released_h5"
EVAL_ROOT="/cluster/scratch/aoezkan/planeseg/eval"
TIME="4:00:00"
AGG_TIME="00:30:00"
CPUS=8
MEM_PER_CPU="8G"
DRY_RUN=0
SCENES=""
FRAMES=""

# ---------- arg parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)                   EXP="$2"; shift 2 ;;
        --methods)               METHODS="$2"; shift 2 ;;
        --datasets)              DATASETS="$2"; shift 2 ;;
        --num-shards-scannetpp)  NUM_SHARDS_SCANNETPP="$2"; shift 2 ;;
        --num-shards-other)      NUM_SHARDS_OTHER="$2"; shift 2 ;;
        --moge-root)             MOGE_ROOT="$2"; shift 2 ;;
        --zp-root)               ZP_ROOT="$2"; shift 2 ;;
        --eval-root)             EVAL_ROOT="$2"; shift 2 ;;
        --time)                  TIME="$2"; shift 2 ;;
        --agg-time)              AGG_TIME="$2"; shift 2 ;;
        --cpus)                  CPUS="$2"; shift 2 ;;
        --mem-per-cpu)           MEM_PER_CPU="$2"; shift 2 ;;
        --scenes)                SCENES="$2"; shift 2 ;;
        --frames)                FRAMES="$2"; shift 2 ;;
        --dry-run)               DRY_RUN=1; shift ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$EXP" ]]; then
    echo "ERROR: --exp is required" >&2
    exit 2
fi

mkdir -p "$LOGS_ROOT/$EXP"

submit() {
    # submit JOBNAME LOGPREFIX TIMELIMIT DEPENDENCY CMD
    local jobname="$1" logprefix="$2" timelimit="$3" dependency="$4" cmd="$5"
    local dep_arg=""
    if [[ -n "$dependency" ]]; then
        dep_arg="--dependency=afterok:${dependency}"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        # Print sbatch invocation to STDERR so it's visible but not captured
        # by `JOB_ID=$(submit ...)` — caller only sees the synthetic job id.
        {
            echo "[dry-run] sbatch --job-name=$jobname --time=$timelimit ${dep_arg}"
            echo "          --cpus-per-task=$CPUS --mem-per-cpu=$MEM_PER_CPU"
            echo "          --output=${logprefix}_%j.out --error=${logprefix}_%j.err"
            echo "          --wrap=\"bash -c '$cmd'\""
        } >&2
        echo "DRY${RANDOM}"
        return
    fi

    # IMPORTANT: SLURM's --wrap runs via /bin/sh (dash on Ubuntu), which lacks
    # the `source` builtin. Wrap in `bash -c '...'` to get bash semantics.
    sbatch --parsable \
        --job-name="$jobname" \
        --time="$timelimit" \
        --cpus-per-task="$CPUS" \
        --mem-per-cpu="$MEM_PER_CPU" \
        --output="${logprefix}_%j.out" \
        --error="${logprefix}_%j.err" \
        ${dep_arg} \
        --wrap="bash -c '$cmd'"
}

echo "================================================================"
echo " Submitting evaluate_gt_moge_zeroplane_v1.py shards"
echo "================================================================"
echo "  exp:                    $EXP"
echo "  methods:                $METHODS"
echo "  datasets:               $DATASETS"
echo "  num_shards_scannetpp:   $NUM_SHARDS_SCANNETPP"
echo "  num_shards_other:       $NUM_SHARDS_OTHER"
echo "  moge root:              $MOGE_ROOT"
echo "  zeroplane root:         $ZP_ROOT"
echo "  eval root:              $EVAL_ROOT"
echo "  time / agg-time:        $TIME / $AGG_TIME"
echo "  cpus / mem-per-cpu:     $CPUS / $MEM_PER_CPU"
echo "  scenes / frames cap:    ${SCENES:-(all)} / ${FRAMES:-(all)}"
echo "  dry-run:                $DRY_RUN"
echo "  logs:                   $LOGS_ROOT/$EXP/"
echo "================================================================"

COMMON_ARGS="--exp $EXP \
    --moge_signals_root $MOGE_ROOT \
    --zeroplane_h5_root $ZP_ROOT \
    --eval_root $EVAL_ROOT \
    --n_jobs $CPUS \
    --num_workers 4"
if [[ -n "$SCENES" ]]; then COMMON_ARGS="$COMMON_ARGS --scenes $SCENES"; fi
if [[ -n "$FRAMES" ]]; then COMMON_ARGS="$COMMON_ARGS --frames $FRAMES"; fi

# One aggregate job per method (depends on ALL of that method's shards across
# all datasets, so the per-(method, dataset) aggregates land before the
# top-level summary.csv is written).
for METHOD in $METHODS; do
    METHOD_DEPS=()

    for DATASET in $DATASETS; do
        if [[ "$DATASET" == "scannetpp" ]]; then
            N_SHARDS="$NUM_SHARDS_SCANNETPP"
        else
            N_SHARDS="$NUM_SHARDS_OTHER"
        fi

        echo ""
        echo "[plan] $METHOD / $DATASET → $N_SHARDS shard(s)"

        for ((SHARD=0; SHARD<N_SHARDS; SHARD++)); do
            SHARD_PADDED=$(printf "%03d" "$SHARD")
            JOB_NAME="${EXP}_${METHOD}_${DATASET}_s${SHARD_PADDED}"
            LOG_PREFIX="$LOGS_ROOT/$EXP/${METHOD}_${DATASET}_s${SHARD_PADDED}"

            CMD="${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python ${EVAL_PY} \
                ${COMMON_ARGS} \
                --methods ${METHOD} \
                --datasets ${DATASET} \
                --shard_id ${SHARD} \
                --num_shards ${N_SHARDS}"

            JOB_ID=$(submit "$JOB_NAME" "$LOG_PREFIX" "$TIME" "" "$CMD")
            METHOD_DEPS+=("$JOB_ID")
            echo "  shard $SHARD → job $JOB_ID"
        done
    done

    # Aggregate job for this method, dependent on all of its shards.
    AGG_JOB_NAME="${EXP}_${METHOD}_aggregate"
    AGG_LOG_PREFIX="$LOGS_ROOT/$EXP/${METHOD}_aggregate"
    DEP_STR=$(IFS=:; echo "${METHOD_DEPS[*]}")
    AGG_CMD="${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python ${EVAL_PY} \
        ${COMMON_ARGS} \
        --methods ${METHOD} \
        --datasets ${DATASETS} \
        --aggregate_only"

    AGG_JOB_ID=$(submit "$AGG_JOB_NAME" "$AGG_LOG_PREFIX" "$AGG_TIME" "$DEP_STR" "$AGG_CMD")
    echo "  aggregate → job $AGG_JOB_ID  (afterok:$DEP_STR)"
done

echo ""
echo "Done. Monitor with:"
echo "  squeue -u \$USER -n \"${EXP}_*\" --format='%.18i %.30j %.8T %.10M %.6D %R'"
echo "Logs:  $LOGS_ROOT/$EXP/"

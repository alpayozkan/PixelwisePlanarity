#!/bin/bash
# Submit evaluate_gt_moge_zeroplane_benchmark.py jobs across (method, dataset)
# cells with per-dataset scene partitioning, then chain a single aggregator job
# behind all of them.
#
# Mirrors submit_eval_gt_moge_zp.sh — same flags, same partitioning logic,
# same env-var override surface — but targets the BENCHMARK pipeline
# (ZeroPlane + PlaneRCNN + PlaneRecTR metrics, see docs/metrics_benchmark.md).
#
# Resource model: ALL jobs are CPU-only. The only torch op (MoGe segmentation
# via compute_vectorized_planar_segments_v5_relative) runs on torch.cpu and is
# small (Sobel + 5×5 unfold). Everything else is numpy/scipy. No GPU needed.
#
# Defaults:
#   EXP                 = "gt_moge_zp_benchmark"
#   METHODS             = "gt moge zeroplane"
#   DATASETS            = "scannetpp nyuv2 sevenscenes"
#   PARTS_SCANNETPP     = 20    (42 scenes → 2-3 per part, round-robin)
#   PARTS_NYUV2         = 1     (single virtual "nyuv2" scene)
#   PARTS_SEVENSCENES   = 1     (7 small scenes)
#   MAX_SCENES          = unset (= all scenes)
#   MAX_FRAMES          = unset (= all frames per scene)
#
# Path defaults match the python script defaults.
#
# Usage (CLI flags or env vars; flags win):
#   bash submit_eval_gt_moge_zp_benchmark.sh
#   bash submit_eval_gt_moge_zp_benchmark.sh --methods moge --scenes 5 --frames 10 --n-jobs 4
#   bash submit_eval_gt_moge_zp_benchmark.sh --exp my_exp --datasets scannetpp
#   METHODS=moge MAX_SCENES=5 MAX_FRAMES=10 bash submit_eval_gt_moge_zp_benchmark.sh
#
# CLI flags:
#   --exp NAME              Experiment name (default: gt_moge_zp_benchmark).
#   --methods "M1 M2 ..."   Subset of {gt,moge,zeroplane} (default: all 3).
#   --datasets "D1 D2 ..."  Subset of {scannetpp,nyuv2,sevenscenes} (default: all 3).
#   --scenes N              Limit to first N scenes per dataset (default: all).
#   --frames N              Limit to N evenly-spaced frames per scene (default: all).
#   --n-jobs N              Number of parts for ScanNet++ (default: 20).
#                           NYU-v2 / 7-Scenes are always 1 part — they're small
#                           enough that splitting them is wasteful.
#   --eval-root PATH        Output root (default: /cluster/scratch/.../eval).
#
# Layout produced:
#   <EVAL_ROOT>/<EXP>/<method>/<dataset>/<scene>/{results,summary}.csv     (workers)
#   <EVAL_ROOT>/<EXP>/<method>/<dataset>/aggregate_*.csv                   (aggregator)
#   <EVAL_ROOT>/<EXP>/summary.csv                                          (aggregator)

set -euo pipefail

# ── Defaults (env vars first; CLI flags override below) ─────────────────────
EXP="${EXP:-gt_moge_zp_benchmark}"
METHODS="${METHODS:-gt moge zeroplane}"
DATASETS="${DATASETS:-scannetpp nyuv2 sevenscenes}"

PARTS_SCANNETPP="${PARTS_SCANNETPP:-20}"
PARTS_NYUV2="${PARTS_NYUV2:-1}"
PARTS_SEVENSCENES="${PARTS_SEVENSCENES:-1}"

MAX_SCENES="${MAX_SCENES:-}"   # empty = all
MAX_FRAMES="${MAX_FRAMES:-}"   # empty = all

EVAL_ROOT="${EVAL_ROOT:-/cluster/scratch/aoezkan/planeseg/eval}"
MOGE_SIGNALS_ROOT="${MOGE_SIGNALS_ROOT:-/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1}"
ZEROPLANE_H5_ROOT="${ZEROPLANE_H5_ROOT:-/cluster/scratch/aoezkan/planeseg/inference/zeroplane_default_dust3r_released_h5}"

SCANNETPP_SPLIT="${SCANNETPP_SPLIT:-/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/splits/scannetpp/test.txt}"

PART_TIME="${PART_TIME:-4:00:00}"
PART_CPUS="${PART_CPUS:-8}"
PART_MEM="${PART_MEM:-4G}"
AGG_TIME="${AGG_TIME:-0:30:00}"

# ── Parse CLI flags (override env defaults) ─────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)               EXP="$2"; shift 2 ;;
        --methods)           METHODS="$2"; shift 2 ;;
        --datasets)          DATASETS="$2"; shift 2 ;;
        --scenes)            MAX_SCENES="$2"; shift 2 ;;
        --frames)            MAX_FRAMES="$2"; shift 2 ;;
        --n-jobs|--parts)    PARTS_SCANNETPP="$2"; shift 2 ;;
        --eval-root|--eval_root) EVAL_ROOT="$2"; shift 2 ;;
        --moge-root)         MOGE_SIGNALS_ROOT="$2"; shift 2 ;;
        --zp-root)           ZEROPLANE_H5_ROOT="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail/p' "$0" | head -n -2 | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown arg: $1 (try --help)" >&2; exit 1 ;;
    esac
done

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
PY_SCRIPT="$PROJECT_ROOT/evaluation/quantitative/evaluate_gt_moge_zeroplane_benchmark.py"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_gt_moge_zp_benchmark/$EXP"
mkdir -p "$LOG_DIR"

if [ ! -f "$SCANNETPP_SPLIT" ]; then
    echo "[ERROR] SCANNETPP_SPLIT not found: $SCANNETPP_SPLIT" >&2
    exit 1
fi

# ── Per-dataset scene lists ─────────────────────────────────────────────────
declare -A DATASET_SCENES
DATASET_SCENES[scannetpp]="$(grep -v '^[[:space:]]*$' "$SCANNETPP_SPLIT" | tr '\n' ' ')"
DATASET_SCENES[nyuv2]="nyuv2"
DATASET_SCENES[sevenscenes]="chess fire heads office pumpkin redkitchen stairs"

declare -A DATASET_PARTS
DATASET_PARTS[scannetpp]="$PARTS_SCANNETPP"
DATASET_PARTS[nyuv2]="$PARTS_NYUV2"
DATASET_PARTS[sevenscenes]="$PARTS_SEVENSCENES"

# ── Build python pass-through args for limits ───────────────────────────────
# NOTE: Don't pass --max_scenes to workers — the bash already caps the SCENES
# array at MAX_SCENES and passes the resulting IDs via --scene_ids, which IS
# the authoritative cap. Passing --max_scenes too would race: the python
# dataset constructor truncates to its first N alphabetic scenes, which can
# differ from the bash-picked first N test.txt scenes, causing --scene_ids to
# filter out everything → 0 scenes processed.
LIMIT_ARGS=""
if [ -n "$MAX_FRAMES" ]; then LIMIT_ARGS+=" --max_frames_per_scene ${MAX_FRAMES}"; fi

echo "================================================================"
echo " evaluate_gt_moge_zeroplane_benchmark — multi-(method, dataset) submission"
echo "================================================================"
echo " EXP:               $EXP"
echo " METHODS:           $METHODS"
echo " DATASETS:          $DATASETS"
echo " parts/dataset:     scannetpp=$PARTS_SCANNETPP  nyuv2=$PARTS_NYUV2  sevenscenes=$PARTS_SEVENSCENES"
echo " --scenes (max):    ${MAX_SCENES:-all}"
echo " --frames (per):    ${MAX_FRAMES:-all}"
echo " moge root:         $MOGE_SIGNALS_ROOT"
echo " zeroplane root:    $ZEROPLANE_H5_ROOT"
echo " eval root:         $EVAL_ROOT"
echo " resources/part:    ${PART_TIME}, ${PART_CPUS} cpu, ${PART_MEM}/cpu (CPU-only)"
echo " log dir:           $LOG_DIR"
echo "================================================================"

# ── Submit per-(method, dataset, part) worker jobs ──────────────────────────

JOB_IDS=()

for METHOD in $METHODS; do
    for DS in $DATASETS; do
        SCENES_RAW="${DATASET_SCENES[$DS]:-}"
        N_PARTS_DS="${DATASET_PARTS[$DS]:-1}"
        if [ -z "$SCENES_RAW" ]; then
            echo "  [skip] unknown dataset: $DS"
            continue
        fi
        IFS=' ' read -ra SCENES <<< "$SCENES_RAW"

        # Apply --scenes cap before partitioning (so N parts cover the capped set,
        # not silently empty).
        if [ -n "$MAX_SCENES" ] && [ "$MAX_SCENES" -lt "${#SCENES[@]}" ]; then
            SCENES=("${SCENES[@]:0:$MAX_SCENES}")
        fi
        N_SCENES=${#SCENES[@]}

        # Don't create more parts than scenes.
        if [ "$N_PARTS_DS" -gt "$N_SCENES" ]; then
            N_PARTS_DS="$N_SCENES"
        fi
        if [ "$N_SCENES" -eq 0 ]; then
            echo "  [skip] $METHOD/$DS: no scenes"
            continue
        fi

        # Round-robin scene → part assignment.
        declare -A PART_SCENES
        for i in "${!SCENES[@]}"; do
            p=$((i % N_PARTS_DS))
            PART_SCENES[$p]+="${SCENES[i]},"
        done

        echo ""
        echo "==== $METHOD / $DS  ($N_SCENES scenes, $N_PARTS_DS parts) ===="

        for PART in $(seq 0 $((N_PARTS_DS - 1))); do
            SCENE_CSV="${PART_SCENES[$PART]:-}"
            if [ -z "$SCENE_CSV" ]; then continue; fi
            SCENE_CSV="${SCENE_CSV%,}"   # strip trailing comma
            NUM_S=$(echo "$SCENE_CSV" | tr ',' '\n' | grep -cv '^[[:space:]]*$')

            JOB_ID=$(/cluster/apps/slurm/bin/sbatch --parsable \
                --time="$PART_TIME" \
                --cpus-per-task="$PART_CPUS" \
                --mem-per-cpu="$PART_MEM" \
                --job-name="evb_${METHOD}_${DS}_p${PART}" \
                --output="$LOG_DIR/${METHOD}_${DS}_p${PART}_of${N_PARTS_DS}_%j.out" \
                --error="$LOG_DIR/${METHOD}_${DS}_p${PART}_of${N_PARTS_DS}_%j.err" \
                <<EOF
#!/bin/bash
set -uo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg

echo "==== $METHOD / $DS — part $PART/$N_PARTS_DS ($NUM_S scenes) ===="
echo "scenes: $SCENE_CSV"
echo ""

set -x
python ${PY_SCRIPT} --exp ${EXP} --methods ${METHOD} --datasets ${DS} --moge_signals_root ${MOGE_SIGNALS_ROOT} --zeroplane_h5_root ${ZEROPLANE_H5_ROOT} --eval_root ${EVAL_ROOT} --scene_ids ${SCENE_CSV} --skip_dataset_aggregates --n_jobs ${PART_CPUS} ${LIMIT_ARGS}
set +x
EOF
)
            JOB_IDS+=("$JOB_ID")
            echo "  $METHOD/$DS part $PART/$N_PARTS_DS ($NUM_S scenes) → job $JOB_ID"
        done
        unset PART_SCENES
    done
done

if [ "${#JOB_IDS[@]}" -eq 0 ]; then
    echo ""
    echo "[ERROR] no worker jobs were submitted (empty METHODS or DATASETS?)" >&2
    exit 1
fi

# ── Final aggregator: depends on every worker (afterany — runs even if some fail) ──
DEP=$(IFS=:; echo "${JOB_IDS[*]}")
AGG_JOB=$(/cluster/apps/slurm/bin/sbatch --parsable \
    --time="$AGG_TIME" \
    --cpus-per-task=2 \
    --mem-per-cpu=4G \
    --dependency="afterany:${DEP}" \
    --job-name="evb_${EXP}_agg" \
    --output="$LOG_DIR/aggregate_%j.out" \
    --error="$LOG_DIR/aggregate_%j.err" \
    <<EOF
#!/bin/bash
set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg

set -x
python ${PY_SCRIPT} --exp ${EXP} --methods ${METHODS} --datasets ${DATASETS} --eval_root ${EVAL_ROOT} --aggregate_only
set +x
EOF
)

echo ""
echo "================================================================"
echo "Submitted ${#JOB_IDS[@]} worker jobs + 1 aggregator (job ${AGG_JOB})"
echo "Aggregator depends on (afterany): ${JOB_IDS[*]}"
echo ""
echo "Logs:    $LOG_DIR"
echo "Output:  $EVAL_ROOT/$EXP/<method>/<dataset>/<scene>/{results,summary}.csv"
echo "         $EVAL_ROOT/$EXP/<method>/<dataset>/aggregate_*.csv"
echo "         $EVAL_ROOT/$EXP/summary.csv"
echo ""
echo "Cancel:  scancel ${JOB_IDS[*]} ${AGG_JOB}"
echo "================================================================"

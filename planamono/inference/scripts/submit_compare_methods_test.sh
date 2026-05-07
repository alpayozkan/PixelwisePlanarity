#!/bin/bash
# Submit compare_plane_param_methods.py over the ScanNet++ test split (42 scenes),
# split round-robin across NUM_PARTS SLURM jobs, then chain a single aggregator
# job behind all of them.
#
# Defaults:
#   NUM_PARTS  = 20
#   METHODS    = "least_squares svd ransac"
#   SPLIT_FILE = planamono/splits/scannetpp/test.txt
#   SIGNALS    = /cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1
#   OUTPUT     = /cluster/scratch/aoezkan/planeseg/inference/compare_methods_test_ls_svd_ransac
#
# Each part runs its assigned scenes serially, all frames per scene (no --frames cap).
# The aggregator job depends on all parts (afterany — runs even if some parts fail)
# and writes:
#     aggregate_results.csv   concat of every per-scene results.csv
#     aggregate_summary.csv   per-method mean/std across all (scene, frame) rows
#                             + n_scenes / n_frames / n_rows totals
#
# Usage:
#   bash submit_compare_methods_test.sh
#   NUM_PARTS=10 OUTPUT_ROOT=/path bash submit_compare_methods_test.sh
#   METHODS="svd ransac" bash submit_compare_methods_test.sh   # custom subset

set -euo pipefail

# ── Configuration (override via environment variables) ──
NUM_PARTS="${NUM_PARTS:-20}"
METHODS="${METHODS:-least_squares svd ransac}"
SPLIT_FILE="${SPLIT_FILE:-/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/splits/scannetpp/test.txt}"
SIGNALS_ROOT="${SIGNALS_ROOT:-/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/scratch/aoezkan/planeseg/inference/compare_methods_test_ls_svd_ransac}"
PART_TIME="${PART_TIME:-12:00:00}"
AGG_TIME="${AGG_TIME:-0:30:00}"

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/compare_methods"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

# ── Read scene list, filter to scenes that have a signals dump ──
mapfile -t ALL_SCENES < <(grep -v '^[[:space:]]*$' "$SPLIT_FILE")
N_TOTAL=${#ALL_SCENES[@]}

SCENES=()
MISSING=()
for s in "${ALL_SCENES[@]}"; do
    if [ -f "$SIGNALS_ROOT/$s/moge_signals.h5" ]; then
        SCENES+=("$s")
    else
        MISSING+=("$s")
    fi
done
N_SCENES=${#SCENES[@]}

echo "================================================================"
echo " compare_plane_param_methods — parallel SLURM submission"
echo "================================================================"
echo " split file:    $SPLIT_FILE  ($N_TOTAL listed)"
echo " scenes used:   $N_SCENES (with moge_signals.h5)"
echo " missing:       ${#MISSING[@]} (no signals dump)"
echo " parts:         $NUM_PARTS"
echo " methods:       $METHODS"
echo " signals_root:  $SIGNALS_ROOT"
echo " output_root:   $OUTPUT_ROOT"
echo " log dir:       $LOG_DIR"
echo "================================================================"

if [ "$N_SCENES" -eq 0 ]; then
    echo "[ERROR] no scenes with moge_signals.h5 found under $SIGNALS_ROOT" >&2
    exit 1
fi

if [ "${#MISSING[@]}" -gt 0 ]; then
    echo " [skipped] missing scenes: ${MISSING[*]}"
fi

# ── Round-robin scene → part assignment (balances 42 across 20 well: 2x3 + 18x2) ──
declare -A PART_SCENES
for i in "${!SCENES[@]}"; do
    p=$((i % NUM_PARTS))
    PART_SCENES[$p]+="${SCENES[i]} "
done

JOB_IDS=()
for PART in $(seq 0 $((NUM_PARTS - 1))); do
    SCENES_FOR_PART="${PART_SCENES[$PART]:-}"
    if [ -z "$SCENES_FOR_PART" ]; then
        echo "  Part $PART/$NUM_PARTS: no scenes (skipped)"
        continue
    fi
    NUM_S=$(echo "$SCENES_FOR_PART" | wc -w)

    JOB_ID=$(sbatch --parsable \
        --time="$PART_TIME" \
        --cpus-per-task=8 \
        --mem-per-cpu=8G \
        --job-name="cmp_p${PART}" \
        --output="${LOG_DIR}/compare_part${PART}_of${NUM_PARTS}_%j.out" \
        --error="${LOG_DIR}/compare_part${PART}_of${NUM_PARTS}_%j.err" \
        <<EOF
#!/bin/bash
set -uo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg

echo "==== Part ${PART}/${NUM_PARTS} — ${NUM_S} scenes ===="
echo "Scenes: ${SCENES_FOR_PART}"
echo ""

FAIL=0
for SCENE in ${SCENES_FOR_PART}; do
    echo ""
    echo "==> Part ${PART} — scene \$SCENE"
    python ${PROJECT_ROOT}/inference/planarity/compare_plane_param_methods.py \\
        --scene_id \$SCENE \\
        --signals_root ${SIGNALS_ROOT} \\
        --output_root ${OUTPUT_ROOT} \\
        --methods ${METHODS} \\
        --device cpu \\
        || { echo "[FAIL] scene \$SCENE"; FAIL=\$((FAIL + 1)); }
done

echo ""
echo "==== Part ${PART} done — \${FAIL} scene failures ===="
EOF
)
    JOB_IDS+=("$JOB_ID")
    echo "  Part $PART/$NUM_PARTS  ${NUM_S} scenes  → job $JOB_ID"
done

# ── Aggregator: depends on all parts, runs whether they succeed or fail ──
DEP=$(IFS=:; echo "${JOB_IDS[*]}")
AGG_ID=$(sbatch --parsable \
    --time="$AGG_TIME" \
    --cpus-per-task=8 \
    --mem-per-cpu=8G \
    --dependency="afterany:${DEP}" \
    --job-name="cmp_agg" \
    --output="${LOG_DIR}/compare_aggregate_%j.out" \
    --error="${LOG_DIR}/compare_aggregate_%j.err" \
    <<EOF
#!/bin/bash
set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg

python ${PROJECT_ROOT}/inference/planarity/aggregate_compare_methods.py \\
    --output_root ${OUTPUT_ROOT} \\
    --methods ${METHODS} \\
    --scenes ${SPLIT_FILE}
EOF
)

echo ""
echo "================================================================"
echo "Submitted ${#JOB_IDS[@]} part jobs + 1 aggregator (job ${AGG_ID})"
echo "Aggregator depends on (afterany): ${JOB_IDS[*]}"
echo "Logs:    $LOG_DIR/compare_part*_${NUM_PARTS}_*.out"
echo "         $LOG_DIR/compare_aggregate_${AGG_ID}.out"
echo "Output:  $OUTPUT_ROOT"
echo "         └── <scene_id>/results.csv  + summary.csv  (per-scene)"
echo "         └── aggregate_results.csv   (concat of all)"
echo "         └── aggregate_summary.csv   (per-method mean/std + totals)"
echo ""
echo "Cancel all: scancel ${JOB_IDS[*]} ${AGG_ID}"
echo "================================================================"

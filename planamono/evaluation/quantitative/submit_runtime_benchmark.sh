#!/bin/bash
# Submit runtime_benchmark_region_growing.py jobs across N shards on
# scannetpp scenes from MoGe signals, then chain a single aggregator
# behind all of them.
#
# Mirrors submit_eval_gt_moge_zp_benchmark.sh structurally — same shard /
# aggregator pattern — but each worker is GPU-enabled (region growing is
# torch.cuda inside compute_vectorized_planar_segments_v5_relative).
#
# Defaults:
#   EXP                 = "runtime_v5rel"
#   PARTS               = 8     (42 scenes → 5-6 per shard)
#   N_WARMUP            = 3
#   N_REPEAT            = 10
#   MAX_FRAMES          = unset (= all frames per scene)
#
# Usage:
#   bash submit_runtime_benchmark.sh
#   bash submit_runtime_benchmark.sh --exp my_run --parts 4 --frames 50
#   bash submit_runtime_benchmark.sh --device cpu          # CPU-only mode
#
# Output:
#   /cluster/scratch/aoezkan/planeseg/eval/moge_runtime_benchmark/
#       <EXP>/scannetpp/<scene>/results.csv             (per worker)
#       <EXP>/scannetpp/aggregate_results.csv           (aggregator)
#       <EXP>/scannetpp/aggregate_per_resolution.csv    (aggregator)

set -euo pipefail

EXP="${EXP:-runtime_v5rel}"
PARTS="${PARTS:-8}"
N_WARMUP="${N_WARMUP:-3}"
N_REPEAT="${N_REPEAT:-10}"
MAX_FRAMES="${MAX_FRAMES:-}"
DEVICE="${DEVICE:-cuda}"

EVAL_ROOT="${EVAL_ROOT:-/cluster/scratch/aoezkan/planeseg/eval/moge_runtime_benchmark}"
MOGE_SIGNALS_ROOT="${MOGE_SIGNALS_ROOT:-/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1}"
SCANNETPP_SPLIT="${SCANNETPP_SPLIT:-/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/splits/scannetpp/test.txt}"

PART_TIME="${PART_TIME:-2:00:00}"
PART_CPUS="${PART_CPUS:-4}"
PART_MEM="${PART_MEM:-8G}"
PART_GPUS="${PART_GPUS:-1}"
AGG_TIME="${AGG_TIME:-0:15:00}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)               EXP="$2"; shift 2 ;;
        --parts|--n-jobs)    PARTS="$2"; shift 2 ;;
        --frames)            MAX_FRAMES="$2"; shift 2 ;;
        --warmup)            N_WARMUP="$2"; shift 2 ;;
        --repeat)            N_REPEAT="$2"; shift 2 ;;
        --device)            DEVICE="$2"; shift 2 ;;
        --eval-root|--eval_root) EVAL_ROOT="$2"; shift 2 ;;
        --moge-root)         MOGE_SIGNALS_ROOT="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail/p' "$0" | head -n -2 | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown arg: $1 (try --help)" >&2; exit 1 ;;
    esac
done

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
PY_SCRIPT="$PROJECT_ROOT/evaluation/quantitative/runtime_benchmark_region_growing.py"

LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/runtime_benchmark/$EXP"
mkdir -p "$LOG_DIR"

if [ ! -f "$SCANNETPP_SPLIT" ]; then
    echo "[ERROR] SCANNETPP_SPLIT not found: $SCANNETPP_SPLIT" >&2
    exit 1
fi

# Build scene list (round-robin shard assignment).
SCENES_RAW="$(grep -v '^[[:space:]]*$' "$SCANNETPP_SPLIT" | tr '\n' ' ')"
IFS=' ' read -ra SCENES <<< "$SCENES_RAW"
N_SCENES=${#SCENES[@]}

if [ "$PARTS" -gt "$N_SCENES" ]; then
    PARTS="$N_SCENES"
fi

declare -A PART_SCENES
for i in "${!SCENES[@]}"; do
    p=$((i % PARTS))
    PART_SCENES[$p]+="${SCENES[i]},"
done

LIMIT_ARGS=""
if [ -n "$MAX_FRAMES" ]; then LIMIT_ARGS+=" --max_frames_per_scene ${MAX_FRAMES}"; fi

GPU_FLAG=""
if [ "$DEVICE" = "cuda" ]; then
    GPU_FLAG="--gpus=${PART_GPUS}"
fi

echo "================================================================"
echo " runtime_benchmark_region_growing — multi-shard submission"
echo "================================================================"
echo " EXP:               $EXP"
echo " PARTS:             $PARTS  (over $N_SCENES scannetpp scenes)"
echo " device:            $DEVICE"
echo " warmup/repeat:     $N_WARMUP / $N_REPEAT"
echo " --frames (per):    ${MAX_FRAMES:-all}"
echo " moge root:         $MOGE_SIGNALS_ROOT"
echo " eval root:         $EVAL_ROOT"
echo " resources/part:    ${PART_TIME}, ${PART_CPUS} cpu, ${PART_MEM}/cpu, gpu=${PART_GPUS} (if cuda)"
echo " log dir:           $LOG_DIR"
echo "================================================================"

JOB_IDS=()

for PART in $(seq 0 $((PARTS - 1))); do
    SCENE_CSV="${PART_SCENES[$PART]:-}"
    if [ -z "$SCENE_CSV" ]; then continue; fi
    SCENE_CSV="${SCENE_CSV%,}"
    NUM_S=$(echo "$SCENE_CSV" | tr ',' '\n' | grep -cv '^[[:space:]]*$')

    JOB_ID=$(/cluster/apps/slurm/bin/sbatch --parsable \
        --time="$PART_TIME" \
        --cpus-per-task="$PART_CPUS" \
        --mem-per-cpu="$PART_MEM" \
        ${GPU_FLAG} \
        --job-name="rtb_p${PART}" \
        --output="$LOG_DIR/p${PART}_of${PARTS}_%j.out" \
        --error="$LOG_DIR/p${PART}_of${PARTS}_%j.err" \
        <<EOF
#!/bin/bash
set -uo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg

echo "==== runtime benchmark — part $PART/$PARTS ($NUM_S scenes) ===="
echo "scenes: $SCENE_CSV"
echo ""

set -x
python ${PY_SCRIPT} --exp ${EXP} --moge_signals_root ${MOGE_SIGNALS_ROOT} --eval_root ${EVAL_ROOT} --device ${DEVICE} --n_warmup ${N_WARMUP} --n_repeat ${N_REPEAT} --scene_ids ${SCENE_CSV} --skip_dataset_aggregates ${LIMIT_ARGS}
set +x
EOF
)
    JOB_IDS+=("$JOB_ID")
    echo "  part $PART/$PARTS ($NUM_S scenes) → job $JOB_ID"
done

if [ "${#JOB_IDS[@]}" -eq 0 ]; then
    echo "[ERROR] no worker jobs were submitted" >&2
    exit 1
fi

DEP=$(IFS=:; echo "${JOB_IDS[*]}")
AGG_JOB=$(/cluster/apps/slurm/bin/sbatch --parsable \
    --time="$AGG_TIME" \
    --cpus-per-task=2 \
    --mem-per-cpu=4G \
    --dependency="afterany:${DEP}" \
    --job-name="rtb_${EXP}_agg" \
    --output="$LOG_DIR/aggregate_%j.out" \
    --error="$LOG_DIR/aggregate_%j.err" \
    <<EOF
#!/bin/bash
set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg

set -x
python ${PY_SCRIPT} --exp ${EXP} --eval_root ${EVAL_ROOT} --aggregate_only
set +x
EOF
)

echo ""
echo "================================================================"
echo "Submitted ${#JOB_IDS[@]} worker jobs + 1 aggregator (job ${AGG_JOB})"
echo "Aggregator depends on (afterany): ${JOB_IDS[*]}"
echo ""
echo "Logs:    $LOG_DIR"
echo "Output:  $EVAL_ROOT/$EXP/scannetpp/<scene>/results.csv"
echo "         $EVAL_ROOT/$EXP/scannetpp/aggregate_results.csv"
echo "         $EVAL_ROOT/$EXP/scannetpp/aggregate_per_resolution.csv"
echo ""
echo "Cancel:  scancel ${JOB_IDS[*]} ${AGG_JOB}"
echo "================================================================"

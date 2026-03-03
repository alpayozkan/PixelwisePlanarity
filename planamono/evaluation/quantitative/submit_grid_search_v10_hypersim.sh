#!/bin/bash
# Submit v10 grid search jobs to SLURM for Hypersim — both affine and metric depth.
#
# Runs two independent pipelines:
#   1. Affine depth  (predict_batch_fast)   → seg_params_v10/
#   2. Metric depth  (predict_batch_fast_metric) → seg_params_v10_metric/
#
# Each pipeline: cache (1 GPU job) → eval (×N CPU jobs) → aggregate (1 job)
#
# Usage:
#   ./submit_grid_search_v10_hypersim.sh                          # both pipelines, 15 eval jobs each
#   ./submit_grid_search_v10_hypersim.sh --num-eval-jobs 10       # 10 eval jobs each
#   ./submit_grid_search_v10_hypersim.sh --only affine            # only affine
#   ./submit_grid_search_v10_hypersim.sh --only metric            # only metric
#   ./submit_grid_search_v10_hypersim.sh --eval-only              # skip cache for both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_EVAL_JOBS=15
EVAL_ONLY=false
ONLY_MODE=""  # "", "affine", or "metric"

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-eval-jobs)
            NUM_EVAL_JOBS="$2"
            shift 2
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --only)
            ONLY_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--num-eval-jobs N] [--eval-only] [--only affine|metric]"
            exit 1
            ;;
    esac
done

# Environment activation command
ENV_ACTIVATE="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono"
EVAL_HOURS=24

# ============================================================
# Helper: count configs from a YAML file
# ============================================================
count_configs() {
    local yaml_path="$1"
    python3 -c "
import yaml
with open('${yaml_path}') as f:
    cfg = yaml.safe_load(f)
base = cfg['base_config']
grid = cfg['grid']
mode = cfg.get('search_mode', 'one_at_a_time')
if mode == 'full_grid':
    n = 1
    for k in sorted(grid.keys()):
        n *= len(grid[k])
    print(n)
else:
    configs_set = set()
    configs_set.add(tuple(sorted(base.items())))
    count = 1
    for param_name in sorted(grid.keys()):
        for val in grid[param_name]:
            c = dict(base)
            c[param_name] = val
            c_tuple = tuple(sorted(c.items()))
            if c_tuple not in configs_set:
                configs_set.add(c_tuple)
                count += 1
    print(count)
"
}

# ============================================================
# Helper: submit one pipeline (cache → eval → aggregate)
# ============================================================
submit_pipeline() {
    local YAML_PATH="$1"
    local TAG="$2"  # "affine" or "metric", used for job name prefix

    if [ ! -f "$YAML_PATH" ]; then
        echo "ERROR: Config file not found: $YAML_PATH"
        return 1
    fi

    local NUM_CONFIGS
    NUM_CONFIGS=$(count_configs "$YAML_PATH")

    local ACTUAL_EVAL_JOBS=$NUM_EVAL_JOBS
    if [ "$ACTUAL_EVAL_JOBS" -gt "$NUM_CONFIGS" ]; then
        ACTUAL_EVAL_JOBS=$NUM_CONFIGS
    fi

    # Compute even distribution
    local JOB_RANGES
    JOB_RANGES=$(python3 -c "
n, j = ${NUM_CONFIGS}, ${ACTUAL_EVAL_JOBS}
base = n // j
extra = n % j
ranges = []
start = 0
for i in range(j):
    size = base + (1 if i < extra else 0)
    ranges.append(f'{start},{start+size}')
    start += size
print(' '.join(ranges))
")

    local MAX_CPJ
    MAX_CPJ=$(python3 -c "
n, j = ${NUM_CONFIGS}, ${ACTUAL_EVAL_JOBS}
print((n + j - 1) // j)
")

    local OUTPUT_DIR
    OUTPUT_DIR=$(python3 -c "
import yaml
with open('${YAML_PATH}') as f:
    cfg = yaml.safe_load(f)
print(cfg['output_dir'])
")
    local LOGS_DIR="${OUTPUT_DIR}/logs"
    mkdir -p "$LOGS_DIR"

    echo ""
    echo "============================================"
    echo "Pipeline: ${TAG} (Hypersim)"
    echo "============================================"
    echo "Config:        $YAML_PATH"
    echo "Total configs: $NUM_CONFIGS"
    echo "Eval jobs:     $ACTUAL_EVAL_JOBS (max $MAX_CPJ configs/job)"
    echo "Eval time:     ${EVAL_HOURS}h each"
    echo "Output:        $OUTPUT_DIR"
    echo "============================================"

    # --- Step 1: Cache (GPU) ---
    local DEPENDENCY_FLAG=""
    if [ "$EVAL_ONLY" = false ]; then
        local CACHE_JOB_ID
        CACHE_JOB_ID=$(sbatch \
            --job-name="gsv10h_${TAG}_cache" \
            --output="${LOGS_DIR}/cache_%j.out" \
            --error="${LOGS_DIR}/cache_%j.err" \
            --time=24:00:00 \
            --cpus-per-task=8 \
            --mem-per-cpu=8G \
            --gpus=rtx_3090:1 \
            --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python grid_search_v10_hypersim.py --yaml ${YAML_PATH} --mode cache'" \
            --parsable)
        echo "  [1/3] Cache: ${CACHE_JOB_ID} (GPU, 24h)"
        DEPENDENCY_FLAG="--dependency=afterok:${CACHE_JOB_ID}"
    else
        echo "  [1/3] Cache: SKIPPED (--eval-only)"
    fi

    # --- Step 2: Eval (CPU) ---
    local EVAL_JOB_IDS=""
    local JOB_IDX=0

    for RANGE in $JOB_RANGES; do
        local CFG_START="${RANGE%%,*}"
        local CFG_END="${RANGE##*,}"
        local N_IN_JOB=$((CFG_END - CFG_START))

        local JOB_ID
        JOB_ID=$(sbatch \
            --job-name="gsv10h_${TAG}_eval_${JOB_IDX}" \
            --output="${LOGS_DIR}/eval_job${JOB_IDX}_cfg${CFG_START}-${CFG_END}_%j.out" \
            --error="${LOGS_DIR}/eval_job${JOB_IDX}_cfg${CFG_START}-${CFG_END}_%j.err" \
            --time="${EVAL_HOURS}:00:00" \
            --cpus-per-task=4 \
            --mem-per-cpu=10G \
            ${DEPENDENCY_FLAG} \
            --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python grid_search_v10_hypersim.py --yaml ${YAML_PATH} --mode eval --config-start ${CFG_START} --config-end ${CFG_END}'" \
            --parsable)

        if [ -z "$EVAL_JOB_IDS" ]; then
            EVAL_JOB_IDS="${JOB_ID}"
        else
            EVAL_JOB_IDS="${EVAL_JOB_IDS}:${JOB_ID}"
        fi
        JOB_IDX=$((JOB_IDX + 1))
    done
    echo "  [2/3] Eval:  ${JOB_IDX} jobs (CPU, ${EVAL_HOURS}h)"

    # --- Step 3: Aggregate ---
    local AGG_JOB_ID
    AGG_JOB_ID=$(sbatch \
        --job-name="gsv10h_${TAG}_agg" \
        --output="${LOGS_DIR}/aggregate_%j.out" \
        --error="${LOGS_DIR}/aggregate_%j.err" \
        --time=24:00:00 \
        --cpus-per-task=1 \
        --mem-per-cpu=4G \
        --dependency=afterok:${EVAL_JOB_IDS} \
        --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python grid_search_v10_hypersim.py --yaml ${YAML_PATH} --mode aggregate'" \
        --parsable)
    echo "  [3/3] Agg:   ${AGG_JOB_ID}"
}

# ============================================================
# Submit pipelines
# ============================================================

if [ "$ONLY_MODE" != "metric" ]; then
    submit_pipeline "${SCRIPT_DIR}/grid_search_v10_hypersim_config.yaml" "affine"
fi

if [ "$ONLY_MODE" != "affine" ]; then
    submit_pipeline "${SCRIPT_DIR}/grid_search_v10_hypersim_config_metric.yaml" "metric"
fi

echo ""
echo "============================================"
echo "Monitor: squeue -u \$USER | grep gsv10h_"
echo "============================================"

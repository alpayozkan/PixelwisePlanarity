#!/bin/bash
# Submit grid search jobs to SLURM.
#
# Usage:
#   ./submit_grid_search.sh [path/to/config.yaml]           # all 3 steps
#   ./submit_grid_search.sh [path/to/config.yaml] --eval-only  # skip cache, eval + aggregate only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_PATH="${1:-${SCRIPT_DIR}/grid_search_config.yaml}"
EVAL_ONLY=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --eval-only) EVAL_ONLY=true ;;
    esac
done

if [ ! -f "$YAML_PATH" ]; then
    echo "ERROR: Config file not found: $YAML_PATH"
    exit 1
fi

# Count total configs using Python
NUM_CONFIGS=$(python3 -c "
import yaml, itertools
with open('${YAML_PATH}') as f:
    cfg = yaml.safe_load(f)
base = cfg['base_config']
grid = cfg['grid']
mode = cfg.get('search_mode', 'one_at_a_time')
if mode == 'full_grid':
    keys = sorted(grid.keys())
    n = 1
    for k in keys:
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
")

echo "============================================"
echo "Grid Search Submission"
echo "============================================"
echo "Config:      $YAML_PATH"
echo "Num configs: $NUM_CONFIGS"
echo "Eval only:   $EVAL_ONLY"
echo "============================================"

# Read output dir from YAML
OUTPUT_DIR=$(python3 -c "
import yaml
with open('${YAML_PATH}') as f:
    cfg = yaml.safe_load(f)
print(cfg['output_dir'])
")
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOGS_DIR"

# ============================================================
# Step 1: Cache job (GPU) — skip if --eval-only
# ============================================================
DEPENDENCY_FLAG=""
if [ "$EVAL_ONLY" = false ]; then
    CACHE_JOB_ID=$(sbatch \
        --job-name="gs_cache" \
        --output="${LOGS_DIR}/cache_%j.out" \
        --error="${LOGS_DIR}/cache_%j.err" \
        --time=2:00:00 \
        --cpus-per-task=8 \
        --mem-per-cpu=8G \
        --gpus=rtx_3090:1 \
        --wrap="bash -c 'source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono && cd ${SCRIPT_DIR} && python grid_search_segmentation.py --yaml ${YAML_PATH} --mode cache'" \
        --parsable)

    echo "[1/3] Cache job submitted: ${CACHE_JOB_ID}"
    DEPENDENCY_FLAG="--dependency=afterok:${CACHE_JOB_ID}"
else
    echo "[1/3] Cache job SKIPPED (--eval-only)"
fi

# ============================================================
# Step 2: Eval jobs (CPU)
# ============================================================
EVAL_JOB_IDS=""

for i in $(seq 0 $((NUM_CONFIGS - 1))); do
    JOB_ID=$(sbatch \
        --job-name="gs_eval_${i}" \
        --output="${LOGS_DIR}/eval_${i}_%j.out" \
        --error="${LOGS_DIR}/eval_${i}_%j.err" \
        --time=1:00:00 \
        --cpus-per-task=4 \
        --mem-per-cpu=10G \
        ${DEPENDENCY_FLAG} \
        --wrap="bash -c 'source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono && cd ${SCRIPT_DIR} && python grid_search_segmentation.py --yaml ${YAML_PATH} --mode eval --config-index ${i}'" \
        --parsable)

    if [ -z "$EVAL_JOB_IDS" ]; then
        EVAL_JOB_IDS="${JOB_ID}"
    else
        EVAL_JOB_IDS="${EVAL_JOB_IDS}:${JOB_ID}"
    fi
done

echo "[2/3] Eval jobs submitted: ${NUM_CONFIGS} jobs"

# ============================================================
# Step 3: Aggregate job (depends on all eval jobs)
# ============================================================
AGG_JOB_ID=$(sbatch \
    --job-name="gs_aggregate" \
    --output="${LOGS_DIR}/aggregate_%j.out" \
    --error="${LOGS_DIR}/aggregate_%j.err" \
    --time=00:10:00 \
    --cpus-per-task=1 \
    --mem-per-cpu=4G \
    --dependency=afterok:${EVAL_JOB_IDS} \
    --wrap="bash -c 'source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono && cd ${SCRIPT_DIR} && python grid_search_segmentation.py --yaml ${YAML_PATH} --mode aggregate'" \
    --parsable)

echo "[3/3] Aggregate job submitted: ${AGG_JOB_ID}"

echo ""
echo "============================================"
echo "Summary"
echo "============================================"
if [ "$EVAL_ONLY" = false ]; then
    echo "Cache job:     ${CACHE_JOB_ID} (GPU)"
fi
echo "Eval jobs:     ${NUM_CONFIGS} jobs (CPU)"
echo "Aggregate job: ${AGG_JOB_ID}"
echo "Logs dir:      ${LOGS_DIR}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "============================================"
echo ""
echo "Monitor: squeue -u \$USER | grep gs_"
echo "Results: cat ${OUTPUT_DIR}/grid_search_summary.csv"

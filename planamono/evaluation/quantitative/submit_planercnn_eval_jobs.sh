#!/bin/bash
# Submit PlaneRCNN evaluation jobs: one per dataset, all independent.
#
# Usage:
#   bash submit_planercnn_eval_jobs.sh                    # All 5 datasets
#   bash submit_planercnn_eval_jobs.sh scannetpp hypersim # Specific datasets
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_planercnn"
ENV_ACTIVATE="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono"

mkdir -p "$LOGS_DIR"

ALL_DATASETS=(scannetpp hypersim synthia vkitti2 pd)

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

echo "============================================"
echo "PlaneRCNN Evaluation: ${DATASETS[*]}"
echo "============================================"

JOB_IDS=()

for DS in "${DATASETS[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --job-name="planercnn_${DS}" \
        --time=24:00:00 \
        --cpus-per-task=16 \
        --mem-per-cpu=8G \
        --output="${LOGS_DIR}/${DS}_%j.out" \
        --error="${LOGS_DIR}/${DS}_%j.err" \
        --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python evaluate_planercnn.py --datasets ${DS}'")

    JOB_IDS+=("$JOB_ID")
    echo "  [${DS}] submitted: ${JOB_ID}"
    sleep 0.5
done

# Aggregate job: runs after ALL dataset jobs complete
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")
AGG_JOB_ID=$(sbatch --parsable \
    --job-name="planercnn_agg" \
    --time=00:10:00 \
    --cpus-per-task=1 \
    --mem-per-cpu=4G \
    --output="${LOGS_DIR}/aggregate_%j.out" \
    --error="${LOGS_DIR}/aggregate_%j.err" \
    --dependency="afterok:${DEP_STR}" \
    --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python evaluate_planercnn.py --aggregate-only'")

echo ""
echo "  [aggregate] submitted: ${AGG_JOB_ID} (after ${DEP_STR})"
echo ""
echo "Monitor: squeue -u \$USER -n planercnn_scannetpp,planercnn_hypersim,planercnn_synthia,planercnn_vkitti2,planercnn_pd,planercnn_agg"

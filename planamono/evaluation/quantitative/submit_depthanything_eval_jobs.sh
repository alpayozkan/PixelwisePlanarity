#!/bin/bash
# Submit DepthAnything evaluation jobs: one per dataset×variant pair.
#
# Usage:
#   bash submit_depthanything_eval_jobs.sh                  # all datasets, both variants
#   bash submit_depthanything_eval_jobs.sh scannetpp         # single dataset, both variants
#   bash submit_depthanything_eval_jobs.sh scannetpp hypersim # multiple datasets
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_depthanything"
ENV_ACTIVATE="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono"

mkdir -p "$LOGS_DIR"

ALL_DATASETS=(scannetpp hypersim synthia vkitti2)
VARIANTS=(dav2_normals moge_normals)

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

echo "============================================"
echo "DepthAnything Evaluation: ${DATASETS[*]}"
echo "============================================"

JOB_IDS=()

for DS in "${DATASETS[@]}"; do
    for VAR in "${VARIANTS[@]}"; do
        JOB_ID=$(sbatch --parsable \
            --job-name="da_${DS}_${VAR}" \
            --time=4:00:00 \
            --cpus-per-task=16 \
            --mem-per-cpu=4G \
            --output="${LOGS_DIR}/${DS}_${VAR}_%j.out" \
            --error="${LOGS_DIR}/${DS}_${VAR}_%j.err" \
            --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python evaluate_depthanything.py --datasets ${DS} --variants ${VAR}'")

        JOB_IDS+=("$JOB_ID")
        echo "  [${DS}/${VAR}] submitted: ${JOB_ID}"
        sleep 0.5
    done
done

# Aggregate job: runs after all evaluation jobs finish
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")
AGG_JOB_ID=$(sbatch --parsable \
    --job-name="da_agg" \
    --time=00:10:00 \
    --cpus-per-task=1 \
    --mem-per-cpu=4G \
    --output="${LOGS_DIR}/aggregate_%j.out" \
    --error="${LOGS_DIR}/aggregate_%j.err" \
    --dependency="afterok:${DEP_STR}" \
    --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python evaluate_depthanything.py --aggregate-only'")

echo ""
echo "  [aggregate] submitted: ${AGG_JOB_ID} (after ${DEP_STR})"
echo ""
echo "Monitor: squeue -u \$USER"

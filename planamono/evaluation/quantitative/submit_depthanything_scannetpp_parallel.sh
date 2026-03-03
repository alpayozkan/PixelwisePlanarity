#!/bin/bash
# Submit DepthAnything evaluation on ScanNet++ split across 4 parallel jobs.
# Each worker handles ~10-11 of the 42 scenes; a merge job runs after all complete.
#
# Usage:
#   bash submit_depthanything_scannetpp_parallel.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_depthanything"
ENV_ACTIVATE="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono"

NUM_PARTS=4
DATASET=scannetpp

mkdir -p "$LOGS_DIR"

echo "============================================"
echo "DepthAnything ScanNet++ — ${NUM_PARTS}-way parallel"
echo "============================================"

WORKER_IDS=()

for VAR in dav2_normals moge_normals; do
    echo ""
    echo "--- Variant: ${VAR} ---"
    for PART in 0 1 2 3; do
        JOB_ID=$(sbatch --parsable \
            --job-name="da_snpp_${VAR}_p${PART}" \
            --time=2:00:00 \
            --cpus-per-task=16 \
            --mem-per-cpu=4G \
            --output="${LOGS_DIR}/scannetpp_${VAR}_part${PART}_%j.out" \
            --error="${LOGS_DIR}/scannetpp_${VAR}_part${PART}_%j.err" \
            --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python evaluate_depthanything.py --datasets ${DATASET} --variants ${VAR} --part-id ${PART} --num-parts ${NUM_PARTS}'")
        WORKER_IDS+=("$JOB_ID")
        echo "  [part ${PART}/${NUM_PARTS} / ${VAR}] submitted: ${JOB_ID}"
    done
done

# Merge job: runs after all 8 workers finish
DEP_STR=$(IFS=:; echo "${WORKER_IDS[*]}")
MERGE_JOB=$(sbatch --parsable \
    --job-name="da_snpp_merge" \
    --time=00:15:00 \
    --cpus-per-task=2 \
    --mem-per-cpu=8G \
    --dependency="afterok:${DEP_STR}" \
    --output="${LOGS_DIR}/scannetpp_merge_%j.out" \
    --error="${LOGS_DIR}/scannetpp_merge_%j.err" \
    --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python evaluate_depthanything.py --datasets ${DATASET} --merge --num-parts ${NUM_PARTS}'")

echo ""
echo "  [merge] submitted: ${MERGE_JOB} (after ${DEP_STR})"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${LOGS_DIR}/"

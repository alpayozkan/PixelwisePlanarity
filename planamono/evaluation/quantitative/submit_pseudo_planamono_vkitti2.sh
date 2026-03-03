#!/bin/bash
LOGS_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_pseudo_planamono"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_ACTIVATE="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono"

mkdir -p "$LOGS_DIR"

JOB_ID=$(sbatch --parsable \
    --job-name="pseudo_planamono_vkitti2" \
    --time=4:00:00 \
    --cpus-per-task=16 \
    --mem-per-cpu=8G \
    --output="${LOGS_DIR}/vkitti2_%j.out" \
    --error="${LOGS_DIR}/vkitti2_%j.err" \
    --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python evaluate_vkitti2_all_baselines.py --methods pseudo_planamono --split test --variants clone'")

echo "Submitted: ${JOB_ID}"
echo "Logs: tail -f ${LOGS_DIR}/vkitti2_${JOB_ID}.out"

#!/bin/bash
# Submit single Metric3D evaluation job for ScanNet++ (full test set).
#
# Usage:
#   bash submit_metric3d_eval_scannetpp.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_metric3d"
ENV_ACTIVATE="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono"

mkdir -p "$LOGS_DIR"

JOB_ID=$(sbatch --parsable \
    --job-name="metric3d_scannetpp" \
    --time=8:00:00 \
    --cpus-per-task=16 \
    --mem-per-cpu=4G \
    --output="${LOGS_DIR}/scannetpp_%j.out" \
    --error="${LOGS_DIR}/scannetpp_%j.err" \
    --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python evaluate_metric3d.py --datasets scannetpp'")

echo "Submitted ScanNet++ Metric3D eval: job ${JOB_ID}"
echo "Monitor: squeue -j ${JOB_ID}"
echo "Log:     ${LOGS_DIR}/scannetpp_${JOB_ID}.out"

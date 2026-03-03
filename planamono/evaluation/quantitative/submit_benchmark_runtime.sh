#!/bin/bash
# Submit runtime benchmark jobs for both segmentation variants on both datasets.
#
# Usage:
#   ./submit_benchmark_runtime.sh              # all 4 jobs (scannetpp + hypersim, v5 + v6)
#   ./submit_benchmark_runtime.sh scannetpp    # scannetpp only (2 jobs)
#   ./submit_benchmark_runtime.sh hypersim     # hypersim only (2 jobs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_INIT="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono"

DATASET="${1:-all}"

submit_jobs() {
    local DATASET_NAME=$1
    local LOGS_DIR="/cluster/scratch/aoezkan/planeseg/${DATASET_NAME}/runtime/logs"
    mkdir -p "$LOGS_DIR"

    # Method 1: v5 segmentation (moge_mixed_bce_476644_ep6_v6)
    local JOB_V5=$(sbatch \
        --job-name="bench_${DATASET_NAME}_v5" \
        --output="${LOGS_DIR}/benchmark_v5_%j.out" \
        --error="${LOGS_DIR}/benchmark_v5_%j.err" \
        --time=8:00:00 \
        --cpus-per-task=8 \
        --mem-per-cpu=8G \
        --gpus=rtx_3090:1 \
        --wrap="bash -c '${CONDA_INIT} && cd ${SCRIPT_DIR} && python benchmark_runtime.py --method v5 --dataset ${DATASET_NAME}'" \
        --parsable)

    echo "  v5 (Sobel seg):    ${JOB_V5}"

    # Method 2: v6 segmentation (moge_mixed_bce_476644_ep6_v6seg_v6)
    local JOB_V6=$(sbatch \
        --job-name="bench_${DATASET_NAME}_v6" \
        --output="${LOGS_DIR}/benchmark_v6_%j.out" \
        --error="${LOGS_DIR}/benchmark_v6_%j.err" \
        --time=8:00:00 \
        --cpus-per-task=8 \
        --mem-per-cpu=8G \
        --gpus=rtx_3090:1 \
        --wrap="bash -c '${CONDA_INIT} && cd ${SCRIPT_DIR} && python benchmark_runtime.py --method v6 --dataset ${DATASET_NAME}'" \
        --parsable)

    echo "  v6 (pairwise seg): ${JOB_V6}"
    echo "  Logs: ${LOGS_DIR}"
    echo "  Results: /cluster/scratch/aoezkan/planeseg/${DATASET_NAME}/runtime/"
}

echo ""
echo "============================================"
echo "Runtime Benchmark Jobs"
echo "============================================"

if [ "$DATASET" = "all" ] || [ "$DATASET" = "scannetpp" ]; then
    echo ""
    echo "--- ScanNet++ ---"
    submit_jobs "scannetpp"
fi

if [ "$DATASET" = "all" ] || [ "$DATASET" = "hypersim" ]; then
    echo ""
    echo "--- Hypersim ---"
    submit_jobs "hypersim"
fi

echo ""
echo "============================================"
echo "Monitor: squeue -u \$USER | grep bench_"
echo "============================================"

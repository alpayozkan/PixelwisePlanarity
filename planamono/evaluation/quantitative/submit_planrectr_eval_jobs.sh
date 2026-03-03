#!/bin/bash
# Submit PlanRecTR evaluation jobs: one job per (resolution, dataset) combination.
#
# Usage:
#   bash submit_planrectr_eval_jobs.sh                              # Both resolutions, all datasets
#   bash submit_planrectr_eval_jobs.sh --resolution lowres          # lowres only
#   bash submit_planrectr_eval_jobs.sh --resolution highres         # highres only
#   bash submit_planrectr_eval_jobs.sh --datasets scannetpp hypersim  # Specific datasets
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_planrectr"
ENV_ACTIVATE="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono"

mkdir -p "$LOGS_DIR"

# Parse arguments
RESOLUTIONS=()
DATASETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resolution)
            RESOLUTIONS+=("$2")
            shift 2
            ;;
        --datasets)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATASETS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ ${#RESOLUTIONS[@]} -eq 0 ]; then
    RESOLUTIONS=(lowres highres)
fi
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(scannetpp hypersim synthia vkitti2)
fi

echo "============================================"
echo "PlanRecTR Evaluation"
echo "  Resolutions: ${RESOLUTIONS[*]}"
echo "  Datasets:    ${DATASETS[*]}"
echo "============================================"

for RES in "${RESOLUTIONS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        JOB_ID=$(sbatch --parsable \
            --job-name="planrectr_${RES}_${DS}" \
            --time=4:00:00 \
            --cpus-per-task=16 \
            --mem-per-cpu=8G \
            --output="${LOGS_DIR}/${RES}_${DS}_%j.out" \
            --error="${LOGS_DIR}/${RES}_${DS}_%j.err" \
            --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python evaluate_planrectr.py --resolution ${RES} --datasets ${DS}'")
        echo "  [${RES}/${DS}] submitted: ${JOB_ID}"
    done
done

echo ""
echo "Monitor: squeue -u \$USER | grep planrectr"
echo "Logs:    ${LOGS_DIR}/"

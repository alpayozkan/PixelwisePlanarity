#!/bin/bash
# Submit PlaneTR evaluation jobs: one per (resolution, dataset) combination.
# Runs both lowres and highres by default.
#
# Usage:
#   bash submit_planetr_eval_jobs.sh                              # All datasets, both resolutions
#   bash submit_planetr_eval_jobs.sh --resolution lowres          # All datasets, lowres only
#   bash submit_planetr_eval_jobs.sh --resolution highres         # All datasets, highres only
#   bash submit_planetr_eval_jobs.sh --datasets scannetpp hypersim  # Specific datasets
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_planetr"
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

# Defaults
if [ ${#RESOLUTIONS[@]} -eq 0 ]; then
    RESOLUTIONS=(lowres highres)
fi
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(scannetpp hypersim synthia vkitti2)
fi

echo "============================================"
echo "PlaneTR Evaluation"
echo "  Resolutions: ${RESOLUTIONS[*]}"
echo "  Datasets:    ${DATASETS[*]}"
echo "============================================"

ALL_JOB_IDS=()

for RES in "${RESOLUTIONS[@]}"; do
    RES_JOB_IDS=()
    DS_ARGS=$(IFS=' '; echo "${DATASETS[*]}")

    for DS in "${DATASETS[@]}"; do
        JOB_ID=$(sbatch --parsable \
            --job-name="planetr_${RES}_${DS}" \
            --time=4:00:00 \
            --cpus-per-task=16 \
            --mem-per-cpu=8G \
            --output="${LOGS_DIR}/${RES}_${DS}_%j.out" \
            --error="${LOGS_DIR}/${RES}_${DS}_%j.err" \
            --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python evaluate_planetr.py --resolution ${RES} --datasets ${DS}'")

        RES_JOB_IDS+=("$JOB_ID")
        ALL_JOB_IDS+=("$JOB_ID")
        echo "  [${RES}/${DS}] submitted: ${JOB_ID}"
        sleep 0.5
    done

    # Aggregate job per resolution: runs after all dataset jobs for this resolution
    DEP_STR=$(IFS=:; echo "${RES_JOB_IDS[*]}")
    AGG_JOB_ID=$(sbatch --parsable \
        --job-name="planetr_${RES}_agg" \
        --time=00:10:00 \
        --cpus-per-task=1 \
        --mem-per-cpu=4G \
        --output="${LOGS_DIR}/${RES}_aggregate_%j.out" \
        --error="${LOGS_DIR}/${RES}_aggregate_%j.err" \
        --dependency="afterok:${DEP_STR}" \
        --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python evaluate_planetr.py --resolution ${RES} --aggregate-only'")

    ALL_JOB_IDS+=("$AGG_JOB_ID")
    echo "  [${RES}/aggregate] submitted: ${AGG_JOB_ID} (after ${DEP_STR})"
    echo ""
done

echo "All jobs submitted: ${ALL_JOB_IDS[*]}"
echo "Monitor: squeue -u \$USER | grep planetr"

#!/bin/bash
# Submit pseudo-planamono evaluation jobs: one per dataset, all independent.
#
# Usage:
#   bash submit_pseudo_planamono_eval.sh                    # All datasets
#   bash submit_pseudo_planamono_eval.sh scannetpp hypersim # Specific datasets
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_pseudo_planamono"
ENV_ACTIVATE="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono"

mkdir -p "$LOGS_DIR"

# Dataset → evaluation script mapping
declare -A SCRIPTS
SCRIPTS[scannetpp]="evaluate_all_baselines.py"
SCRIPTS[hypersim]="evaluate_hypersim_all_baselines.py"
SCRIPTS[synthia]="evaluate_synthia_all_baselines.py"
SCRIPTS[vkitti2]="evaluate_vkitti2_all_baselines.py"

ALL_DATASETS=(scannetpp hypersim synthia vkitti2)

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

echo "============================================"
echo "Pseudo-Planamono Evaluation: ${DATASETS[*]}"
echo "============================================"

JOB_IDS=()

for DS in "${DATASETS[@]}"; do
    EVAL_SCRIPT="${SCRIPTS[$DS]}"
    JOB_ID=$(sbatch --parsable \
        --job-name="pseudo_planamono_${DS}" \
        --time=12:00:00 \
        --cpus-per-task=16 \
        --mem-per-cpu=8G \
        --output="${LOGS_DIR}/${DS}_%j.out" \
        --error="${LOGS_DIR}/${DS}_%j.err" \
        --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python ${EVAL_SCRIPT} --methods pseudo_planamono'")

    JOB_IDS+=("$JOB_ID")
    echo "  [${DS}] submitted: ${JOB_ID}"
    sleep 0.5
done

echo ""
echo "Monitor: squeue -u \$USER -n pseudo_planamono_scannetpp,pseudo_planamono_hypersim,pseudo_planamono_synthia,pseudo_planamono_vkitti2"
echo "Logs:    tail -f ${LOGS_DIR}/*_*.out"

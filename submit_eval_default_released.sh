#!/bin/bash
# Evaluate ZeroPlane default released model on ScanNet++ (5 scenes, 3 thresholds)
# Then aggregate with all other ablation results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_scannetpp"
mkdir -p "${LOG_DIR}"

sbatch \
    --job-name="eval_zp_default_released" \
    --output="${LOG_DIR}/eval_zp_default_released.log" \
    --error="${LOG_DIR}/eval_zp_default_released.log" \
    --time=4:00:00 \
    --mem-per-cpu=8G \
    --cpus-per-task=16 \
    --wrap="eval \"\$(/cluster/scratch/aoezkan/miniconda3/condabin/conda shell.bash hook)\" && conda activate planamono && cd ${SCRIPT_DIR} && python planamono/evaluation/quantitative/evaluate_zeroplane_ablations.py --experiments default_dust3r_released --max-scenes 5"

#!/usr/bin/env bash
# Submit GT quality evaluation jobs for all 3 methods.
#
# Usage:
#   bash submit_gt_quality_eval_jobs.sh
#
# After all jobs finish, run aggregation:
#   python evaluate_gt_quality.py --aggregate-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/evaluate_gt_quality.py"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

for METHOD in our_gt_scannetpp planercnn_gt_scannetpp scannet_gt; do
    echo "Submitting ${METHOD}..."
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gt_qual_${METHOD}
#SBATCH --output=${LOG_DIR}/gt_quality_${METHOD}_%j.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1

source ~/.bashrc
conda activate planeseg

echo "=== GT Quality Evaluation: ${METHOD} ==="
echo "Job ID: \${SLURM_JOB_ID}"
echo "Node:   \$(hostname)"
echo "Start:  \$(date)"

python "${SCRIPT}" --method ${METHOD}

echo "End:    \$(date)"
EOF
done

echo ""
echo "All jobs submitted. After completion, run:"
echo "  python ${SCRIPT} --aggregate-only"

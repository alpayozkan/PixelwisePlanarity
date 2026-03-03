#!/bin/bash
# Submit a SLURM job to add extra method visuals to existing qualitative samples.
#
# Usage:
#   bash submit_planercnn_gt_qualitative.sh                          # both methods
#   bash submit_planercnn_gt_qualitative.sh --methods planercnn_gt   # just one
#   bash submit_planercnn_gt_qualitative.sh --methods planar_recon

set -euo pipefail

METHODS="${*}"  # pass all args through

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/add_planercnn_gt_qualitative.py"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/qualitative"
mkdir -p "${LOG_DIR}"

JOB=$(sbatch --parsable \
    --job-name="qual_extra" \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem-per-cpu=4G \
    --output="${LOG_DIR}/qual_extra_%j.out" \
    --error="${LOG_DIR}/qual_extra_%j.err" \
    --wrap="
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg
cd ${SCRIPT_DIR}
python ${SCRIPT} ${METHODS}
")

echo "Submitted job ${JOB}"
echo "Logs: ${LOG_DIR}/qual_extra_${JOB}.{out,err}"

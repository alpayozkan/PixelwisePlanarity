#!/bin/bash
# Submit 4 parallel SLURM jobs for Metric3D ScanNet++ evaluation.
#
# Splits 42 test scenes evenly across 4 workers, then submits a 5th
# merge job that runs after all workers finish.
#
# Usage:
#   bash submit_metric3d_eval_scannetpp_parallel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/evaluate_metric3d.py"
LOGS_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_metric3d"
ENV_ACTIVATE="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono"

mkdir -p "$LOGS_DIR"

# 42 scenes split across 4 jobs: [0,11) [11,22) [22,32) [32,42)
SHARDS=("0 11" "11 22" "22 32" "32 42")
N_JOBS=${#SHARDS[@]}

echo "Submitting $N_JOBS parallel ScanNet++ eval jobs..."

JOB_IDS=()
for SHARD in "${SHARDS[@]}"; do
    START=$(echo $SHARD | awk '{print $1}')
    END=$(echo $SHARD | awk '{print $2}')

    JOB_ID=$(sbatch --parsable \
        --job-name="metric3d_snpp_${START}_${END}" \
        --time=4:00:00 \
        --cpus-per-task=16 \
        --mem-per-cpu=4G \
        --output="${LOGS_DIR}/scannetpp_shard_${START}_${END}_%j.out" \
        --error="${LOGS_DIR}/scannetpp_shard_${START}_${END}_%j.err" \
        --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python ${SCRIPT} --datasets scannetpp --scene-start ${START} --scene-end ${END}'")

    JOB_IDS+=("$JOB_ID")
    echo "  Shard [${START}:${END}) → job ${JOB_ID}"
done

# Merge job: runs after all workers succeed
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")
MERGE_JOB_ID=$(sbatch --parsable \
    --job-name="metric3d_snpp_merge" \
    --time=00:10:00 \
    --cpus-per-task=1 \
    --mem-per-cpu=4G \
    --output="${LOGS_DIR}/scannetpp_merge_%j.out" \
    --error="${LOGS_DIR}/scannetpp_merge_%j.err" \
    --dependency="afterok:${DEP_STR}" \
    --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python ${SCRIPT} --datasets scannetpp --merge-shards'")

echo "  Merge job → ${MERGE_JOB_ID} (after ${DEP_STR})"
echo ""
echo "Monitor: squeue -u \$USER -n metric3d_snpp_0_11,metric3d_snpp_11_22,metric3d_snpp_22_32,metric3d_snpp_32_42,metric3d_snpp_merge"
echo "Logs:    ${LOGS_DIR}/scannetpp_shard_*"

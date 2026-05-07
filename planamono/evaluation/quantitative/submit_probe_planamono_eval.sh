#!/bin/bash
# Submit 5 parallel SLURM jobs for probe_planamono ScanNet++ evaluation.
#
# Splits 42 test scenes across 5 workers, then submits a merge job
# that runs after all workers finish.
#
# Results saved to: /cluster/scratch/aoezkan/planeseg/scannetpp/eval/probe_planamono_v6/
#
# Usage:
#   bash submit_probe_planamono_eval.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/evaluate_probe_planamono.py"
LOGS_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_probe_planamono"
ENV_ACTIVATE="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono"

mkdir -p "$LOGS_DIR"

# 42 scenes split across 5 jobs: ~8-9 scenes each
TOTAL_SCENES=42
N_JOBS=5
CHUNK=$(( (TOTAL_SCENES + N_JOBS - 1) / N_JOBS ))  # ceiling division = 9

echo "============================================"
echo "probe_planamono Parallel Evaluation"
echo "  Total scenes: ${TOTAL_SCENES}"
echo "  Jobs: ${N_JOBS}, ~${CHUNK} scenes each"
echo "  Logs: ${LOGS_DIR}"
echo "============================================"

JOB_IDS=()

for i in $(seq 0 $((N_JOBS - 1))); do
    START=$((i * CHUNK))
    END=$(( (i + 1) * CHUNK ))
    if [ $END -gt $TOTAL_SCENES ]; then
        END=$TOTAL_SCENES
    fi
    if [ $START -ge $TOTAL_SCENES ]; then
        break
    fi

    JOB_ID=$(sbatch --parsable \
        --job-name="probe_pm_${START}_${END}" \
        --time=8:00:00 \
        --cpus-per-task=16 \
        --mem-per-cpu=4G \
        --output="${LOGS_DIR}/shard_${START}_${END}_%j.out" \
        --error="${LOGS_DIR}/shard_${START}_${END}_%j.err" \
        --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python ${SCRIPT} --scene-start ${START} --scene-end ${END}'")

    JOB_IDS+=("$JOB_ID")
    echo "  Shard [${START}:${END}) → job ${JOB_ID}"
    sleep 0.2
done

# Merge job: runs after all workers succeed
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")
MERGE_JOB_ID=$(sbatch --parsable \
    --job-name="probe_pm_merge" \
    --time=00:10:00 \
    --cpus-per-task=1 \
    --mem-per-cpu=4G \
    --output="${LOGS_DIR}/merge_%j.out" \
    --error="${LOGS_DIR}/merge_%j.err" \
    --dependency="afterok:${DEP_STR}" \
    --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python ${SCRIPT} --merge-shards'")

echo "  Merge → job ${MERGE_JOB_ID} (after ${DEP_STR})"
echo ""
echo "============================================"
echo "Submitted $((${#JOB_IDS[@]} + 1)) jobs total"
echo "Monitor: squeue -u \$USER | grep probe_pm"
echo "============================================"

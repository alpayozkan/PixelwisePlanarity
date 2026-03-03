#!/bin/bash
#
# Submit 5 parallel SLURM jobs to download Hypersim RGB + depth.
# Each job gets ~92 scenes (457 total / 5 jobs).
#
# Usage:
#   bash planamono/shared/datasets/submit_download_hypersim.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_hypersim.py"
OUTPUT_DIR="/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
SPLIT_DIR="${SCRIPT_DIR}/../../splits/hypersim"
LOG_DIR="${SCRIPT_DIR}/logs"
N_JOBS=5

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Collect all unique scene IDs from all splits
ALL_SCENES=$(cat "$SPLIT_DIR"/train.txt "$SPLIT_DIR"/val.txt "$SPLIT_DIR"/test.txt | sort -u)
TOTAL=$(echo "$ALL_SCENES" | wc -l)
CHUNK_SIZE=$(( (TOTAL + N_JOBS - 1) / N_JOBS ))

echo "[INFO] Total scenes: $TOTAL"
echo "[INFO] Jobs: $N_JOBS, ~$CHUNK_SIZE scenes per job"
echo "[INFO] Output: $OUTPUT_DIR"
echo "[INFO] Logs: $LOG_DIR"
echo ""

# Split scenes into chunks and submit jobs
JOB_IDX=0
echo "$ALL_SCENES" | split -l "$CHUNK_SIZE" -d -a 1 - /tmp/hypersim_chunk_

for chunk_file in /tmp/hypersim_chunk_*; do
    CHUNK_SCENES=$(cat "$chunk_file" | tr '\n' ' ')
    N_CHUNK=$(cat "$chunk_file" | wc -l)

    echo "[INFO] Job $JOB_IDX: $N_CHUNK scenes"

    sbatch \
        --job-name="dl_hypersim_${JOB_IDX}" \
        --output="${LOG_DIR}/download_hypersim_${JOB_IDX}_%j.out" \
        --error="${LOG_DIR}/download_hypersim_${JOB_IDX}_%j.err" \
        --time=24:00:00 \
        --cpus-per-task=2 \
        --mem-per-cpu=4G \
        --wrap="eval \"\$(conda shell.bash hook 2>/dev/null)\"; conda activate planamono; python ${DOWNLOAD_SCRIPT} -d ${OUTPUT_DIR} --scenes ${CHUNK_SCENES}"

    JOB_IDX=$(( JOB_IDX + 1 ))
done

# Cleanup
rm -f /tmp/hypersim_chunk_*

echo ""
echo "[INFO] Submitted $JOB_IDX jobs. Monitor with: squeue -u \$USER"
echo "[INFO] Check logs: ls ${LOG_DIR}/download_hypersim_*.out"

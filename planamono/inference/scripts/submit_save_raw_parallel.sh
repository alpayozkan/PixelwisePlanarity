#!/bin/bash
# Submit parallel MoGe raw inference as separate SLURM jobs.
#
# Usage:
#   bash submit_save_raw_parallel.sh                  # 10 parts, defaults
#   bash submit_save_raw_parallel.sh 5                # 5 parts
#   NUM_PARTS=20 MODEL_PATH=/path/to/model.pt bash submit_save_raw_parallel.sh
#
# All parts write to the same OUTPUT_ROOT (one subdir per scene, no conflicts).
# After all complete, run segment_from_raw.py on the combined output.

set -euo pipefail

# ── Configuration (override via environment variables) ──
NUM_PARTS="${NUM_PARTS:-${1:-10}}"
MODEL_PATH="${MODEL_PATH:-/cluster/scratch/ayavuz/moge_HIRES_4datasets/model_epoch2.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/scratch/aoezkan/planeseg/scannetpp/inference_raw/moge_hires_4ds_ep2_raw}"
DATASET_DIR="${DATASET_DIR:-/cluster/scratch/aoezkan/planeseg/dataset/scannetpp}"
RGB_ROOT="${RGB_ROOT:-/cluster/project/cvg/Shared_datasets/scannet++/data}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TOKENS="${NUM_TOKENS:-1600}"
ARCHITECTURE="${ARCHITECTURE:-4head}"
METRIC_DEPTH="${METRIC_DEPTH:---metric_depth}"

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/inference"
mkdir -p "$LOG_DIR"

METRIC_FLAG=""
if [ -n "$METRIC_DEPTH" ]; then
    METRIC_FLAG="--metric_depth"
fi

echo "============================================================"
echo "Submitting parallel MoGe raw inference"
echo "  Parts:        $NUM_PARTS"
echo "  Model:        $MODEL_PATH"
echo "  Output:       $OUTPUT_ROOT"
echo "  Split:        $SPLIT"
echo "  Architecture: $ARCHITECTURE"
echo "  Metric depth: ${METRIC_DEPTH:-no}"
echo "============================================================"

JOB_IDS=()

for PART_ID in $(seq 0 $((NUM_PARTS - 1))); do
    JOB_ID=$(sbatch --parsable \
        --time=4:00:00 \
        --cpus-per-task=4 \
        --mem-per-cpu=8G \
        --gpus=1 \
        --gres=gpumem:24g \
        --output="${LOG_DIR}/save_raw_part${PART_ID}_of${NUM_PARTS}_%j.out" \
        --error="${LOG_DIR}/save_raw_part${PART_ID}_of${NUM_PARTS}_%j.err" \
        <<EOF
#!/bin/bash
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

python ${PROJECT_ROOT}/inference/planarity/save_moge_raw_parallel.py \
    --model_path ${MODEL_PATH} \
    --output_root ${OUTPUT_ROOT} \
    --dataset_dir ${DATASET_DIR} \
    --rgb_root ${RGB_ROOT} \
    --split ${SPLIT} \
    --batch_size ${BATCH_SIZE} \
    --num_tokens ${NUM_TOKENS} \
    --architecture ${ARCHITECTURE} \
    --part_id ${PART_ID} \
    --num_parts ${NUM_PARTS} \
    ${METRIC_FLAG}
EOF
)
    JOB_IDS+=("$JOB_ID")
    echo "  Part ${PART_ID}/${NUM_PARTS} → job $JOB_ID"
done

echo ""
echo "Submitted ${NUM_PARTS} jobs: ${JOB_IDS[*]}"
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${LOG_DIR}/save_raw_part*_of${NUM_PARTS}_*.out"
echo ""
echo "Cancel all: scancel ${JOB_IDS[*]}"

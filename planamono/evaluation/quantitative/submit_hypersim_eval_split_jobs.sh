#!/bin/bash
# Split-based Hypersim evaluation: splits test scenes into N parts and submits
# one SLURM job per part. After all jobs finish, run the merge command.
#
# Usage:
#   ./submit_hypersim_eval_split_jobs.sh                          # 15 parts, method=gt
#   ./submit_hypersim_eval_split_jobs.sh --num-parts 10           # 10 parts
#   ./submit_hypersim_eval_split_jobs.sh --methods gt moge_ours   # Multiple methods
#   ./submit_hypersim_eval_split_jobs.sh --inlier-gates 0.5 0.7 0.8 0.9  # Multi-gate
#   ./submit_hypersim_eval_split_jobs.sh --time 2:00:00           # Custom time limit

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
SPLITS_DIR="${PROJECT_ROOT}/splits/hypersim"
SPLIT_FILE="${SPLITS_DIR}/test.txt"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_hypersim_split"
SPLIT_TMP_DIR="/cluster/scratch/aoezkan/planeseg/tmp/scene_splits"

# Defaults
NUM_PARTS=15
METHODS=("gt")
SPLIT="test"
TIME_LIMIT="4:00:00"
CPUS=16
MEM_PER_CPU="8G"
INLIER_GATES_ARG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-parts)
            NUM_PARTS="$2"
            shift 2
            ;;
        --methods)
            METHODS=()
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                METHODS+=("$1")
                shift
            done
            ;;
        --split)
            SPLIT="$2"
            SPLIT_FILE="${SPLITS_DIR}/${SPLIT}.txt"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --mem-per-cpu)
            MEM_PER_CPU="$2"
            shift 2
            ;;
        --inlier-gates)
            INLIER_GATES_ARG="--inlier-gates"
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                INLIER_GATES_ARG="$INLIER_GATES_ARG $1"
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate
if [ ! -f "$SPLIT_FILE" ]; then
    echo "[ERROR] Split file not found: $SPLIT_FILE"
    exit 1
fi

NUM_SCENES=$(wc -l < "$SPLIT_FILE")
METHODS_STR="${METHODS[*]}"

echo "============================================================"
echo "Split-based Hypersim Evaluation"
echo "============================================================"
echo "Split file:   $SPLIT_FILE ($NUM_SCENES scenes)"
echo "Num parts:    $NUM_PARTS"
echo "Methods:      $METHODS_STR"
echo "Time limit:   $TIME_LIMIT"
echo "CPUs:         $CPUS"
echo "Mem/CPU:      $MEM_PER_CPU"
if [ -n "$INLIER_GATES_ARG" ]; then
    echo "Inlier gates: $INLIER_GATES_ARG"
fi
echo "============================================================"
echo ""

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$SPLIT_TMP_DIR"

# Split scene file into N parts
echo "[1/3] Splitting $NUM_SCENES scenes into $NUM_PARTS parts..."

# Clean up any previous split files
rm -f "${SPLIT_TMP_DIR}"/scenes_part*.txt

# Calculate lines per part (ceiling division)
LINES_PER_PART=$(( (NUM_SCENES + NUM_PARTS - 1) / NUM_PARTS ))

# Read scenes into array and split manually (avoids split's 0-based naming issues)
mapfile -t ALL_SCENES < "$SPLIT_FILE"
PART_FILES=()
PART_IDX=1
for (( i=0; i<${#ALL_SCENES[@]}; i+=LINES_PER_PART )); do
    PART_ID=$(printf '%02d' $PART_IDX)
    PART_FILE="${SPLIT_TMP_DIR}/scenes_part${PART_ID}.txt"
    # Write this chunk of scenes
    printf '%s\n' "${ALL_SCENES[@]:$i:$LINES_PER_PART}" > "$PART_FILE"
    N=$(wc -l < "$PART_FILE")
    PART_FILES+=("$PART_FILE")
    echo "  Part $PART_ID: $N scenes"
    PART_IDX=$((PART_IDX + 1))
done

ACTUAL_PARTS=${#PART_FILES[@]}
echo ""
echo "Created $ACTUAL_PARTS part files"
echo ""

# Submit SLURM jobs
echo "[2/3] Submitting $ACTUAL_PARTS SLURM jobs..."
echo ""

JOB_IDS=()
for i in $(seq 1 $ACTUAL_PARTS); do
    PART_ID=$(printf '%02d' $i)
    SCENE_LIST="${SPLIT_TMP_DIR}/scenes_part${PART_ID}.txt"

    echo "Submitting part $PART_ID ($(wc -l < "$SCENE_LIST") scenes)..."

    JOB_SCRIPT="/tmp/eval_hypersim_split_${PART_ID}_${RANDOM}.sh"

    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=eval_hyp_p${PART_ID}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem-per-cpu=${MEM_PER_CPU}
#SBATCH --output=${LOG_DIR}/part${PART_ID}_%j.out
#SBATCH --error=${LOG_DIR}/part${PART_ID}_%j.err

echo "============================================================"
echo "Hypersim Split Evaluation - Part ${PART_ID}"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start: \$(date)"
echo "============================================================"

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

cd ${SCRIPT_DIR}

python evaluate_hypersim_gt_split.py \\
    --methods ${METHODS_STR} \\
    --scene-list ${SCENE_LIST} \\
    --part-id ${PART_ID} \\
    --split ${SPLIT} \\
    ${INLIER_GATES_ARG}

EXIT_CODE=\$?

echo ""
echo "============================================================"
echo "End: \$(date)"
echo "Exit code: \$EXIT_CODE"
echo "============================================================"

exit \$EXIT_CODE
EOF

    JOB_OUTPUT=$(sbatch "$JOB_SCRIPT")
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
    JOB_IDS+=("$JOB_ID")

    echo "  -> Job $JOB_ID"

    rm "$JOB_SCRIPT"
    sleep 0.3
done

# Print summary
echo ""
echo "============================================================"
echo "[3/3] Summary"
echo "============================================================"
echo "Submitted ${#JOB_IDS[@]} jobs:"
for i in $(seq 0 $((${#JOB_IDS[@]} - 1))); do
    PART_ID=$(printf '%02d' $((i + 1)))
    echo "  Part $PART_ID: Job ${JOB_IDS[$i]}"
done

echo ""
echo "After all jobs finish, merge results:"
echo ""
echo "  cd $SCRIPT_DIR"
echo "  python evaluate_hypersim_gt_split.py \\"
echo "      --methods $METHODS_STR \\"
echo "      --merge --num-parts $ACTUAL_PARTS $INLIER_GATES_ARG"
echo ""
echo "Check status:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f ${LOG_DIR}/part*_*.out"
echo ""
echo "Cancel all jobs:"
ALL_IDS=$(IFS=,; echo "${JOB_IDS[*]}")
echo "  scancel $ALL_IDS"
echo "============================================================"

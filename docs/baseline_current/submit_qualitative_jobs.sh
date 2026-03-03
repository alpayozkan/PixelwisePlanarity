#!/bin/bash
# Submit one SLURM job per (dataset, method) pair for qualitative visualization.
# All jobs use the same seed/n-samples so frames are consistent across methods.
#
# Usage:
#   ./submit_qualitative_jobs.sh                        # Submit all datasets × methods
#   ./submit_qualitative_jobs.sh scannetpp hypersim     # Submit specific datasets only

N_SAMPLES=50
SEED=42
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/generate_qualitative_report.py"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/qualitative"
mkdir -p "${LOG_DIR}"

# Parse dataset arguments (default: all)
if [ $# -eq 0 ]; then
    DATASETS=("scannetpp" "hypersim" "synthia" "vkitti2")
else
    DATASETS=("$@")
fi

METHODS=(
    "ours"
    "ours_indoors"
    "zeroplane_released"
    "zeroplane_finetuned_dinov2_60k"
    "zeroplane_finetuned_indoors_dinov2_60k"
    "planercnn"
    "planeTR"
    "moge_4ds_ep2_v5_rel"
    "moge_4ds_ep2_v5origparams_rel"
)

echo "============================================================"
echo "Submitting Qualitative Visualization Jobs"
echo "============================================================"
echo "Datasets: ${DATASETS[@]}"
echo "Methods:  ${METHODS[@]}"
echo "Samples:  ${N_SAMPLES} (seed=${SEED})"
echo "============================================================"
echo ""

JOB_IDS=()
for DS in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        JOB_NAME="qual_${DS}_${METHOD}"
        echo "Submitting ${JOB_NAME}..."

        JOB_SCRIPT="/tmp/${JOB_NAME}_${RANDOM}.sh"

        cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

echo "============================================================"
echo "Qualitative: ${DS} / ${METHOD}"
echo "============================================================"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start time: \$(date)"
echo "============================================================"
echo ""

# Activate conda environment
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

cd ${SCRIPT_DIR}

python ${SCRIPT} \
    --datasets ${DS} \
    --methods ${METHOD} \
    --n-samples ${N_SAMPLES} \
    --seed ${SEED}

EXIT_CODE=\$?

echo ""
echo "============================================================"
echo "End time: \$(date)"
echo "Exit code: \$EXIT_CODE"
echo "============================================================"

exit \$EXIT_CODE
EOF

        JOB_OUTPUT=$(sbatch "$JOB_SCRIPT")
        JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
        JOB_IDS+=("$JOB_ID")

        echo "  → Submitted as job $JOB_ID"

        rm "$JOB_SCRIPT"
        sleep 0.5
    done
done

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Submitted ${#JOB_IDS[@]} jobs."
echo ""
echo "Check status:"
echo "  squeue -u \$USER"
echo ""
echo "View logs (after jobs start):"
echo "  tail -f ${LOG_DIR}/qual_*_*.out"
echo ""
echo "Output directory:"
echo "  /cluster/scratch/aoezkan/planeseg/eval/qualitative/<dataset>/{segmentation,inliers}/<method>/"
echo ""

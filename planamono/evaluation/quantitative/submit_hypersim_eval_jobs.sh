#!/bin/bash
# Submit separate SLURM jobs for each Hypersim evaluation method
#
# Usage:
#   ./submit_hypersim_eval_jobs.sh                    # Submit all methods
#   ./submit_hypersim_eval_jobs.sh gt ours_mixed      # Submit specific methods

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"

# Methods to evaluate (pass as arguments or use default)
if [ $# -eq 0 ]; then
    METHODS=("gt" "ours_mixed" "ours")
else
    METHODS=("$@")
fi

# Create logs directory
mkdir -p /cluster/scratch/aoezkan/planeseg/logs/eval_hypersim

echo "============================================================"
echo "Submitting Hypersim Evaluation Jobs"
echo "============================================================"
echo "Methods: ${METHODS[@]}"
echo "============================================================"
echo ""

# Submit a job for each method
JOB_IDS=()
for METHOD in "${METHODS[@]}"; do
    echo "Submitting job for method: $METHOD"

    # Create a temporary job script
    JOB_SCRIPT="/tmp/eval_hypersim_${METHOD}_${RANDOM}.sh"

    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=eval_${METHOD}
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/eval_hypersim/eval_${METHOD}_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/eval_hypersim/eval_${METHOD}_%j.err

echo "============================================================"
echo "Hypersim Evaluation: $METHOD"
echo "============================================================"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start time: \$(date)"
echo "============================================================"
echo ""

# Activate conda environment
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

# Navigate to evaluation directory
cd $SCRIPT_DIR

# Run evaluation for this method
python evaluate_hypersim_all_baselines.py --methods $METHOD

EXIT_CODE=\$?

echo ""
echo "============================================================"
echo "End time: \$(date)"
echo "Exit code: \$EXIT_CODE"
echo "============================================================"

exit \$EXIT_CODE
EOF

    # Submit the job
    JOB_OUTPUT=$(sbatch "$JOB_SCRIPT")
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
    JOB_IDS+=("$JOB_ID")

    echo "  → Submitted as job $JOB_ID"

    # Clean up temporary script
    rm "$JOB_SCRIPT"

    # Small delay to avoid overwhelming the scheduler
    sleep 0.5
done

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Submitted ${#JOB_IDS[@]} jobs:"
for i in "${!METHODS[@]}"; do
    echo "  ${METHODS[$i]}: ${JOB_IDS[$i]}"
done
echo ""
echo "Check status:"
echo "  squeue -u \$USER"
echo ""
echo "View logs (after jobs start):"
echo "  tail -f /cluster/scratch/aoezkan/planeseg/logs/eval_hypersim/eval_*_*.out"
echo ""
echo "Cancel all jobs:"
for JOB_ID in "${JOB_IDS[@]}"; do
    echo "  scancel $JOB_ID"
done
echo "============================================================"

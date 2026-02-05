#!/bin/bash
# Submit SLURM jobs for ZeroPlane mixed variants only

SCRIPT_DIR="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/scannetpp/eval_logs"

# Conda environment name (change if yours is different)
CONDA_ENV="planamono"

# Create logs directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# ZeroPlane variants only
METHODS=(
    "zeroplane_mixed"
    "zeroplane_mixed_dust3r"
)

echo "Submitting SLURM jobs for ZeroPlane variants (nonp version)..."
echo "Conda environment: ${CONDA_ENV}"
echo "Log directory: ${LOG_DIR}"
echo ""

# Submit a job for each method
for method in "${METHODS[@]}"; do
    job_name="eval_${method}_nonp"
    log_file="${LOG_DIR}/${job_name}_%j.out"
    err_file="${LOG_DIR}/${job_name}_%j.err"

    echo "Submitting: ${method}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_file}
#SBATCH --error=${err_file}
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=0

set -e  # Exit on error

echo "=========================================="
echo "Job: ${job_name}"
echo "Method: ${method}"
echo "Started at: \$(date)"
echo "Node: \$(hostname)"
echo "=========================================="
echo ""

# Change to script directory
cd ${SCRIPT_DIR}
echo "Working directory: \$(pwd)"
echo ""

# Activate conda environment
# Method 1: Using conda.sh (most reliable for SLURM)
if [ -f "\${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "\${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "\${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    # Method 2: Try conda shell hook
    eval "\$(conda shell.bash hook)" 2>/dev/null || true
fi

conda activate ${CONDA_ENV}
echo "Conda environment: \${CONDA_DEFAULT_ENV}"
echo "Python: \$(which python)"
echo ""

# Run evaluation for this method
echo "Running: python evaluate_all_baselines_nonp.py --methods ${method}"
python evaluate_all_baselines_nonp.py --methods ${method}

exit_code=\$?

echo ""
echo "=========================================="
echo "Job completed at: \$(date)"
echo "Exit code: \${exit_code}"
echo "=========================================="

exit \${exit_code}
EOF

    sleep 0.5  # Small delay between submissions
done

echo ""
echo "All jobs submitted!"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  tail -f ${LOG_DIR}/eval_zeroplane_*_nonp_*.out"
echo ""
echo "View errors:"
echo "  tail -f ${LOG_DIR}/eval_zeroplane_*_nonp_*.err"

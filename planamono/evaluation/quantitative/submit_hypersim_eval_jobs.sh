#!/bin/bash
# Submit separate SLURM jobs for each Hypersim evaluation method
#
# Usage:
#   ./submit_hypersim_eval_jobs.sh                    # Submit all hardcoded methods
#   ./submit_hypersim_eval_jobs.sh gt ours_mixed      # Submit specific methods
#   ./submit_hypersim_eval_jobs.sh --discover [model_dir ...]  # Auto-discover ZeroPlane experiments

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"

# Parse --discover flag
DISCOVER_MODE=false
DISCOVER_ARGS=()
METHODS=()

if [ "$1" = "--discover" ]; then
    DISCOVER_MODE=true
    shift
    # Remaining args are model dirs to scan (or empty for all)
    DISCOVER_ARGS=("$@")
else
    # Legacy mode: methods as positional args
    if [ $# -eq 0 ]; then
        METHODS=("gt" "moge_ours" "moge_mixed_bce" "zeroplane_mixed_dust3r" "zeroplane_mixed")
    else
        METHODS=("$@")
    fi
fi

# Create logs directory
mkdir -p /cluster/scratch/aoezkan/planeseg/logs/eval_hypersim

echo "============================================================"
echo "Submitting Hypersim Evaluation Jobs"
echo "============================================================"

if [ "$DISCOVER_MODE" = true ]; then
    # Discover methods by running the script in dry-run mode
    echo "Discovery mode: scanning H5_ROOT for ZeroPlane experiments..."
    if [ ${#DISCOVER_ARGS[@]} -gt 0 ]; then
        DISCOVER_FLAG="--discover-zeroplane ${DISCOVER_ARGS[*]}"
    else
        DISCOVER_FLAG="--discover-zeroplane"
    fi

    # Run discovery by importing the function directly
    if [ ${#DISCOVER_ARGS[@]} -gt 0 ]; then
        MODEL_DIRS_PY=$(printf '"%s",' "${DISCOVER_ARGS[@]}")
        MODEL_DIRS_PY="[${MODEL_DIRS_PY%,}]"
    else
        MODEL_DIRS_PY="None"
    fi

    DISCOVERED=$(cd "$SCRIPT_DIR" && python -c "
from evaluate_hypersim_all_baselines import discover_zeroplane_methods, H5_ROOT
methods = discover_zeroplane_methods(H5_ROOT, ${MODEL_DIRS_PY})
for k in sorted(methods):
    print(k)
" 2>/dev/null)

    if [ -z "$DISCOVERED" ]; then
        echo "[ERROR] No ZeroPlane experiments discovered. Check H5_ROOT."
        exit 1
    fi

    # Convert to array
    mapfile -t METHODS <<< "$DISCOVERED"
    echo "Discovered ${#METHODS[@]} methods:"
    for M in "${METHODS[@]}"; do
        echo "  - $M"
    done
fi

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
#SBATCH --time=4:00:00
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
EOF

    if [ "$DISCOVER_MODE" = true ]; then
        # Use --discover-zeroplane so the method is registered, then --methods to pick it
        if [ ${#DISCOVER_ARGS[@]} -gt 0 ]; then
            echo "python evaluate_hypersim_all_baselines.py --discover-zeroplane ${DISCOVER_ARGS[*]} --methods $METHOD" >> "$JOB_SCRIPT"
        else
            echo "python evaluate_hypersim_all_baselines.py --discover-zeroplane --methods $METHOD" >> "$JOB_SCRIPT"
        fi
    else
        echo "python evaluate_hypersim_all_baselines.py --methods $METHOD" >> "$JOB_SCRIPT"
    fi

    cat >> "$JOB_SCRIPT" <<EOF

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
echo "After all jobs finish, aggregate results:"
if [ "$DISCOVER_MODE" = true ]; then
    if [ ${#DISCOVER_ARGS[@]} -gt 0 ]; then
        echo "  python evaluate_hypersim_all_baselines.py --discover-zeroplane ${DISCOVER_ARGS[*]} --aggregate-only"
    else
        echo "  python evaluate_hypersim_all_baselines.py --discover-zeroplane --aggregate-only"
    fi
else
    echo "  python evaluate_hypersim_all_baselines.py --aggregate-only"
fi
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

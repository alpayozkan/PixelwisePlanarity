#!/bin/bash
# Submit separate SLURM jobs for each ZeroPlane val-split evaluation,
# then a dependent aggregation job that runs after all finish.
#
# Usage:
#   ./submit_scannetpp_val_eval_jobs.sh                                                     # All methods
#   ./submit_scannetpp_val_eval_jobs.sh default_dust3r_released/model_0000000                # One method
#   ./submit_scannetpp_val_eval_jobs.sh mixed_dust3r_noobj05_reproduce/model_0074999 mixed_dust3r_noobj05_reproduce/model_0144999

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_scannetpp_val"
INFERENCE_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference_val"

# Auto-discover or use arguments
if [ $# -eq 0 ]; then
    METHODS=()
    for method_dir in "$INFERENCE_ROOT"/*/; do
        method_name=$(basename "$method_dir")
        for ckpt_dir in "$method_dir"/*/; do
            ckpt_name=$(basename "$ckpt_dir")
            if [ -d "${ckpt_dir}/thresh_0.5" ]; then
                METHODS+=("${method_name}/${ckpt_name}")
            fi
        done
    done
else
    METHODS=("$@")
fi

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Submitting ScanNet++ Val Evaluation Jobs"
echo "============================================================"
echo "Methods (${#METHODS[@]}):"
for M in "${METHODS[@]}"; do
    echo "  - $M"
done
echo "Logs: ${LOG_DIR}"
echo "============================================================"
echo ""

# Submit a job for each method
JOB_IDS=()
for METHOD in "${METHODS[@]}"; do
    # Create a safe name for SLURM (replace / with __)
    SAFE_NAME="${METHOD//\//__}"
    echo -n "[SUBMIT] eval_val_${SAFE_NAME} ... "

    JOB_SCRIPT="/tmp/eval_val_${SAFE_NAME}_${RANDOM}.sh"

    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=eval_val_${SAFE_NAME}
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=${LOG_DIR}/eval_val_${SAFE_NAME}_%j.log
#SBATCH --error=${LOG_DIR}/eval_val_${SAFE_NAME}_%j.log

echo "============================================================"
echo "ScanNet++ Val Evaluation: $METHOD"
echo "Job ID: \$SLURM_JOB_ID | Node: \$SLURM_NODELIST"
echo "Start: \$(date)"
echo "============================================================"

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

cd ${SCRIPT_DIR}

python evaluate_scannetpp_val.py --methods $METHOD

echo "Exit code: \$? | End: \$(date)"
EOF

    JOB_OUTPUT=$(sbatch "$JOB_SCRIPT")
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
    JOB_IDS+=("$JOB_ID")
    echo "job $JOB_ID"

    rm "$JOB_SCRIPT"
    sleep 0.5
done

# Submit aggregation job dependent on all eval jobs
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")

echo ""
echo -n "[SUBMIT] eval_val_aggregate (after all) ... "

AGG_SCRIPT="/tmp/eval_val_agg_${RANDOM}.sh"
METHODS_STR="${METHODS[@]}"

cat > "$AGG_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=eval_val_aggregate
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --dependency=afterany:${DEP_STR}
#SBATCH --output=${LOG_DIR}/eval_val_aggregate_%j.log
#SBATCH --error=${LOG_DIR}/eval_val_aggregate_%j.log

echo "============================================================"
echo "Val Aggregation"
echo "Methods: ${METHODS_STR}"
echo "Job ID: \$SLURM_JOB_ID | Start: \$(date)"
echo "============================================================"

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

cd ${SCRIPT_DIR}

python evaluate_scannetpp_val.py --methods ${METHODS_STR} --aggregate-only

echo "Exit code: \$? | End: \$(date)"
EOF

AGG_OUTPUT=$(sbatch "$AGG_SCRIPT")
AGG_ID=$(echo "$AGG_OUTPUT" | awk '{print $NF}')
echo "job $AGG_ID (depends on ${DEP_STR})"

rm "$AGG_SCRIPT"

echo ""
echo "============================================================"
echo "Summary: ${#JOB_IDS[@]} eval jobs + 1 aggregation job"
echo "============================================================"
for i in "${!METHODS[@]}"; do
    echo "  ${METHODS[$i]}: ${JOB_IDS[$i]}"
done
echo "  aggregate: ${AGG_ID} (runs after all eval jobs)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ${LOG_DIR}/eval_val_*.log"
echo "============================================================"

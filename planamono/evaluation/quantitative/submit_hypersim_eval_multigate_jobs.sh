#!/bin/bash
# Submit separate SLURM jobs for each Hypersim multigate evaluation method,
# then a dependent aggregation job that runs after all finish.
#
# Usage:
#   ./submit_hypersim_eval_multigate_jobs.sh                            # All methods
#   ./submit_hypersim_eval_multigate_jobs.sh gt moge_mixed_bce          # Specific methods
#   GATES="0.5 0.7 0.8 0.9" ./submit_hypersim_eval_multigate_jobs.sh   # Custom gates

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_hypersim_multigate"
GATES="${GATES:-0.5 0.7 0.8 0.9}"

# Methods to evaluate (pass as arguments or use default)
if [ $# -eq 0 ]; then
    METHODS=("gt" "moge_ours" "moge_mixed_bce" "zeroplane_mixed_dust3r" "zeroplane_mixed")
else
    METHODS=("$@")
fi

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Submitting Hypersim Multigate Evaluation Jobs"
echo "============================================================"
echo "Methods: ${METHODS[@]}"
echo "Gates:   ${GATES}"
echo "Logs:    ${LOG_DIR}"
echo "============================================================"
echo ""

# Submit a job for each method
JOB_IDS=()
for METHOD in "${METHODS[@]}"; do
    echo -n "[SUBMIT] eval_mg_${METHOD} ... "

    JOB_SCRIPT="/tmp/eval_hypersim_mg_${METHOD}_${RANDOM}.sh"

    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=eval_mg_${METHOD}
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=${LOG_DIR}/eval_mg_${METHOD}_%j.log
#SBATCH --error=${LOG_DIR}/eval_mg_${METHOD}_%j.log

echo "============================================================"
echo "Hypersim Multigate Evaluation: $METHOD"
echo "Gates: ${GATES}"
echo "Job ID: \$SLURM_JOB_ID | Node: \$SLURM_NODELIST"
echo "Start: \$(date)"
echo "============================================================"

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

cd ${SCRIPT_DIR}

python evaluate_hypersim_all_baselines.py --methods $METHOD --inlier-gates ${GATES}

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
echo -n "[SUBMIT] eval_mg_aggregate (after all) ... "

AGG_SCRIPT="/tmp/eval_hypersim_mg_agg_${RANDOM}.sh"
METHODS_STR="${METHODS[@]}"

cat > "$AGG_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=eval_mg_aggregate
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --dependency=afterany:${DEP_STR}
#SBATCH --output=${LOG_DIR}/eval_mg_aggregate_%j.log
#SBATCH --error=${LOG_DIR}/eval_mg_aggregate_%j.log

echo "============================================================"
echo "Multigate Aggregation"
echo "Methods: ${METHODS_STR}"
echo "Gates: ${GATES}"
echo "Job ID: \$SLURM_JOB_ID | Start: \$(date)"
echo "============================================================"

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

cd ${SCRIPT_DIR}

python evaluate_hypersim_all_baselines.py --methods ${METHODS_STR} --inlier-gates ${GATES} --aggregate-only

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
echo "Logs:    tail -f ${LOG_DIR}/eval_mg_*.log"
echo "============================================================"

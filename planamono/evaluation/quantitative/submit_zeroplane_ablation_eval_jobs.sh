#!/bin/bash
# Submit separate SLURM jobs for each ZeroPlane ablation variant evaluation.
#
# Each job evaluates one variant (experiment/checkpoint/threshold combination).
# After all jobs finish, run: python evaluate_zeroplane_ablations.py --aggregate-only
#
# Usage:
#   ./submit_zeroplane_ablation_eval_jobs.sh                                          # All discovered variants
#   ./submit_zeroplane_ablation_eval_jobs.sh mixed_dust3r                             # All variants from experiment
#   ./submit_zeroplane_ablation_eval_jobs.sh mixed_dust3r/model_0074999               # All thresholds for one model
#   ./submit_zeroplane_ablation_eval_jobs.sh mixed_dust3r/model_0074999 default_dust3r_released/model_0000000  # Multiple

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_zp_ablations"
CONDA_ENV="planamono"
H5_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference"

mkdir -p "${LOG_DIR}"

echo "Discovering ablation variants in ${H5_ROOT}..."

VARIANTS=""

if [ $# -eq 0 ]; then
    # No args: discover all
    for model_dir in "${H5_ROOT}"/*/model_*; do
        [ -d "$model_dir" ] || continue
        exp_dir=$(basename "$(dirname "$model_dir")")
        for thresh_dir in "$model_dir"/thresh_*; do
            [ -d "$thresh_dir" ] || continue
            if ls "$thresh_dir"/*/planes.h5 &>/dev/null; then
                variant="${exp_dir}/$(basename "$model_dir")/$(basename "$thresh_dir")"
                VARIANTS="${VARIANTS:+${VARIANTS}
}${variant}"
            fi
        done
    done
else
    # Args given: each arg is either "exp_name" or "exp_name/model_name"
    for arg in "$@"; do
        n_slashes=$(echo "$arg" | tr -cd '/' | wc -c)

        if [ "$n_slashes" -eq 0 ]; then
            # Experiment name only -> all models + all thresholds
            for model_dir in "${H5_ROOT}/${arg}"/model_*; do
                [ -d "$model_dir" ] || continue
                for thresh_dir in "$model_dir"/thresh_*; do
                    [ -d "$thresh_dir" ] || continue
                    if ls "$thresh_dir"/*/planes.h5 &>/dev/null; then
                        variant="${arg}/$(basename "$model_dir")/$(basename "$thresh_dir")"
                        VARIANTS="${VARIANTS:+${VARIANTS}
}${variant}"
                    fi
                done
            done
        elif [ "$n_slashes" -eq 1 ]; then
            # exp_name/model_name -> all thresholds for that model
            for thresh_dir in "${H5_ROOT}/${arg}"/thresh_*; do
                [ -d "$thresh_dir" ] || continue
                if ls "$thresh_dir"/*/planes.h5 &>/dev/null; then
                    variant="${arg}/$(basename "$thresh_dir")"
                    VARIANTS="${VARIANTS:+${VARIANTS}
}${variant}"
                fi
            done
        else
            # Full path exp/model/thresh
            if [ -d "${H5_ROOT}/${arg}" ] && ls "${H5_ROOT}/${arg}"/*/planes.h5 &>/dev/null; then
                VARIANTS="${VARIANTS:+${VARIANTS}
}${arg}"
            else
                echo "[WARN] Not found: ${arg}"
            fi
        fi
    done
fi

if [ -z "$VARIANTS" ]; then
    echo "[ERROR] No variants discovered!"
    exit 1
fi

VARIANT_COUNT=$(echo "$VARIANTS" | wc -l)

echo "============================================================"
echo "Submitting ZeroPlane Ablation Evaluation Jobs"
echo "============================================================"
echo "Variants to evaluate: ${VARIANT_COUNT}"
echo "Log directory: ${LOG_DIR}"
echo "============================================================"
echo ""

JOB_IDS=()
VARIANT_NAMES=()

while IFS= read -r VARIANT; do
    # Create a safe job name from the variant path
    # e.g. mixed_dust3r/model_0024999/thresh_default -> mixed_dust3r_m0024999_tdefault
    EXP=$(echo "$VARIANT" | cut -d/ -f1)
    MODEL=$(echo "$VARIANT" | cut -d/ -f2 | sed 's/model_/m/')
    THRESH=$(echo "$VARIANT" | cut -d/ -f3 | sed 's/thresh_/t/')
    JOB_NAME="zpa_${EXP}_${MODEL}_${THRESH}"

    echo "Submitting: ${VARIANT}"

    JOB_OUTPUT=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

set -e

echo "=========================================="
echo "ZeroPlane Ablation Evaluation"
echo "Variant: ${VARIANT}"
echo "Started at: \$(date)"
echo "Node: \$(hostname)"
echo "=========================================="
echo ""

cd ${SCRIPT_DIR}

# Activate conda
if [ -f "/cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh" ]; then
    source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
elif [ -f "\${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "\${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "\${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    eval "\$(conda shell.bash hook)" 2>/dev/null || true
fi

conda activate ${CONDA_ENV}
echo "Python: \$(which python)"
echo ""

python evaluate_zeroplane_ablations.py --variant "${VARIANT}"

exit_code=\$?

echo ""
echo "=========================================="
echo "Completed at: \$(date)"
echo "Exit code: \${exit_code}"
echo "=========================================="

exit \${exit_code}
EOF
    )

    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
    JOB_IDS+=("$JOB_ID")
    VARIANT_NAMES+=("$VARIANT")

    echo "  -> Job ${JOB_ID}"

    sleep 0.3
done <<< "$VARIANTS"

echo ""
echo "============================================================"
echo "Summary: Submitted ${#JOB_IDS[@]} jobs"
echo "============================================================"
for i in "${!VARIANT_NAMES[@]}"; do
    echo "  ${JOB_IDS[$i]}  ${VARIANT_NAMES[$i]}"
done
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo ""
echo "Logs:"
echo "  tail -f ${LOG_DIR}/zpa_*.out"
echo ""
echo "After all jobs finish, aggregate:"
echo "  python evaluate_zeroplane_ablations.py --aggregate-only"
echo "============================================================"

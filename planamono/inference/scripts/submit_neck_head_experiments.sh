#!/bin/bash
# Submit inference + evaluation for neck_head and proj_neck_head architectures
# on both ScanNet++ and Hypersim datasets.
#
# Pipeline: 8 inference jobs (2 arch x 2 epochs x 2 datasets) -> 8 evaluation jobs (dependent)
#
# Usage:
#   ./submit_neck_head_experiments.sh          # Submit all 16 jobs
#   ./submit_neck_head_experiments.sh --dry-run  # Print commands without submitting

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
EVAL_DIR="$PROJECT_ROOT/evaluation/quantitative"
LOG_ROOT="/cluster/scratch/aoezkan/planeseg/logs"

# Conda
CONDA_SH="/cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="planamono"

# Checkpoint roots
NECK_HEAD_DIR="/cluster/scratch/ayavuz/moge_neck_head_output"
PROJ_NECK_HEAD_DIR="/cluster/scratch/ayavuz/moge_proj_neck_head_output"

# Common inference parameters
SPLIT="test"
BATCH_SIZE=8
NUM_TOKENS=1024
THRESHOLD_PLANARITY=0.6
NORMAL_THRESHOLD_DEG=10.0
DEPTH_THRESHOLD=0.05
NEIGHBOR_MATCH_COUNT_THRESH=24

# ScanNet++ paths
SPP_OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference"
SPP_DATASET_DIR="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
SPP_RGB_ROOT="/cluster/project/cvg/Shared_datasets/scannet++/data"

# Hypersim paths
HYP_OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/hypersim/inference"
HYP_ROOT="/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
HYP_PLANE_ROOT="/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
HYP_PARAMS_ROOT="/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"

# Check for --dry-run
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Commands will be printed but not submitted."
    echo ""
fi

# Create log directories
mkdir -p "$LOG_ROOT/inference" "$LOG_ROOT/eval_scannetpp" "$LOG_ROOT/eval_hypersim"

# ============================================================
# Define experiments: (name, architecture, checkpoint)
# ============================================================
declare -a EXP_NAMES=(
    "moge_neck_head_ep1"
    "moge_neck_head_ep2"
    "moge_proj_neck_head_ep1"
    "moge_proj_neck_head_ep2"
)
declare -a EXP_ARCHS=(
    "neck_head"
    "neck_head"
    "proj_neck_head"
    "proj_neck_head"
)
declare -a EXP_CKPTS=(
    "$NECK_HEAD_DIR/model_epoch1.pt"
    "$NECK_HEAD_DIR/model_epoch2.pt"
    "$PROJ_NECK_HEAD_DIR/model_epoch1.pt"
    "$PROJ_NECK_HEAD_DIR/model_epoch2.pt"
)

echo "============================================================"
echo "Submitting neck_head / proj_neck_head experiments"
echo "============================================================"
echo ""

INFER_JOB_IDS=()
EVAL_JOB_IDS=()

for i in "${!EXP_NAMES[@]}"; do
    EXP="${EXP_NAMES[$i]}"
    ARCH="${EXP_ARCHS[$i]}"
    CKPT="${EXP_CKPTS[$i]}"

    echo ">>> Experiment: $EXP (arch=$ARCH)"
    echo "    Checkpoint:  $CKPT"

    # ----------------------------------------------------------
    # 1) ScanNet++ inference
    # ----------------------------------------------------------
    SPP_H5="${SPP_OUTPUT_ROOT}/${EXP}_h5"
    SPP_INFER_SCRIPT="/tmp/infer_spp_${EXP}_${RANDOM}.sh"

    cat > "$SPP_INFER_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=inf_spp_${EXP}
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=${LOG_ROOT}/inference/${EXP}_scannetpp_%j.out
#SBATCH --error=${LOG_ROOT}/inference/${EXP}_scannetpp_%j.err

set -euo pipefail
source ${CONDA_SH}
conda activate ${CONDA_ENV}

echo "ScanNet++ inference: ${EXP} (arch=${ARCH})"
mkdir -p "${SPP_H5}"

python ${PROJECT_ROOT}/inference/planarity/inference_to_h5.py \
    --model_path "${CKPT}" \
    --output_root "${SPP_H5}" \
    --dataset_dir "${SPP_DATASET_DIR}" \
    --rgb_root "${SPP_RGB_ROOT}" \
    --split ${SPLIT} \
    --batch_size ${BATCH_SIZE} \
    --num_tokens ${NUM_TOKENS} \
    --threshold_planarity ${THRESHOLD_PLANARITY} \
    --normal_threshold_deg ${NORMAL_THRESHOLD_DEG} \
    --depth_threshold ${DEPTH_THRESHOLD} \
    --neighbor_match_count_thresh ${NEIGHBOR_MATCH_COUNT_THRESH} \
    --architecture ${ARCH}

echo "[DONE] ScanNet++ inference: ${EXP}"
EOF

    if [ "$DRY_RUN" = true ]; then
        echo "    [ScanNet++ infer] sbatch $SPP_INFER_SCRIPT"
        SPP_INFER_JID="DRY_RUN"
    else
        SPP_INFER_JID=$(sbatch "$SPP_INFER_SCRIPT" | awk '{print $NF}')
        echo "    [ScanNet++ infer] job=$SPP_INFER_JID -> $SPP_H5"
    fi
    rm "$SPP_INFER_SCRIPT"
    INFER_JOB_IDS+=("$SPP_INFER_JID")

    # ----------------------------------------------------------
    # 2) Hypersim inference
    # ----------------------------------------------------------
    HYP_H5="${HYP_OUTPUT_ROOT}/${EXP}_h5"
    HYP_INFER_SCRIPT="/tmp/infer_hyp_${EXP}_${RANDOM}.sh"

    cat > "$HYP_INFER_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=inf_hyp_${EXP}
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=${LOG_ROOT}/inference/${EXP}_hypersim_%j.out
#SBATCH --error=${LOG_ROOT}/inference/${EXP}_hypersim_%j.err

set -euo pipefail
source ${CONDA_SH}
conda activate ${CONDA_ENV}

echo "Hypersim inference: ${EXP} (arch=${ARCH})"
mkdir -p "${HYP_H5}"

python ${PROJECT_ROOT}/inference/planarity/inference_to_h5_hypersim.py \
    --model_path "${CKPT}" \
    --output_root "${HYP_H5}" \
    --hypersim_root "${HYP_ROOT}" \
    --plane_label_root "${HYP_PLANE_ROOT}" \
    --params_root "${HYP_PARAMS_ROOT}" \
    --split ${SPLIT} \
    --batch_size ${BATCH_SIZE} \
    --num_tokens ${NUM_TOKENS} \
    --threshold_planarity ${THRESHOLD_PLANARITY} \
    --normal_threshold_deg ${NORMAL_THRESHOLD_DEG} \
    --depth_threshold ${DEPTH_THRESHOLD} \
    --neighbor_match_count_thresh ${NEIGHBOR_MATCH_COUNT_THRESH} \
    --architecture ${ARCH}

echo "[DONE] Hypersim inference: ${EXP}"
EOF

    if [ "$DRY_RUN" = true ]; then
        echo "    [Hypersim infer]  sbatch $HYP_INFER_SCRIPT"
        HYP_INFER_JID="DRY_RUN"
    else
        HYP_INFER_JID=$(sbatch "$HYP_INFER_SCRIPT" | awk '{print $NF}')
        echo "    [Hypersim infer]  job=$HYP_INFER_JID -> $HYP_H5"
    fi
    rm "$HYP_INFER_SCRIPT"
    INFER_JOB_IDS+=("$HYP_INFER_JID")

    # ----------------------------------------------------------
    # 3) ScanNet++ evaluation (depends on ScanNet++ inference)
    # ----------------------------------------------------------
    SPP_EVAL_SCRIPT="/tmp/eval_spp_${EXP}_${RANDOM}.sh"

    cat > "$SPP_EVAL_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=eval_spp_${EXP}
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=${LOG_ROOT}/eval_scannetpp/eval_${EXP}_%j.out
#SBATCH --error=${LOG_ROOT}/eval_scannetpp/eval_${EXP}_%j.err

set -euo pipefail
source ${CONDA_SH}
conda activate ${CONDA_ENV}

echo "ScanNet++ evaluation: ${EXP}"
cd ${EVAL_DIR}
python evaluate_all_baselines.py --methods ${EXP}

echo "[DONE] ScanNet++ evaluation: ${EXP}"
EOF

    if [ "$DRY_RUN" = true ]; then
        echo "    [ScanNet++ eval]  sbatch --dependency=afterok:$SPP_INFER_JID $SPP_EVAL_SCRIPT"
        SPP_EVAL_JID="DRY_RUN"
    else
        SPP_EVAL_JID=$(sbatch --dependency=afterok:${SPP_INFER_JID} "$SPP_EVAL_SCRIPT" | awk '{print $NF}')
        echo "    [ScanNet++ eval]  job=$SPP_EVAL_JID (depends on $SPP_INFER_JID)"
    fi
    rm "$SPP_EVAL_SCRIPT"
    EVAL_JOB_IDS+=("$SPP_EVAL_JID")

    # ----------------------------------------------------------
    # 4) Hypersim evaluation (depends on Hypersim inference)
    # ----------------------------------------------------------
    HYP_EVAL_SCRIPT="/tmp/eval_hyp_${EXP}_${RANDOM}.sh"

    cat > "$HYP_EVAL_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=eval_hyp_${EXP}
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=${LOG_ROOT}/eval_hypersim/eval_${EXP}_%j.out
#SBATCH --error=${LOG_ROOT}/eval_hypersim/eval_${EXP}_%j.err

set -euo pipefail
source ${CONDA_SH}
conda activate ${CONDA_ENV}

echo "Hypersim evaluation: ${EXP}"
cd ${EVAL_DIR}
python evaluate_hypersim_all_baselines.py --methods ${EXP}

echo "[DONE] Hypersim evaluation: ${EXP}"
EOF

    if [ "$DRY_RUN" = true ]; then
        echo "    [Hypersim eval]   sbatch --dependency=afterok:$HYP_INFER_JID $HYP_EVAL_SCRIPT"
        HYP_EVAL_JID="DRY_RUN"
    else
        HYP_EVAL_JID=$(sbatch --dependency=afterok:${HYP_INFER_JID} "$HYP_EVAL_SCRIPT" | awk '{print $NF}')
        echo "    [Hypersim eval]   job=$HYP_EVAL_JID (depends on $HYP_INFER_JID)"
    fi
    rm "$HYP_EVAL_SCRIPT"
    EVAL_JOB_IDS+=("$HYP_EVAL_JID")

    echo ""
    sleep 0.5
done

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "Summary"
echo "============================================================"
echo ""
echo "Inference jobs: ${INFER_JOB_IDS[*]}"
echo "Evaluation jobs (dependent): ${EVAL_JOB_IDS[*]}"
echo ""
echo "Check status:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f ${LOG_ROOT}/inference/*neck_head*.out"
echo "  tail -f ${LOG_ROOT}/eval_scannetpp/eval_moge_*neck_head*.out"
echo "  tail -f ${LOG_ROOT}/eval_hypersim/eval_moge_*neck_head*.out"
echo ""
echo "After all jobs finish, aggregate results:"
echo "  cd ${EVAL_DIR}"
echo "  python evaluate_all_baselines.py --aggregate-only"
echo "  python evaluate_hypersim_all_baselines.py --aggregate-only"
echo ""
echo "Cancel all:"
for JID in "${INFER_JOB_IDS[@]}" "${EVAL_JOB_IDS[@]}"; do
    echo "  scancel $JID"
done
echo "============================================================"

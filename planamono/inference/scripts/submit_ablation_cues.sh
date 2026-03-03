#!/bin/bash
# Submit segmentation cue ablation pipeline on ScanNet++ (full test split).
#
# Pipeline per ablation:
#   Job 1: inference_to_h5.py (GPU: MoGe inference + segmentation → planes.h5)
#   Job 2: evaluate_all_baselines.py (CPU, depends on Job 1)
# Final:
#   Job 3: aggregate all 4 ablations (depends on all eval jobs)
#
# Total: 4 infer + 4 eval + 1 agg = 9 jobs
#
# Usage:
#   bash submit_ablation_cues.sh

set -euo pipefail

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/ablation_cues"
mkdir -p "$LOG_DIR"

MODEL_PATH="/cluster/scratch/ayavuz/moge_mixed_output_476644_fixed_cosLR_singlePhase_mixed_HiRes/model_epoch3.pt"
H5_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference"
RGB_ROOT="/cluster/project/cvg/Shared_datasets/scannet++/data"
DATASET_DIR="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"

# Conda
CONDA_INIT="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planeseg"

# v5 segmentation defaults (matching notebook)
SEG_VERSION="v5"
NORMAL_DEG=10.0
DEPTH_THRESH=0.05       # absolute meters (v5)
PLANARITY_THRESH=0.5
NMC_THRESH=8            # v5 default

SPLIT="test"
BATCH_SIZE=16
NUM_TOKENS=1600
TOTAL_SCENES=42         # ScanNet++ test split

# ============================================================
# ABLATION CONFIGS
# ============================================================
declare -A ABLATION_PLANARITY
declare -A ABLATION_NORMAL
declare -A ABLATION_DEPTH

ABLATION_NAMES=("only_normal" "only_depth" "normal_depth" "full")

ABLATION_PLANARITY[only_normal]=-1.0      # disabled
ABLATION_NORMAL[only_normal]=$NORMAL_DEG  # enabled
ABLATION_DEPTH[only_normal]=1e10          # disabled

ABLATION_PLANARITY[only_depth]=-1.0       # disabled
ABLATION_NORMAL[only_depth]=180.0         # disabled
ABLATION_DEPTH[only_depth]=$DEPTH_THRESH  # enabled

ABLATION_PLANARITY[normal_depth]=-1.0     # disabled
ABLATION_NORMAL[normal_depth]=$NORMAL_DEG # enabled
ABLATION_DEPTH[normal_depth]=$DEPTH_THRESH # enabled

ABLATION_PLANARITY[full]=$PLANARITY_THRESH # enabled
ABLATION_NORMAL[full]=$NORMAL_DEG          # enabled
ABLATION_DEPTH[full]=$DEPTH_THRESH         # enabled

echo "=========================================="
echo "  Segmentation Cue Ablation Pipeline"
echo "=========================================="
echo "Model:       $MODEL_PATH"
echo "Seg version: $SEG_VERSION"
echo "Split:       $SPLIT ($TOTAL_SCENES scenes)"
echo "Ablations:   ${ABLATION_NAMES[*]}"
echo ""

# ============================================================
# PER-ABLATION: INFER → EVAL
# ============================================================
ALL_METHODS=""
ALL_EVAL_JOBIDS=""

for ABL in "${ABLATION_NAMES[@]}"; do
    METHOD="ablation_${ABL}"
    OUTPUT="$H5_ROOT/${METHOD}_h5"
    P_THRESH="${ABLATION_PLANARITY[$ABL]}"
    N_DEG="${ABLATION_NORMAL[$ABL]}"
    D_THRESH="${ABLATION_DEPTH[$ABL]}"

    ALL_METHODS="$ALL_METHODS $METHOD"

    echo "--- $METHOD ---"
    echo "  planarity=$P_THRESH  normal_deg=$N_DEG  depth=$D_THRESH"
    echo "  output: $OUTPUT"

    # --- Inference job (GPU: MoGe + segmentation → H5) ---
    INFER_JOB=$(sbatch --parsable \
        --job-name="abl_inf_${ABL}" \
        --time=12:00:00 \
        --cpus-per-task=4 \
        --mem-per-cpu=8G \
        --gpus=rtx_3090:1 \
        --output="$LOG_DIR/infer_${ABL}_%j.out" \
        --error="$LOG_DIR/infer_${ABL}_%j.err" \
        --wrap="
$CONDA_INIT
mkdir -p $OUTPUT
python $PROJECT_ROOT/inference/planarity/inference_to_h5.py \
    --model_path $MODEL_PATH \
    --output_root $OUTPUT \
    --dataset_dir $DATASET_DIR \
    --rgb_root $RGB_ROOT \
    --split $SPLIT \
    --batch_size $BATCH_SIZE \
    --num_tokens $NUM_TOKENS \
    --seg_version $SEG_VERSION \
    --threshold_planarity $P_THRESH \
    --normal_threshold_deg $N_DEG \
    --depth_threshold $D_THRESH \
    --neighbor_match_count_thresh $NMC_THRESH
")
    echo "  [INFER] $INFER_JOB"

    # --- Eval job (CPU, depends on infer) ---
    EVAL_JOB=$(sbatch --parsable \
        --dependency=afterok:$INFER_JOB \
        --job-name="abl_eval_${ABL}" \
        --time=8:00:00 \
        --cpus-per-task=16 \
        --mem-per-cpu=4G \
        --output="$LOG_DIR/eval_${ABL}_%j.out" \
        --error="$LOG_DIR/eval_${ABL}_%j.err" \
        --wrap="
$CONDA_INIT
python $PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py \
    --methods $METHOD
")
    echo "  [EVAL]  $EVAL_JOB (depends on $INFER_JOB)"

    ALL_EVAL_JOBIDS="$ALL_EVAL_JOBIDS:$EVAL_JOB"
    echo ""
done

# ============================================================
# SINGLE AGGREGATION JOB
# ============================================================
ALL_EVAL_JOBIDS="${ALL_EVAL_JOBIDS#:}"

AGG_JOB=$(sbatch --parsable \
    --dependency=afterok:$ALL_EVAL_JOBIDS \
    --job-name="abl_agg" \
    --time=0:10:00 \
    --cpus-per-task=2 \
    --mem-per-cpu=4G \
    --output="$LOG_DIR/agg_%j.out" \
    --error="$LOG_DIR/agg_%j.err" \
    --wrap="
$CONDA_INIT
python $PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py \
    --methods $ALL_METHODS \
    --aggregate-only
")
echo "[AGG] $AGG_JOB (depends on all eval jobs)"

echo ""
echo "=========================================="
echo "  9 jobs submitted"
echo "=========================================="
echo "  4 infer (GPU) → 4 eval (CPU) → 1 agg"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    $LOG_DIR/"

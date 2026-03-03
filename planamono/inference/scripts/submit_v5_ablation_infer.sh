#!/bin/bash
# Submit all v5 segmentation ablation variants using one-shot inference (inference_to_h5.py).
#
# 8 variants = 2 parameter sets × 4 seg variants:
#   NEW  params (plan=0.3, norm=5.0°, match=8):  v5, v5_relative, v5_no_sobel, v5_dotprod_relative
#   ORIG params (plan=0.6, norm=10.0°, match=24): same 4 seg variants
#
# Each variant runs as an independent job chain: infer (GPU) → eval (CPU) → agg (CPU).
#
# Usage:
#   bash submit_v5_ablation_infer.sh                                      # all 8 variants
#   bash submit_v5_ablation_infer.sh moge_hires_ep3_v5seg                 # specific method key(s)
#   bash submit_v5_ablation_infer.sh moge_hires_ep3_v5_relative_seg moge_hires_ep3_v5origparams_relative_seg

set -euo pipefail

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/v5_ablation_infer"
mkdir -p "$LOG_DIR"

MODEL_PATH="/cluster/scratch/ayavuz/moge_mixed_output_476644_fixed_cosLR_singlePhase_mixed_HiRes/model_epoch3.pt"
DATASET_DIR="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
RGB_ROOT="/cluster/project/cvg/Shared_datasets/scannet++/data"
H5_BASE="/cluster/scratch/aoezkan/planeseg/scannetpp/inference"
SPLIT="test"
BATCH_SIZE=8
NUM_TOKENS=1600

# ---- Variant definitions ----
# Format: "eval_method:seg_version:depth_threshold:plan_thresh:normal_deg:match_thresh"
declare -a ALL_VARIANTS=(
    # NEW params (plan=0.3, norm=5.0°, match=8)
    "moge_hires_ep3_v5seg:v5:0.05:0.3:5.0:8"
    "moge_hires_ep3_v5_relative_seg:v5_relative:0.025:0.3:5.0:8"
    "moge_hires_ep3_v5_no_sobel_seg:v5_no_sobel:0.05:0.3:5.0:8"
    "moge_hires_ep3_v5_dotprod_relative_seg:v5_dotprod_relative:0.025:0.3:5.0:8"
    # ORIG params (plan=0.6, norm=10.0°, match=24)
    "moge_hires_ep3_v5origparams_seg:v5:0.05:0.6:10.0:24"
    "moge_hires_ep3_v5origparams_relative_seg:v5_relative:0.025:0.6:10.0:24"
    "moge_hires_ep3_v5origparams_no_sobel_seg:v5_no_sobel:0.05:0.6:10.0:24"
    "moge_hires_ep3_v5origparams_dotprod_relative_seg:v5_dotprod_relative:0.025:0.6:10.0:24"
)

# Determine which variants to run
if [ $# -eq 0 ]; then
    SELECTED=("${ALL_VARIANTS[@]}")
else
    SELECTED=()
    for spec in "${ALL_VARIANTS[@]}"; do
        method="${spec%%:*}"
        for arg in "$@"; do
            if [ "$method" = "$arg" ]; then
                SELECTED+=("$spec")
                break
            fi
        done
    done
    if [ ${#SELECTED[@]} -eq 0 ]; then
        echo "ERROR: No matching variants for: $*"
        echo "Available: $(printf '%s\n' "${ALL_VARIANTS[@]}" | cut -d: -f1 | tr '\n' ' ')"
        exit 1
    fi
fi

echo "============================================================"
echo "v5 Segmentation Ablation — One-Shot Inference (ScanNet++)"
echo "============================================================"
echo "Model:    $MODEL_PATH"
echo "Variants: ${#SELECTED[@]}"
echo "============================================================"
echo ""

for spec in "${SELECTED[@]}"; do
    IFS=':' read -r METHOD SEG_VERSION DEPTH_THR PLAN_THR NORMAL_DEG MATCH_THRESH <<< "$spec"
    OUTPUT_ROOT="${H5_BASE}/${METHOD}_h5"

    echo "--- $METHOD ---"
    echo "  seg:   $SEG_VERSION  depth: ${DEPTH_THR}  plan: ${PLAN_THR}  norm: ${NORMAL_DEG}°  match: ${MATCH_THRESH}"

    # ── Inference (GPU) ──
    JOB_INFER=$(sbatch --parsable \
        --job-name="infer_${METHOD}" \
        --time=12:00:00 \
        --cpus-per-task=8 \
        --mem-per-cpu=16G \
        --gpus=rtx_3090:1 \
        --output="$LOG_DIR/infer_${METHOD}_%j.out" \
        --error="$LOG_DIR/infer_${METHOD}_%j.err" \
        --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
mkdir -p $OUTPUT_ROOT
python $PROJECT_ROOT/inference/planarity/inference_to_h5.py \
    --model_path $MODEL_PATH \
    --output_root $OUTPUT_ROOT \
    --dataset_dir $DATASET_DIR \
    --rgb_root $RGB_ROOT \
    --split $SPLIT \
    --batch_size $BATCH_SIZE \
    --num_tokens $NUM_TOKENS \
    --seg_version $SEG_VERSION \
    --threshold_planarity $PLAN_THR \
    --normal_threshold_deg $NORMAL_DEG \
    --depth_threshold $DEPTH_THR \
    --neighbor_match_count_thresh $MATCH_THRESH
'")
    echo "  [INFER] $JOB_INFER"

    # ── Evaluation (CPU, depends on inference) ──
    JOB_EVAL=$(sbatch --parsable \
        --job-name="eval_${METHOD}" \
        --dependency=afterok:$JOB_INFER \
        --time=12:00:00 \
        --cpus-per-task=16 \
        --mem-per-cpu=4G \
        --output="$LOG_DIR/eval_${METHOD}_%j.out" \
        --error="$LOG_DIR/eval_${METHOD}_%j.err" \
        --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
python $PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py --methods $METHOD
'")
    echo "  [EVAL]  $JOB_EVAL (after $JOB_INFER)"
    echo "  Pipeline: $JOB_INFER → $JOB_EVAL"
    echo ""
done

echo "============================================================"
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: $LOG_DIR"
echo "============================================================"

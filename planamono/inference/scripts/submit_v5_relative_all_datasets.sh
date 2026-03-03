#!/bin/bash
# Submit v5_relative and v5origparams_relative inference + eval across all 5 datasets.
# Each dataset × method runs as an independent job chain: infer (GPU) → eval (CPU) → agg (CPU).
#
# Datasets: scannetpp, hypersim, synthia, vkitti2, pd
# Methods:
#   moge_hires_ep3_v5_relative_seg       (plan=0.3, norm=5.0°, match=8,  depth_rel=0.025)
#   moge_hires_ep3_v5origparams_relative_seg (plan=0.6, norm=10.0°, match=24, depth_rel=0.025)
#
# Usage:
#   bash submit_v5_relative_all_datasets.sh

set -euo pipefail

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/v5_relative_all_datasets"
mkdir -p "$LOG_DIR"

MODEL_PATH="/cluster/scratch/ayavuz/moge_mixed_output_476644_fixed_cosLR_singlePhase_mixed_HiRes/model_epoch3.pt"
SEG_VERSION="v5_relative"
DEPTH_THRESHOLD=0.025  # relative fraction of center depth

# ---- Dataset configurations ----
# Format: "name:infer_script:eval_script:output_base:extra_args"
# extra_args: dataset-specific path arguments
declare -a DATASETS=(
    "scannetpp:inference_to_h5.py:evaluate_all_baselines.py:/cluster/scratch/aoezkan/planeseg/scannetpp/inference:--dataset_dir /cluster/scratch/aoezkan/planeseg/dataset/scannetpp --rgb_root /cluster/project/cvg/Shared_datasets/scannet++/data --split test --batch_size 8 --num_tokens 1600"
    "hypersim:inference_to_h5_hypersim.py:evaluate_hypersim_all_baselines.py:/cluster/scratch/aoezkan/planeseg/hypersim/inference:--hypersim_root /cluster/scratch/aoezkan/planeseg/dataset/hypersim --plane_label_root /cluster/scratch/aoezkan/planeseg/dataset/hypersim --params_root /cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params --split test --batch_size 8 --num_tokens 1600"
    "synthia:inference_to_h5_synthia.py:evaluate_synthia_all_baselines.py:/cluster/scratch/aoezkan/planeseg/synthia/inference:--data_root /cluster/scratch/ayavuz/dataset/synthia_planes --split test --batch_size 16 --num_tokens 1024"
    "vkitti2:inference_to_h5_vkitti2.py:evaluate_vkitti2_all_baselines.py:/cluster/scratch/aoezkan/planeseg/vkitti2/inference:--data_root /cluster/scratch/ayavuz/dataset/vkitti2_planes --split test --batch_size 16 --num_tokens 1024"
)

# ---- Method configurations ----
# Format: "method_key:plan_thresh:normal_deg:match_thresh"
declare -a METHODS=(
    "moge_hires_ep3_v5_relative_seg:0.3:5.0:8"
    "moge_hires_ep3_v5origparams_relative_seg:0.6:10.0:24"
)

echo "============================================================"
echo "v5_relative Ablation — All Datasets"
echo "============================================================"
echo "Model:    $MODEL_PATH"
echo "Datasets: ${#DATASETS[@]}  Methods: ${#METHODS[@]}"
echo "Total chains: $(( ${#DATASETS[@]} * ${#METHODS[@]} ))"
echo "============================================================"
echo ""

for method_spec in "${METHODS[@]}"; do
    IFS=':' read -r METHOD PLAN_THR NORMAL_DEG MATCH_THRESH <<< "$method_spec"

    echo "==== Method: $METHOD ===="
    echo "  plan=${PLAN_THR}  norm=${NORMAL_DEG}°  match=${MATCH_THRESH}  depth_rel=${DEPTH_THRESHOLD}"
    echo ""

    for ds_spec in "${DATASETS[@]}"; do
        IFS=':' read -r DS_NAME INFER_SCRIPT EVAL_SCRIPT OUTPUT_BASE EXTRA_ARGS <<< "$ds_spec"
        OUTPUT_ROOT="${OUTPUT_BASE}/${METHOD}_h5"

        echo "  -- Dataset: $DS_NAME --"
        echo "     output: $OUTPUT_ROOT"

        # ── Inference (GPU) ──
        JOB_INFER=$(sbatch --parsable \
            --job-name="infer_${METHOD}_${DS_NAME}" \
            --time=12:00:00 \
            --cpus-per-task=8 \
            --mem-per-cpu=16G \
            --gpus=rtx_3090:1 \
            --output="$LOG_DIR/infer_${METHOD}_${DS_NAME}_%j.out" \
            --error="$LOG_DIR/infer_${METHOD}_${DS_NAME}_%j.err" \
            --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
mkdir -p $OUTPUT_ROOT
python $PROJECT_ROOT/inference/planarity/$INFER_SCRIPT \
    --model_path $MODEL_PATH \
    --output_root $OUTPUT_ROOT \
    --seg_version $SEG_VERSION \
    --threshold_planarity $PLAN_THR \
    --normal_threshold_deg $NORMAL_DEG \
    --depth_threshold $DEPTH_THRESHOLD \
    --neighbor_match_count_thresh $MATCH_THRESH \
    $EXTRA_ARGS
'")
        echo "     [INFER] $JOB_INFER"

        # ── Evaluation (CPU, depends on inference) ──
        JOB_EVAL=$(sbatch --parsable \
            --job-name="eval_${METHOD}_${DS_NAME}" \
            --dependency=afterok:$JOB_INFER \
            --time=12:00:00 \
            --cpus-per-task=16 \
            --mem-per-cpu=4G \
            --output="$LOG_DIR/eval_${METHOD}_${DS_NAME}_%j.out" \
            --error="$LOG_DIR/eval_${METHOD}_${DS_NAME}_%j.err" \
            --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
python $PROJECT_ROOT/evaluation/quantitative/$EVAL_SCRIPT --methods $METHOD
'")
        echo "     [EVAL]  $JOB_EVAL (after $JOB_INFER)"
        echo "     Chain:  $JOB_INFER → $JOB_EVAL"
        echo ""
    done
done

echo "============================================================"
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: $LOG_DIR"
echo "============================================================"

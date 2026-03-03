#!/bin/bash
# Segmentation experiments: seg_v10, seg_v10_merge_v5, seg_v9vote
# Three independent pipelines: GPU inference → distributed CPU eval → aggregation
#
# Usage:
#   bash run_seg_experiments.sh [scannetpp|hypersim] [NUM_EVAL_JOBS]
#   bash run_seg_experiments.sh              # defaults: scannetpp, 5 eval jobs
#   bash run_seg_experiments.sh scannetpp 8  # 8 eval jobs

set -euo pipefail

DATASET="${1:-scannetpp}"
NUM_EVAL_JOBS="${2:-5}"

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/inference"
mkdir -p "$LOG_DIR"

MODEL_PATH="/cluster/scratch/ayavuz/moge_mixed_output_476644_fixed_cosLR_singlePhase_mixed_HiRes/model_epoch3.pt"
SPLIT="test"
BATCH_SIZE=8
NUM_TOKENS=1600

DATASET_DIR="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
RGB_ROOT="/cluster/project/cvg/Shared_datasets/scannet++/data"
TOTAL_SCENES=42

CONDA_CMD="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh || true; conda activate planamono"
HDF5_FIX="export HDF5_USE_FILE_LOCKING=FALSE"

echo "============================================================"
echo "Segmentation Experiments Pipeline"
echo "Dataset:    $DATASET"
echo "Eval jobs:  $NUM_EVAL_JOBS"
echo "Model:      $MODEL_PATH"
echo "Split:      $SPLIT"
echo "============================================================"

SCENES_PER_JOB=$(( (TOTAL_SCENES + NUM_EVAL_JOBS - 1) / NUM_EVAL_JOBS ))

# ============================================================
# Helper: submit 3-stage pipeline (infer → eval → agg)
# ============================================================
submit_pipeline() {
    local METHOD_NAME="$1"
    local OUTPUT_ROOT="$2"
    local INFER_EXTRA_ARGS="$3"
    local TAG="$4"

    echo ""
    echo "--- Pipeline: $METHOD_NAME ---"
    echo "Output: $OUTPUT_ROOT"

    # Job 1: GPU inference + segmentation
    local INFER_JOB
    INFER_JOB=$(sbatch --parsable \
        --job-name="infer_${TAG}" \
        --time=24:00:00 \
        --cpus-per-task=8 \
        --mem-per-cpu=16G \
        --gpus=rtx_3090:1 \
        --output="$LOG_DIR/${TAG}_infer_%j.out" \
        --error="$LOG_DIR/${TAG}_infer_%j.err" \
        --wrap="
$CONDA_CMD
$HDF5_FIX
mkdir -p $OUTPUT_ROOT
python $PROJECT_ROOT/inference/planarity/inference_to_h5.py \
    --model_path $MODEL_PATH \
    --output_root $OUTPUT_ROOT \
    --dataset_dir $DATASET_DIR \
    --rgb_root $RGB_ROOT \
    --split $SPLIT \
    --batch_size $BATCH_SIZE \
    --num_tokens $NUM_TOKENS \
    --metric_depth \
    $INFER_EXTRA_ARGS
")
    echo "[SUBMITTED] Inference job: $INFER_JOB"

    # Job 2: CPU evaluation (array job)
    local EVAL_JOB
    EVAL_JOB=$(sbatch --parsable \
        --job-name="eval_${TAG}" \
        --dependency=afterok:$INFER_JOB \
        --array=0-$((NUM_EVAL_JOBS - 1)) \
        --time=8:00:00 \
        --cpus-per-task=16 \
        --mem-per-cpu=4G \
        --output="$LOG_DIR/${TAG}_eval_%A_%a.out" \
        --error="$LOG_DIR/${TAG}_eval_%A_%a.err" \
        --wrap="
$CONDA_CMD
$HDF5_FIX
SCENE_START=\$(( SLURM_ARRAY_TASK_ID * $SCENES_PER_JOB ))
SCENE_END=\$(( SCENE_START + $SCENES_PER_JOB ))
if [ \$SCENE_END -gt $TOTAL_SCENES ]; then SCENE_END=$TOTAL_SCENES; fi
echo \"Eval scenes [\$SCENE_START:\$SCENE_END]\"
python $PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py \
    --methods $METHOD_NAME \
    --scene-start \$SCENE_START \
    --scene-end \$SCENE_END
")
    echo "[SUBMITTED] Evaluation array job: $EVAL_JOB (${NUM_EVAL_JOBS} tasks, depends on $INFER_JOB)"

    # Job 3: Aggregate results
    local AGG_JOB
    AGG_JOB=$(sbatch --parsable \
        --job-name="agg_${TAG}" \
        --dependency=afterok:$EVAL_JOB \
        --time=0:10:00 \
        --cpus-per-task=2 \
        --mem-per-cpu=4G \
        --output="$LOG_DIR/${TAG}_agg_%j.out" \
        --error="$LOG_DIR/${TAG}_agg_%j.err" \
        --wrap="
$CONDA_CMD
$HDF5_FIX
python $PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py \
    --methods $METHOD_NAME \
    --aggregate-only
")
    echo "[SUBMITTED] Aggregation job: $AGG_JOB (depends on $EVAL_JOB)"
    echo "Pipeline: $INFER_JOB (infer) -> $EVAL_JOB (eval x${NUM_EVAL_JOBS}) -> $AGG_JOB (agg)"
}


# ============================================================
# Experiment 1: moge_hires_ep3_seg_v10 (adaptive threshold, no merge)
# ============================================================
submit_pipeline \
    "moge_hires_ep3_seg_v10" \
    "/cluster/scratch/aoezkan/planeseg/scannetpp/inference/moge_hires_ep3_seg_v10_h5" \
    "--seg_version v10 \
     --threshold_planarity 0.3 \
     --normal_threshold_deg 5.0 \
     --depth_threshold 0.025 \
     --neighbor_match_count_thresh 24 \
     --adaptive_frac 0.75 \
     --min_valid_neighbors 3 \
     --min_segment_pixels 50" \
    "moge_hires_ep3_seg_v10"


# ============================================================
# Experiment 2: moge_hires_ep3_seg_v10_merge_v5 (adaptive threshold + merge_v5)
# ============================================================
submit_pipeline \
    "moge_hires_ep3_seg_v10_merge_v5" \
    "/cluster/scratch/aoezkan/planeseg/scannetpp/inference/moge_hires_ep3_seg_v10_merge_v5_h5" \
    "--seg_version v10 \
     --threshold_planarity 0.3 \
     --normal_threshold_deg 5.0 \
     --depth_threshold 0.025 \
     --neighbor_match_count_thresh 24 \
     --adaptive_frac 0.75 \
     --min_valid_neighbors 3 \
     --min_segment_pixels 50 \
     --merge_version v5 \
     --merge_normal_deg 5.0 \
     --merge_offset_m 0.05 \
     --merge_min_pixels 100 \
     --merge_gap_px 20 \
     --merge_nn_dist_m 0.2 \
     --merge_topk 20" \
    "moge_hires_ep3_seg_v10_merge_v5"


# ============================================================
# Experiment 3: moge_hires_ep3_seg_v9vote (planarity voting, no merge)
# ============================================================
submit_pipeline \
    "moge_hires_ep3_seg_v9vote" \
    "/cluster/scratch/aoezkan/planeseg/scannetpp/inference/moge_hires_ep3_seg_v9vote_h5" \
    "--seg_version v9_vote \
     --normal_threshold_deg 10.0 \
     --depth_threshold 0.05 \
     --neighbor_match_count_thresh 18 \
     --planarity_threshold 0.6 \
     --planarity_ratio 0.5" \
    "moge_hires_ep3_seg_v9vote"


echo ""
echo "============================================================"
echo "All 3 pipelines submitted successfully!"
echo "============================================================"

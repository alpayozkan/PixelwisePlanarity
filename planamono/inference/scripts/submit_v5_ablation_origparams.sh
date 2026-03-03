#!/bin/bash
# Submit v5 segmentation ablation jobs with ORIGINAL inference_to_h5.py parameters.
#
# Same 2x2 ablation grid as submit_v5_ablation.sh but using the default v5 params:
#   planarity=0.6, normal=10°, depth_abs=0.05m, depth_rel=0.025, match_thresh=24
#
# Usage:
#   bash submit_v5_ablation_origparams.sh                                   # all 4 variants
#   bash submit_v5_ablation_origparams.sh v5 v5_relative                    # specific variants

set -euo pipefail

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/v5_ablation_origparams"
mkdir -p "$LOG_DIR"

# Shared raw MoGe predictions (Stage 1 output)
RAW_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference_raw/moge_hires_ep3_raw"
H5_BASE="/cluster/scratch/aoezkan/planeseg/scannetpp/inference"

# Original v5 parameters (from inference_to_h5.py defaults)
THRESHOLD_PLANARITY=0.6
NORMAL_THRESHOLD_DEG=10.0
NEIGHBOR_MATCH_COUNT_THRESH=24

# Depth thresholds per type
DEPTH_THRESHOLD_ABS=0.05    # meters (for v5 and v5_no_sobel)
DEPTH_THRESHOLD_REL=0.025   # fraction of center depth (for v5_relative and v5_dotprod_relative)

# Eval parameters
NUM_EVAL_JOBS=5
TOTAL_SCENES=42
SCENES_PER_JOB=$(( (TOTAL_SCENES + NUM_EVAL_JOBS - 1) / NUM_EVAL_JOBS ))

# ---- Variant definitions ----
declare -A VARIANT_SEG_VERSION
declare -A VARIANT_DEPTH_THRESHOLD
declare -A VARIANT_EVAL_METHOD

VARIANT_SEG_VERSION[v5]="v5"
VARIANT_DEPTH_THRESHOLD[v5]="$DEPTH_THRESHOLD_ABS"
VARIANT_EVAL_METHOD[v5]="moge_hires_ep3_v5origparams_seg"

VARIANT_SEG_VERSION[v5_relative]="v5_relative"
VARIANT_DEPTH_THRESHOLD[v5_relative]="$DEPTH_THRESHOLD_REL"
VARIANT_EVAL_METHOD[v5_relative]="moge_hires_ep3_v5origparams_relative_seg"

VARIANT_SEG_VERSION[v5_no_sobel]="v5_no_sobel"
VARIANT_DEPTH_THRESHOLD[v5_no_sobel]="$DEPTH_THRESHOLD_ABS"
VARIANT_EVAL_METHOD[v5_no_sobel]="moge_hires_ep3_v5origparams_no_sobel_seg"

VARIANT_SEG_VERSION[v5_dotprod_relative]="v5_dotprod_relative"
VARIANT_DEPTH_THRESHOLD[v5_dotprod_relative]="$DEPTH_THRESHOLD_REL"
VARIANT_EVAL_METHOD[v5_dotprod_relative]="moge_hires_ep3_v5origparams_dotprod_relative_seg"

ALL_VARIANTS=(v5 v5_relative v5_no_sobel v5_dotprod_relative)

# Determine which variants to run
if [ $# -eq 0 ]; then
    VARIANTS=("${ALL_VARIANTS[@]}")
else
    VARIANTS=("$@")
fi

echo "============================================================"
echo "v5 Segmentation Ablation — ORIGINAL PARAMS (ScanNet++)"
echo "============================================================"
echo "Raw input:    $RAW_ROOT"
echo "Variants:     ${VARIANTS[*]}"
echo "Planarity θ:  $THRESHOLD_PLANARITY"
echo "Normal θ:     ${NORMAL_THRESHOLD_DEG}°"
echo "Depth abs:    ${DEPTH_THRESHOLD_ABS}m"
echo "Depth rel:    ${DEPTH_THRESHOLD_REL}"
echo "Match thresh: $NEIGHBOR_MATCH_COUNT_THRESH"
echo "Eval jobs:    $NUM_EVAL_JOBS"
echo "============================================================"
echo ""

for VARIANT in "${VARIANTS[@]}"; do
    SEG_VERSION="${VARIANT_SEG_VERSION[$VARIANT]}"
    DEPTH_THR="${VARIANT_DEPTH_THRESHOLD[$VARIANT]}"
    EVAL_METHOD="${VARIANT_EVAL_METHOD[$VARIANT]}"
    OUTPUT_ROOT="${H5_BASE}/${EVAL_METHOD}_h5"

    echo "--- Submitting: $VARIANT (origparams) ---"
    echo "  seg_version:     $SEG_VERSION"
    echo "  depth_threshold: $DEPTH_THR"
    echo "  eval_method:     $EVAL_METHOD"
    echo "  output:          $OUTPUT_ROOT"

    # --- Job 1: Segmentation (GPU for torch ops) ---
    SEG_JOB=$(sbatch --parsable \
        --job-name="seg_op_${VARIANT}" \
        --time=2:00:00 \
        --cpus-per-task=8 \
        --mem-per-cpu=8G \
        --gpus=rtx_3090:1 \
        --output="$LOG_DIR/seg_${VARIANT}_%j.out" \
        --error="$LOG_DIR/seg_${VARIANT}_%j.err" \
        --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
python $PROJECT_ROOT/inference/planarity/segment_from_raw.py \
    --raw_root $RAW_ROOT \
    --output_root $OUTPUT_ROOT \
    --dataset scannetpp \
    --seg_version $SEG_VERSION \
    --threshold_planarity $THRESHOLD_PLANARITY \
    --normal_threshold_deg $NORMAL_THRESHOLD_DEG \
    --depth_threshold $DEPTH_THR \
    --neighbor_match_count_thresh $NEIGHBOR_MATCH_COUNT_THRESH
'")
    echo "  [SEG] Job: $SEG_JOB"

    # --- Job 2: Evaluation (array job, depends on segmentation) ---
    EVAL_JOB=$(sbatch --parsable \
        --job-name="eval_op_${VARIANT}" \
        --dependency=afterok:$SEG_JOB \
        --array=0-$((NUM_EVAL_JOBS - 1)) \
        --time=8:00:00 \
        --cpus-per-task=16 \
        --mem-per-cpu=4G \
        --output="$LOG_DIR/eval_${VARIANT}_%A_%a.out" \
        --error="$LOG_DIR/eval_${VARIANT}_%A_%a.err" \
        --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
SCENE_START=\$(( SLURM_ARRAY_TASK_ID * $SCENES_PER_JOB ))
SCENE_END=\$(( SCENE_START + $SCENES_PER_JOB ))
if [ \$SCENE_END -gt $TOTAL_SCENES ]; then SCENE_END=$TOTAL_SCENES; fi
echo \"Eval scenes [\$SCENE_START:\$SCENE_END] for $VARIANT (origparams)\"
python $PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py \
    --methods $EVAL_METHOD \
    --scene-start \$SCENE_START \
    --scene-end \$SCENE_END
'")
    echo "  [EVAL] Array job: $EVAL_JOB ($NUM_EVAL_JOBS tasks, depends on $SEG_JOB)"

    # --- Job 3: Aggregation (depends on all eval jobs) ---
    AGG_JOB=$(sbatch --parsable \
        --job-name="agg_op_${VARIANT}" \
        --dependency=afterok:$EVAL_JOB \
        --time=0:10:00 \
        --cpus-per-task=2 \
        --mem-per-cpu=4G \
        --output="$LOG_DIR/agg_${VARIANT}_%j.out" \
        --error="$LOG_DIR/agg_${VARIANT}_%j.err" \
        --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
python $PROJECT_ROOT/evaluation/quantitative/evaluate_all_baselines.py \
    --methods $EVAL_METHOD \
    --aggregate-only
'")
    echo "  [AGG] Job: $AGG_JOB (depends on $EVAL_JOB)"
    echo "  Pipeline: $SEG_JOB → $EVAL_JOB → $AGG_JOB"
    echo ""
done

echo "============================================================"
echo "All variant jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: $LOG_DIR"
echo "============================================================"

#!/bin/bash
# Hypersim ablation studies for moge_mixed_bce_476644 epoch 6
# 4 jobs total:
#   GT Planarity + Our Seg inference (GPU) → eval (CPU)
#   Our Planarity + GT Seg inference (GPU) → eval (CPU)
# Run: bash run_hypersim_ablations_476644_ep6.sh

set -euo pipefail

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/inference"
mkdir -p "$LOG_DIR"

# --- Job 1: GT Planarity + Our Seg (inference, GPU) ---
JOB_GTPLAN_INFER=$(sbatch --parsable \
    --time=12:00:00 \
    --cpus-per-task=8 \
    --mem-per-cpu=8G \
    --gpus=rtx_3090:1 \
    --output="$LOG_DIR/hypersim_gtplanarity_ourseg_476644_ep6_infer_%j.out" \
    --error="$LOG_DIR/hypersim_gtplanarity_ourseg_476644_ep6_infer_%j.err" \
    --wrap="bash -c 'source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono && cd $PROJECT_ROOT/evaluation/quantitative && python evaluate_hypersim_gtplanarity_ourseg_fast.py --inference-only'")

echo "Submitted Hypersim GT Plan + Our Seg infer:  $JOB_GTPLAN_INFER"

# --- Job 2: Our Planarity + GT Seg (inference, GPU) ---
JOB_OURPLAN_INFER=$(sbatch --parsable \
    --time=12:00:00 \
    --cpus-per-task=8 \
    --mem-per-cpu=8G \
    --gpus=rtx_3090:1 \
    --output="$LOG_DIR/hypersim_ourplanarity_gtseg_476644_ep6_infer_%j.out" \
    --error="$LOG_DIR/hypersim_ourplanarity_gtseg_476644_ep6_infer_%j.err" \
    --wrap="bash -c 'source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono && cd $PROJECT_ROOT/evaluation/quantitative && python evaluate_hypersim_ourplanarity_gtseg_fast.py --inference-only'")

echo "Submitted Hypersim Our Plan + GT Seg infer:  $JOB_OURPLAN_INFER"

# --- Job 3: GT Planarity + Our Seg (eval, CPU, depends on Job 1) ---
JOB_GTPLAN_EVAL=$(sbatch --parsable \
    --dependency=afterok:$JOB_GTPLAN_INFER \
    --time=12:00:00 \
    --cpus-per-task=16 \
    --mem-per-cpu=8G \
    --output="$LOG_DIR/hypersim_gtplanarity_ourseg_476644_ep6_eval_%j.out" \
    --error="$LOG_DIR/hypersim_gtplanarity_ourseg_476644_ep6_eval_%j.err" \
    --wrap="bash -c 'source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono && python $PROJECT_ROOT/evaluation/quantitative/evaluate_hypersim_all_baselines.py --methods gtplanarity_ourseg_476644_ep6'")

echo "Submitted Hypersim GT Plan + Our Seg eval:   $JOB_GTPLAN_EVAL (after $JOB_GTPLAN_INFER)"

# --- Job 4: Our Planarity + GT Seg (eval, CPU, depends on Job 2) ---
JOB_OURPLAN_EVAL=$(sbatch --parsable \
    --dependency=afterok:$JOB_OURPLAN_INFER \
    --time=12:00:00 \
    --cpus-per-task=16 \
    --mem-per-cpu=8G \
    --output="$LOG_DIR/hypersim_ourplanarity_gtseg_476644_ep6_eval_%j.out" \
    --error="$LOG_DIR/hypersim_ourplanarity_gtseg_476644_ep6_eval_%j.err" \
    --wrap="bash -c 'source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono && python $PROJECT_ROOT/evaluation/quantitative/evaluate_hypersim_all_baselines.py --methods ourplanarity_gtseg_476644_ep6'")

echo "Submitted Hypersim Our Plan + GT Seg eval:   $JOB_OURPLAN_EVAL (after $JOB_OURPLAN_INFER)"

echo ""
echo "Output dirs:"
echo "  GT Plan + Our Seg H5:   /cluster/scratch/aoezkan/planeseg/hypersim/inference/hypersim_gtplanarity_ourseg_moge_mixed_bce_476644_ep6_v1_h5/"
echo "  Our Plan + GT Seg H5:   /cluster/scratch/aoezkan/planeseg/hypersim/inference/hypersim_ourplanarity_gtseg_moge_mixed_bce_476644_ep6_v1_h5/"
echo "  GT Plan + Our Seg eval: /cluster/scratch/aoezkan/planeseg/hypersim/eval/hypersim_gtplanarity_ourseg_moge_mixed_bce_476644_ep6_v1/"
echo "  Our Plan + GT Seg eval: /cluster/scratch/aoezkan/planeseg/hypersim/eval/hypersim_ourplanarity_gtseg_moge_mixed_bce_476644_ep6_v1/"
echo ""
echo "Monitor: squeue -u \$USER"

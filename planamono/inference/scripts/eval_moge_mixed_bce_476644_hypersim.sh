#!/bin/bash
# MoGe Mixed BCE 476644 — Hypersim Evaluation only
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_mixed_bce_476644_hypersim_eval_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_mixed_bce_476644_hypersim_eval_%j.err

set -euo pipefail

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Hypersim evaluation (moge_mixed_bce_476644_ep6)"

python "$PROJECT_ROOT/evaluation/quantitative/evaluate_hypersim_all_baselines.py" \
    --methods moge_mixed_bce_476644_ep6

echo "[DONE] Hypersim eval → /cluster/scratch/aoezkan/planeseg/hypersim/eval/moge_mixed_bce_476644_v3/"

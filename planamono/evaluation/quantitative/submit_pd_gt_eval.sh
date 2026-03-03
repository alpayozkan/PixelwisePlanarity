#!/bin/bash
# Submit GT evaluation on Parallel Domain (CPU only, no inference needed)
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/pd_gt_eval_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/pd_gt_eval_%j.err

set -euo pipefail

source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Parallel Domain GT evaluation"
python /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative/evaluate_pd_all_baselines.py \
    --methods gt --split val

echo "[DONE] PD GT eval"

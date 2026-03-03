#!/bin/bash
#SBATCH --job-name=vis_unified
#SBATCH --output=logs/vis_unified_%j.out
#SBATCH --error=logs/vis_unified_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ============================================================
# Unified Visualization Script
# ============================================================
# Usage (local):
#   ./run_visualizations.sh                                         # All datasets, 20 samples
#   ./run_visualizations.sh --datasets scannetpp --n-samples 5      # ScanNet++ only
#   ./run_visualizations.sh --datasets hypersim --methods GT moge_mixed_bce
#   ./run_visualizations.sh --mode slurm --datasets scannetpp hypersim
#
# Usage (SLURM):
#   sbatch run_visualizations.sh                                    # Submit directly
#   ./run_visualizations.sh --mode slurm --datasets scannetpp       # Submit sub-jobs
# ============================================================

set -e

# Activate conda environment
source ~/.bashrc
conda activate planeseg

# Navigate to evaluation directory
cd /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative

echo "============================================================"
echo "Unified Visualization Pipeline"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Default: local mode, both datasets, 20 samples
python run_visualizations.py \
    --mode local \
    --datasets scannetpp hypersim \
    --n-samples 20 \
    --random-seed 42 \
    --format both \
    "$@"

echo ""
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"

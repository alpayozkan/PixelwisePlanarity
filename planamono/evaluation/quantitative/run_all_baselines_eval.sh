#!/bin/bash
#SBATCH --job-name=eval_baselines
#SBATCH --output=eval_baselines_%j.out
#SBATCH --error=eval_baselines_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# ============================================================
# Baseline Evaluation Script
# ============================================================
# Usage:
#   ./run_all_baselines_eval.sh              # Run all methods
#   ./run_all_baselines_eval.sh ours         # Run single method
#   ./run_all_baselines_eval.sh --agg        # Aggregate only
#   sbatch run_all_baselines_eval.sh         # Submit to SLURM
# ============================================================

set -e  # Exit on error

# Activate conda environment
source ~/.bashrc
conda activate planeseg

# Navigate to evaluation directory
cd /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative

echo "============================================================"
echo "Baseline Evaluation"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Parse arguments
if [ "$1" == "--agg" ] || [ "$1" == "--aggregate-only" ]; then
    echo "Mode: Aggregate only"
    python evaluate_all_baselines.py --aggregate-only --output-dir .
elif [ -n "$1" ]; then
    echo "Mode: Evaluate specific methods: $@"
    python evaluate_all_baselines.py --methods $@ --output-dir .
else
    echo "Mode: Evaluate all methods"
    python evaluate_all_baselines.py --output-dir .
fi

echo ""
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"

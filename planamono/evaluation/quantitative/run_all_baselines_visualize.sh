#!/bin/bash
#SBATCH --job-name=vis_baselines
#SBATCH --output=vis_baselines_%j.out
#SBATCH --error=vis_baselines_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ============================================================
# Baseline Visualization Script
# ============================================================
# Usage:
#   ./run_all_baselines_visualize.sh                    # 10 random samples
#   ./run_all_baselines_visualize.sh 20                 # 20 random samples
#   ./run_all_baselines_visualize.sh 10 123             # 10 samples, seed 123
#   ./run_all_baselines_visualize.sh --specific "scene1:frame1,scene2:frame2"
#   sbatch run_all_baselines_visualize.sh               # Submit to SLURM
# ============================================================

set -e  # Exit on error

# Activate conda environment
source ~/.bashrc
conda activate planeseg

# Navigate to evaluation directory
cd /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative

echo "============================================================"
echo "Baseline Inlier Visualization"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Default values
N_SAMPLES=20
SEED=42

# Parse arguments
if [ "$1" == "--specific" ]; then
    # Specific frames mode
    echo "Mode: Specific frames"
    echo "Frames: $2"
    python visualize_scannetpp_all_baselines_v1.py --specific-frames "$2"
    # python visualize_scannetpp_all_baselines.py --specific-frames "$2"
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage:"
    echo "  ./run_all_baselines_visualize.sh [N_SAMPLES] [SEED]"
    echo "  ./run_all_baselines_visualize.sh --specific \"scene1:frame1,scene2:frame2\""
    echo ""
    echo "Examples:"
    echo "  ./run_all_baselines_visualize.sh              # 10 samples, seed 42"
    echo "  ./run_all_baselines_visualize.sh 20           # 20 samples, seed 42"
    echo "  ./run_all_baselines_visualize.sh 20 123       # 20 samples, seed 123"
    echo "  ./run_all_baselines_visualize.sh --specific \"f3d64c30f8:frame_006300\""
    exit 0
else
    # Random samples mode
    if [ -n "$1" ]; then
        N_SAMPLES=$1
    fi
    if [ -n "$2" ]; then
        SEED=$2
    fi

    echo "Mode: Random sampling"
    echo "N samples: $N_SAMPLES"
    echo "Random seed: $SEED"

    # python visualize_scannetpp_all_baselines.py \
    python visualize_scannetpp_all_baselines_v1.py \
        --n-samples $N_SAMPLES \
        --random-seed $SEED
fi

echo ""
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"

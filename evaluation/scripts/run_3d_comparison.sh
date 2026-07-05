#!/bin/bash
# Generate 3D plane-comparison renders (GT vs prediction) for a list of
# ScanNet++ scenes.
#
# Usage:
#   ./run_3d_comparison.sh [scene_list] [output_root] [checkpoint]
#
# For SLURM: uncomment the directives, mkdir -p logs, then sbatch run_3d_comparison.sh
##SBATCH --job-name=3d_vis
##SBATCH --output=logs/3d_vis_%j.out
##SBATCH --error=logs/3d_vis_%j.err
##SBATCH --time=12:00:00
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=8G
##SBATCH --gpus=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

SCENE_LIST="${1:-$REPO_ROOT/evaluation/qualitative/scannetpp_vis_scenes.txt}"
OUTPUT_ROOT="${2:-/cluster/scratch/aoezkan/planeseg/3d_vis/scannetpp}"
CHECKPOINT="${3:-$(python -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from paths import planarity_model_path; print(planarity_model_path)")}"

python "$REPO_ROOT/evaluation/qualitative/generate_3d_comparison.py" \
    --scene_list "$SCENE_LIST" \
    --output_root "$OUTPUT_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --num_tokens 1600 \
    --rotations 0 1 2 3 \
    --rot_x 20 \
    --point_radius 1

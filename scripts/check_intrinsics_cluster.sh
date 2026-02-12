#!/bin/bash
# Check that camera intrinsics are constant across all scenes/frames
# for SYNTHIA and VKITTI2 on the cluster.
#
# Usage:
#   sbatch scripts/check_intrinsics_cluster.sh
#   # or directly:
#   bash scripts/check_intrinsics_cluster.sh

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/check_intrinsics_%j.out
#SBATCH --error=logs/check_intrinsics_%j.err

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Cluster paths
SYNTHIA_ROOT="/cluster/scratch/ayavuz/dataset/synthia"
VKITTI2_ROOT="/cluster/scratch/ayavuz/dataset/virtual_kitti"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p logs

echo "============================================"
echo "Intrinsics Constancy Check"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================"

python scripts/check_intrinsics_constant.py \
    --synthia "$SYNTHIA_ROOT" \
    --vkitti2 "$VKITTI2_ROOT"

EXIT_CODE=$?

echo ""
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE

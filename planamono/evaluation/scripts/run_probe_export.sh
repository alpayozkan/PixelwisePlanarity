#!/bin/bash
#SBATCH --job-name=probe_export
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=logs/probe_export_%j.out
#SBATCH --error=logs/probe_export_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module load stack/2024-05  gcc/13.2.0 eth_proxy

# Set paths
REPO_ROOT="/cluster/home/ayavuz/PixelwisePlanarity"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Parameters
DATASET=${1:-all}
PROBE_CKPT=${2:-/cluster/scratch/ayavuz/moge_HIRES_4datasets_NORMAL_PROBE/probe_epoch2.pt}
OUTPUT_DIR=${3:-}

echo "=============================================="
echo "Probe Export (Frozen MoGe + Conv Probe -> plan2seg)"
echo "=============================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $DATASET"
echo "Probe checkpoint: $PROBE_CKPT"
echo "=============================================="

# Build command
CMD="python $REPO_ROOT/planamono/evaluation/run_probe_export.py \
    --probe_checkpoint $PROBE_CKPT \
    --dataset $DATASET"

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

eval $CMD

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="

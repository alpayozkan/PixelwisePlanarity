#!/bin/bash
# MoGe Planarity Inference + Segmentation → H5
# Runs full pipeline: MoGe inference → vectorized segmentation → planes.h5
#
# Usage:
#   ./run_inference_to_h5.sh <model_path> <output_root> [split] [batch_size]
#
# Examples:
#   ./run_inference_to_h5.sh /path/to/model.pt /path/to/output_h5
#   ./run_inference_to_h5.sh /path/to/model.pt /path/to/output_h5 val 8
#   ./run_inference_to_h5.sh /path/to/model.pt /path/to/output_h5 test 16

# SLURM options (uncomment for cluster use)
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/inference_h5_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/inference_h5_%j.err

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"

# Change to project root to ensure relative paths work
cd "$PROJECT_ROOT"

# Activate conda environment
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
echo "[INFO] Activated conda environment: planamono"

# Parse arguments
MODEL_PATH="${1}"
OUTPUT_ROOT="${2}"
SPLIT="${3:-val}"
BATCH_SIZE="${4:-8}"

# Validate required arguments
if [[ -z "$MODEL_PATH" ]] || [[ -z "$OUTPUT_ROOT" ]]; then
    echo "Usage: $0 <model_path> <output_root> [split] [batch_size]"
    echo ""
    echo "Arguments:"
    echo "  model_path    Path to trained MoGe checkpoint (.pt file)"
    echo "  output_root   Root directory for H5 output"
    echo "  split         Dataset split: train/val/test (default: val)"
    echo "  batch_size    Batch size for inference (default: 8)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/model.pt /path/to/output_h5"
    echo "  $0 /path/to/model.pt /path/to/output_h5 val 8"
    echo "  $0 /path/to/model.pt /path/to/output_h5 test 16"
    exit 1
fi

# Validate model exists
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ERROR] Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

# Default dataset paths (modify if needed)
DATASET_DIR="${DATASET_DIR:-/cluster/scratch/aoezkan/planeseg/dataset/scannetpp}"
RGB_ROOT="${RGB_ROOT:-/cluster/project/cvg/Shared_datasets/scannet++/data}"

# Segmentation parameters (can be overridden via environment variables)
THRESHOLD_PLANARITY="${THRESHOLD_PLANARITY:-0.6}"
NORMAL_THRESHOLD_DEG="${NORMAL_THRESHOLD_DEG:-10.0}"
DEPTH_THRESHOLD="${DEPTH_THRESHOLD:-0.05}"
NEIGHBOR_MATCH_COUNT_THRESH="${NEIGHBOR_MATCH_COUNT_THRESH:-24}"
NUM_TOKENS="${NUM_TOKENS:-1024}"

echo "=========================================="
echo "MoGe Inference + Segmentation → H5"
echo "=========================================="
echo "Model:          $MODEL_PATH"
echo "Output:         $OUTPUT_ROOT"
echo "Split:          $SPLIT"
echo "Batch size:     $BATCH_SIZE"
echo "Dataset dir:    $DATASET_DIR"
echo "RGB root:       $RGB_ROOT"
echo "------------------------------------------"
echo "Segmentation parameters:"
echo "  Planarity θ:  $THRESHOLD_PLANARITY"
echo "  Normal θ:     $NORMAL_THRESHOLD_DEG°"
echo "  Depth θ:      $DEPTH_THRESHOLD m"
echo "  Neighbor θ:   $NEIGHBOR_MATCH_COUNT_THRESH"
echo "  Num tokens:   $NUM_TOKENS"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# Create logs directory in scratch for SLURM
mkdir -p /cluster/scratch/aoezkan/planeseg/logs

# Run inference
python "$PROJECT_ROOT/inference/planarity/inference_to_h5.py" \
    --model_path "$MODEL_PATH" \
    --output_root "$OUTPUT_ROOT" \
    --dataset_dir "$DATASET_DIR" \
    --rgb_root "$RGB_ROOT" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --num_tokens "$NUM_TOKENS" \
    --threshold_planarity "$THRESHOLD_PLANARITY" \
    --normal_threshold_deg "$NORMAL_THRESHOLD_DEG" \
    --depth_threshold "$DEPTH_THRESHOLD" \
    --neighbor_match_count_thresh "$NEIGHBOR_MATCH_COUNT_THRESH"

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "[SUCCESS] Inference completed"
    echo "[INFO] Results saved to: $OUTPUT_ROOT"
    echo ""
    echo "Output structure:"
    echo "  $OUTPUT_ROOT/{scene_id}/planes.h5"
    echo "    - planes: (N_frames, H, W) segmentation labels"
    echo "    - frame_ids: list of frame IDs"
else
    echo "[ERROR] Inference failed"
    exit 1
fi

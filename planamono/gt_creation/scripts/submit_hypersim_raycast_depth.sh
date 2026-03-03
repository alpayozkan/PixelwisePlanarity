#!/bin/bash
# Submit parallel Hypersim raycasted depth jobs
# Splits scene list into 15 parts and submits each as a separate SLURM job
#
# Usage:
#   bash submit_hypersim_raycast_depth.sh                        # z-depth (default)
#   bash submit_hypersim_raycast_depth.sh --depth_type euclidean # Euclidean depth
#   bash submit_hypersim_raycast_depth.sh --depth_type both      # both types

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Parse --depth_type from any position in args
DEPTH_TYPE="zdepth"
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --depth_type)
            DEPTH_TYPE="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Configuration (positional args override defaults)
PARAMS_ROOT="${POSITIONAL_ARGS[0]:-/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params}"
PLANE_ROOT="${POSITIONAL_ARGS[1]:-/cluster/scratch/ayavuz/dataset/hypersim_mesh_ours}"
OUTPUT_ROOT="${POSITIONAL_ARGS[2]:-/cluster/scratch/aoezkan/planeseg/dataset/hypersim}"
FRAME_SKIP="${POSITIONAL_ARGS[3]:-1}"
METADATA_CSV="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/shared/datasets/metadata_camera_parameters.csv"
NUM_SPLITS=15

# Absolute paths (resolved at submission time using realpath)
BASH_SCRIPT="$(realpath "$SCRIPT_DIR/hypersim_raycast_depth.sh")"
PYTHON_SCRIPT="$(realpath "$SCRIPT_DIR/../hypersim/raycast_depth.py")"

echo "[INFO] BASH_SCRIPT: $BASH_SCRIPT"
echo "[INFO] PYTHON_SCRIPT: $PYTHON_SCRIPT"
echo "[INFO] DEPTH_TYPE: $DEPTH_TYPE"

# Create temporary directory for split files
SPLIT_DIR="$SCRIPT_DIR/splits_raycast_tmp"
mkdir -p "$SPLIT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

# Scene list from repo splits (train + val + test = all scenes)
SCENE_LIST="$(realpath "$SCRIPT_DIR/../../splits/hypersim/all_scenes.txt")"
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Scene list not found: $SCENE_LIST"
    exit 1
fi

# Copy scene list to split dir
cp "$SCENE_LIST" "$SPLIT_DIR/all_scenes.txt"
TOTAL=$(wc -l < "$SPLIT_DIR/all_scenes.txt")
PER_SPLIT=$(( (TOTAL + NUM_SPLITS - 1) / NUM_SPLITS ))

echo "[INFO] Total scenes: $TOTAL"
echo "[INFO] Scenes per split: ~$PER_SPLIT"

# Split into files
split -l $PER_SPLIT "$SPLIT_DIR/all_scenes.txt" "$SPLIT_DIR/split_"

# Determine which depth types to run
if [[ "$DEPTH_TYPE" == "both" ]]; then
    DEPTH_TYPES=("zdepth" "euclidean")
else
    DEPTH_TYPES=("$DEPTH_TYPE")
fi

# Submit jobs for each split and each depth type
i=0
for dtype in "${DEPTH_TYPES[@]}"; do
    for split_file in "$SPLIT_DIR"/split_*; do
        i=$((i + 1))
        count=$(wc -l < "$split_file")
        echo "[INFO] Submitting job $i ($dtype) with $count scenes"

        sbatch --job-name="raycast_${dtype}_$i" \
               --time=72:00:00 \
               --cpus-per-task=4 \
               --mem-per-cpu=16G \
               --output="$SCRIPT_DIR/logs/raycast_${dtype}_${i}_%j.out" \
               --error="$SCRIPT_DIR/logs/raycast_${dtype}_${i}_%j.err" \
               "$BASH_SCRIPT" \
               "$split_file" \
               "$PARAMS_ROOT" \
               "$PLANE_ROOT" \
               "$OUTPUT_ROOT" \
               "$FRAME_SKIP" \
               "$PYTHON_SCRIPT" \
               "$METADATA_CSV" \
               "$dtype"
    done
done

echo "[INFO] Submitted $i jobs"
echo "[INFO] Split files saved in: $SPLIT_DIR"

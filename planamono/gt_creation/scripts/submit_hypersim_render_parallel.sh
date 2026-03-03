#!/bin/bash
# Submit parallel Hypersim plane rendering jobs
# Splits scene list into 8 parts and submits each as a separate SLURM job

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration
# INPUT_ROOT="${1:-/cluster/scratch/ayavuz/dataset/Hypersim_params}"
# PLANE_ROOT="${2:-/cluster/scratch/ayavuz/dataset/Hypersim_ours}"
INPUT_ROOT="${1:-/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params}"
PLANE_ROOT="${2:-/cluster/scratch/ayavuz/dataset/hypersim_mesh_ours}"
# OUTPUT_ROOT="${3:-/cluster/scratch/ayavuz/dataset/Hypersim_rendered}"
OUTPUT_ROOT="${3:-/cluster/scratch/aoezkan/planeseg/dataset/hypersim/plane_rendered}"
FRAME_SKIP="${4:-1}"
METADATA_CSV="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/shared/datasets/metadata_camera_parameters.csv"
NUM_SPLITS=8

# Absolute paths (resolved at submission time using realpath)
BASH_SCRIPT="$(realpath "$SCRIPT_DIR/hypersim_render_planes.sh")"
PYTHON_SCRIPT="$(realpath "$SCRIPT_DIR/../hypersim/rendering.py")"

echo "[INFO] BASH_SCRIPT: $BASH_SCRIPT"
echo "[INFO] PYTHON_SCRIPT: $PYTHON_SCRIPT"

# Create temporary directory for split files
SPLIT_DIR="$SCRIPT_DIR/splits_tmp"
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

# Split into 8 files
split -l $PER_SPLIT "$SPLIT_DIR/all_scenes.txt" "$SPLIT_DIR/split_"

# Submit jobs for each split
i=0
for split_file in "$SPLIT_DIR"/split_*; do
    i=$((i + 1))
    count=$(wc -l < "$split_file")
    echo "[INFO] Submitting job $i with $count scenes"

    sbatch --job-name="render_$i" \
           --time=72:00:00 \
           --cpus-per-task=4 \
           --mem-per-cpu=16G \
           --output="$SCRIPT_DIR/logs/render_${i}_%j.out" \
           --error="$SCRIPT_DIR/logs/render_${i}_%j.err" \
           "$BASH_SCRIPT" \
           "$split_file" \
           "$INPUT_ROOT" \
           "$PLANE_ROOT" \
           "$OUTPUT_ROOT" \
           "$FRAME_SKIP" \
           "$PYTHON_SCRIPT" \
           "$METADATA_CSV"
done

echo "[INFO] Submitted $i jobs"
echo "[INFO] Split files saved in: $SPLIT_DIR"

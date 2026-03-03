#!/bin/bash
# Submit 10 jobs for the 20 remaining euclidean raycast scenes (2 scenes each)
# These are scenes not yet processed by the original 15 jobs, excluding 27 with missing meshes.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASH_SCRIPT="$(realpath "$SCRIPT_DIR/hypersim_raycast_depth.sh")"
PYTHON_SCRIPT="$(realpath "$SCRIPT_DIR/../hypersim/raycast_depth.py")"
METADATA_CSV="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/shared/datasets/metadata_camera_parameters.csv"

PARAMS_ROOT="/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"
PLANE_ROOT="/cluster/scratch/ayavuz/dataset/hypersim_mesh_ours"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/dataset/hypersim"

SPLIT_DIR="$SCRIPT_DIR/splits_raycast_remaining"
mkdir -p "$SPLIT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

# 20 remaining scenes (verified: not yet completed, not missing meshes)
cat > "$SPLIT_DIR/remaining_all.txt" << 'EOF'
ai_044_007
ai_044_008
ai_044_009
ai_044_010
ai_045_001
ai_045_004
ai_045_005
ai_045_006
ai_045_008
ai_045_010
ai_050_002
ai_050_003
ai_050_004
ai_055_004
ai_055_005
ai_055_006
ai_055_007
ai_055_008
ai_055_009
ai_055_010
EOF

# Split into 10 files of 2 scenes each
split -l 2 "$SPLIT_DIR/remaining_all.txt" "$SPLIT_DIR/split_"

# Submit 10 jobs
i=0
for split_file in "$SPLIT_DIR"/split_*; do
    i=$((i + 1))
    count=$(wc -l < "$split_file")
    scenes=$(tr '\n' ' ' < "$split_file")
    echo "[INFO] Submitting job $i with $count scenes: $scenes"

    sbatch --job-name="raycast_euc_r${i}" \
           --time=24:00:00 \
           --cpus-per-task=4 \
           --mem-per-cpu=16G \
           --output="$SCRIPT_DIR/logs/raycast_euc_remaining_${i}_%j.out" \
           --error="$SCRIPT_DIR/logs/raycast_euc_remaining_${i}_%j.err" \
           "$BASH_SCRIPT" \
           "$split_file" \
           "$PARAMS_ROOT" \
           "$PLANE_ROOT" \
           "$OUTPUT_ROOT" \
           "1" \
           "$PYTHON_SCRIPT" \
           "$METADATA_CSV" \
           "euclidean"
done

echo ""
echo "[INFO] Submitted $i jobs for 20 remaining scenes"
echo "[INFO] Split files saved in: $SPLIT_DIR"

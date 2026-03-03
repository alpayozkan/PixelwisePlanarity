#!/bin/bash
# Test Hypersim Plane Rendering (fixed remap_plane_ids)
# Submits 5 scenes as separate SLURM jobs with frame_skip=1 to verify rendering.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/hypersim_render_test"
mkdir -p "$LOG_DIR"

# Paths
PYTHON_SCRIPT="$SCRIPT_DIR/../hypersim/rendering.py"
INPUT_ROOT="/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"
PLANE_ROOT="/cluster/scratch/ayavuz/dataset/hypersim_mesh_ours"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
METADATA_CSV="$SCRIPT_DIR/../../shared/datasets/metadata_camera_parameters.csv"
FRAME_SKIP=1

# Test scenes
SCENES="ai_001_001 ai_006_004 ai_013_009 ai_021_002 ai_053_005 ai_008_005 ai_043_005 ai_048_001 ai_024_014 ai_036_001"

echo "[INFO] Submitting test rendering jobs (frame_skip=$FRAME_SKIP)"
echo "[INFO] Output: $OUTPUT_ROOT"
echo "[INFO] Logs:   $LOG_DIR"
echo ""

for SCENE in $SCENES; do
    echo "[SUBMIT] $SCENE"
    sbatch \
        --job-name="render_test_${SCENE}" \
        --time=4:00:00 \
        --cpus-per-task=4 \
        --mem-per-cpu=16G \
        --output="$LOG_DIR/render_test_${SCENE}_%j.out" \
        --error="$LOG_DIR/render_test_${SCENE}_%j.err" \
        --wrap="eval \"\$(conda shell.bash hook 2>/dev/null)\"; conda activate planamono; python $PYTHON_SCRIPT $SCENE --input_root $INPUT_ROOT --plane_root $PLANE_ROOT --output_root $OUTPUT_ROOT --frame_skip $FRAME_SKIP --metadata_csv $METADATA_CSV"
done

echo ""
echo "[INFO] Submitted ${#SCENES[@]} jobs. Monitor with: squeue -u \$USER"

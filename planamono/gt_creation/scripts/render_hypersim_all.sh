#!/bin/bash
# Render all Hypersim scenes (fixed remap_plane_ids)
# Submits 15 parallel SLURM jobs, each rendering ~30 scenes.
# Skips scenes without planes.ply in PLANE_ROOT.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/hypersim_render_all"
SPLIT_DIR="$SCRIPT_DIR/splits_render_tmp"
mkdir -p "$LOG_DIR" "$SPLIT_DIR"

# Paths
PYTHON_SCRIPT="$(realpath "$SCRIPT_DIR/../hypersim/rendering.py")"
INPUT_ROOT="/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"
PLANE_ROOT="/cluster/scratch/ayavuz/dataset/hypersim_mesh_ours"
OUTPUT_ROOT="/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
METADATA_CSV="$(realpath "$SCRIPT_DIR/../../shared/datasets/metadata_camera_parameters.csv")"
FRAME_SKIP=1
NUM_JOBS=15

# Scene list: all scenes from train + val + test
ALL_SCENES="$(realpath "$SCRIPT_DIR/../../splits/hypersim/all_scenes.txt")"

# Filter to scenes that have planes.ply
VALID_SCENES="$SPLIT_DIR/valid_scenes.txt"
> "$VALID_SCENES"
while IFS= read -r scene_id; do
    [[ -z "$scene_id" ]] && continue
    if [[ -f "$PLANE_ROOT/$scene_id/planes.ply" ]]; then
        echo "$scene_id" >> "$VALID_SCENES"
    fi
done < "$ALL_SCENES"

TOTAL=$(wc -l < "$VALID_SCENES")
PER_JOB=$(( (TOTAL + NUM_JOBS - 1) / NUM_JOBS ))

echo "[INFO] All scenes:   $(wc -l < "$ALL_SCENES")"
echo "[INFO] Valid scenes:  $TOTAL (have planes.ply)"
echo "[INFO] Skipped:       $(($(wc -l < "$ALL_SCENES") - TOTAL)) (missing planes.ply)"
echo "[INFO] Jobs:          $NUM_JOBS (~$PER_JOB scenes each)"
echo "[INFO] Frame skip:    $FRAME_SKIP"
echo "[INFO] Output:        $OUTPUT_ROOT"
echo "[INFO] Logs:          $LOG_DIR"
echo ""

# Split into NUM_JOBS files
split -l "$PER_JOB" "$VALID_SCENES" "$SPLIT_DIR/chunk_"

# Submit jobs
i=0
for chunk in "$SPLIT_DIR"/chunk_*; do
    i=$((i + 1))
    count=$(wc -l < "$chunk")
    echo "[SUBMIT] Job $i: $count scenes ($(head -1 "$chunk") .. $(tail -1 "$chunk"))"

    sbatch \
        --job-name="render_all_${i}" \
        --time=24:00:00 \
        --cpus-per-task=4 \
        --mem-per-cpu=16G \
        --output="$LOG_DIR/render_${i}_%j.out" \
        --error="$LOG_DIR/render_${i}_%j.err" \
        --wrap="$(cat <<WRAP
eval "\$(conda shell.bash hook 2>/dev/null)"
conda activate planamono
while IFS= read -r scene_id; do
    [ -z "\$scene_id" ] && continue
    echo "======== Rendering: \$scene_id ========"
    python $PYTHON_SCRIPT \$scene_id \
        --input_root $INPUT_ROOT \
        --plane_root $PLANE_ROOT \
        --output_root $OUTPUT_ROOT \
        --frame_skip $FRAME_SKIP \
        --metadata_csv $METADATA_CSV
done < $chunk
WRAP
)"
done

echo ""
echo "[INFO] Submitted $i jobs. Monitor with: squeue -u \$USER"
echo "[INFO] Split files saved in: $SPLIT_DIR"

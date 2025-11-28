#!/bin/bash

# Root folder containing split directories like split_0/, split_1/, ...
SPLIT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/splits/scannetpp/scene_splits"

# SLURM job template (will submit this for each scene list)
# JOB_SCRIPT="run_processing.sh"  # replace with your actual SLURM script
# JOB_SCRIPT="run_raycastsem.sh"  # replace with your actual SLURM script

JOB_SCRIPT="run_render_depth.sh"
# JOB_SCRIPT="run_raycast_plane.sh"

# Make logs directory if not exists
mkdir -p logs_depth

# Loop through each split
for split_dir in "$SPLIT_ROOT"/split_*; do
    if [[ -d "$split_dir" ]]; then
        # Extract index from split_#
        idx=$(basename "$split_dir" | cut -d'_' -f2)
        scene_list="$split_dir/scene_list_${idx}.txt"

        if [[ -f "$scene_list" ]]; then
            echo "[INFO] Submitting job for split_$idx"

            sbatch \
                --job-name=depth_split_${idx} \
                --output=logs_depth/split_${idx}.out \
                --error=logs_depth/split_${idx}.err \
                "$JOB_SCRIPT" "$scene_list"
        else
            echo "[WARN] Scene list not found: $scene_list"
        fi
    fi
done
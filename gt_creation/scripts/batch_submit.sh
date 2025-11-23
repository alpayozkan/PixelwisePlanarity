#!/bin/bash
# Batch Submission Script for SLURM Clusters
# Submits jobs for multiple scene splits

# Configuration
SPLIT_ROOT="${1:-scene_splits}"
JOB_SCRIPT="${2:-scannetpp_plane_extraction.sh}"
DATASET="${3:-scannetpp}"  # scannetpp or hypersim

# Validate inputs
if [[ ! -d "$SPLIT_ROOT" ]]; then
    echo "[ERROR] Split root directory not found: $SPLIT_ROOT"
    echo "Usage: $0 <split_root> <job_script> [dataset]"
    exit 1
fi

if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "[ERROR] Job script not found: $JOB_SCRIPT"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "[INFO] Batch submission for $DATASET"
echo "[INFO] Split root: $SPLIT_ROOT"
echo "[INFO] Job script: $JOB_SCRIPT"

# Loop through each split directory
for split_dir in "$SPLIT_ROOT"/split_*; do
    if [[ -d "$split_dir" ]]; then
        # Extract split index
        idx=$(basename "$split_dir" | cut -d'_' -f2)
        scene_list="$split_dir/scene_list_${idx}.txt"

        if [[ -f "$scene_list" ]]; then
            echo "================================================================"
            echo "[INFO] Submitting job for split_$idx"

            # Submit to SLURM
            sbatch \
                --job-name="${DATASET}_split_${idx}" \
                --output="logs/split_${idx}.out" \
                --error="logs/split_${idx}.err" \
                "$JOB_SCRIPT" "$scene_list"

            echo "[INFO] Job submitted: split_${idx}"
        else
            echo "[WARN] Scene list not found: $scene_list"
        fi
    fi
done

echo "================================================================"
echo "[INFO] Batch submission completed."
echo "[INFO] Monitor jobs with: squeue -u \$USER"
echo "[INFO] Check logs in: logs/"

#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1

# Optional: GPU request
## #SBATCH --job-name=planercnn_dataset
## #SBATCH --output=planercnn_dataset_%j.log
## #SBATCH --error=planercnn_dataset_%j.log
## #SBATCH --gres=gpu:1

# Input argument: scene list file
SCENE_LIST="$1"

if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Scene list file not found: $SCENE_LIST"
    exit 1
fi

echo "[INFO] Starting Rendering on: $(hostname)"
echo "[INFO] Scene list file: $SCENE_LIST"


cd /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/gt_creation/scannetpp

while IFS= read -r scene_id; do
    [[ -z "$scene_id" ]] && continue  # Skip empty lines

    echo "[INFO] Running rendering for scene: $scene_id"
    # python render_scene_depth.py "$scene_id"
    # python render_scene_depth_h5.py "$scene_id"
    python render_scene_depth.py "$scene_id"
    
done < "$SCENE_LIST"

echo "[INFO] Job completed."
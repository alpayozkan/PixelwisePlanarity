#!/bin/bash
# ScanNet++ Plane Extraction Script
# Extracts planes from semantic meshes for a list of scenes

# SLURM options (uncomment for cluster use)
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/scannetpp_extract_%j.out
#SBATCH --error=logs/scannetpp_extract_%j.err

# Get script directory (works from any location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration
SCENE_LIST="${1:-scene_list.txt}"
CONFIG="${2:-../configs/scannetpp_default.yml}"
INPUT_ROOT="${3:-/path/to/scannetpp/data}"
OUTPUT_ROOT="${4:-/path/to/output}"

# Validate inputs
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Scene list file not found: $SCENE_LIST"
    echo "Usage: $0 <scene_list.txt> [config.yml] [input_root] [output_root]"
    exit 1
fi

echo "[INFO] Starting plane extraction on: $(hostname)"
echo "[INFO] Scene list: $SCENE_LIST"
echo "[INFO] Config: $CONFIG"
echo "[INFO] Input root: $INPUT_ROOT"
echo "[INFO] Output root: $OUTPUT_ROOT"

# Process each scene
while IFS= read -r scene_id; do
    [[ -z "$scene_id" ]] && continue  # Skip empty lines
    [[ "$scene_id" =~ ^#.*$ ]] && continue  # Skip comments

    echo "================================================================"
    echo "[INFO] Processing scene: $scene_id"
    python "$SCRIPT_DIR/../scannetpp/scene_runner.py" "$scene_id" \
        --config "$SCRIPT_DIR/$CONFIG" \
        --input_root "$INPUT_ROOT" \
        --output_root "$OUTPUT_ROOT"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Completed: $scene_id"
    else
        echo "[ERROR] Failed: $scene_id"
    fi
done < "$SCENE_LIST"

echo "================================================================"
echo "[INFO] Plane extraction job completed."

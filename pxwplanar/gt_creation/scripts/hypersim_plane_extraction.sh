#!/bin/bash
# Hypersim Plane Extraction Script
# Extracts planes from Hypersim scenes

# SLURM options (uncomment for cluster use)
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/hypersim_extract_%j.out
#SBATCH --error=logs/hypersim_extract_%j.err

# Get script directory (works from any location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration
SCENE_LIST="${1:-scene_list.txt}"
CONFIG="${2:-../configs/hypersim_default.yml}"
INPUT_ROOT="${3:-}"
OUTPUT_ROOT="${4:-}"

# Resolve a relative config against the script directory (absolute paths pass through)
[[ "$CONFIG" = /* ]] || CONFIG="$SCRIPT_DIR/$CONFIG"

# Roots are optional — when omitted the runner uses input_root/output_root from the config
ROOT_ARGS=()
[[ -n "$INPUT_ROOT" ]] && ROOT_ARGS+=(--input_root "$INPUT_ROOT")
[[ -n "$OUTPUT_ROOT" ]] && ROOT_ARGS+=(--output_root "$OUTPUT_ROOT")

# Validate inputs
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "[ERROR] Scene list file not found: $SCENE_LIST"
    echo "Usage: $0 <scene_list.txt> [config.yml] [input_root] [output_root]"
    exit 1
fi

echo "[INFO] Starting Hypersim plane extraction on: $(hostname)"
echo "[INFO] Scene list: $SCENE_LIST"
echo "[INFO] Config: $CONFIG"

# Process each scene
while IFS= read -r scene_id; do
    [[ -z "$scene_id" ]] && continue
    [[ "$scene_id" =~ ^#.*$ ]] && continue

    echo "================================================================"
    echo "[INFO] Processing scene: $scene_id"
    python "$SCRIPT_DIR/../hypersim/scene_runner.py" "$scene_id" \
        --config "$CONFIG" "${ROOT_ARGS[@]}"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Completed: $scene_id"
    else
        echo "[ERROR] Failed: $scene_id"
    fi
done < "$SCENE_LIST"

echo "================================================================"
echo "[INFO] Hypersim plane extraction completed."

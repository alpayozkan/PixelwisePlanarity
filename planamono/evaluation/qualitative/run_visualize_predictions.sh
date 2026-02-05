#!/bin/bash
# Wrapper script for visualize_predictions_pdf.py
# Examples:
#   ./run_visualize_predictions.sh 10              # 10 random samples
#   ./run_visualize_predictions.sh 5 3             # 5 samples from 3 scenes
#   ./run_visualize_predictions.sh 10 scene_id     # 10 samples from specific scene

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PRED_ROOT="/cluster/scratch/aoezkan/planeseg/scannetpp/inference/moge_mixed_bce_h5"

# Activate conda environment
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

# Parse arguments
N_SAMPLES="${1:-10}"
ARG2="${2}"

if [[ "$ARG2" =~ ^[0-9]+$ ]]; then
    # Second argument is a number -> n_scenes
    python "$SCRIPT_DIR/visualize_predictions_pdf.py" \
        --pred-root "$PRED_ROOT" \
        --n-samples "$N_SAMPLES" \
        --n-scenes "$ARG2"
elif [[ -n "$ARG2" ]]; then
    # Second argument is a string -> scene ID
    python "$SCRIPT_DIR/visualize_predictions_pdf.py" \
        --pred-root "$PRED_ROOT" \
        --scene "$ARG2" \
        --n-samples "$N_SAMPLES"
else
    # Only n_samples specified
    python "$SCRIPT_DIR/visualize_predictions_pdf.py" \
        --pred-root "$PRED_ROOT" \
        --n-samples "$N_SAMPLES"
fi

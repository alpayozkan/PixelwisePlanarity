#!/bin/bash
# Resubmit synthia eval jobs for v5_relative methods (timed out at 12h limit).
# Run from a login node: bash submit_synthia_eval_rerun.sh

set -euo pipefail

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/v5_relative_all_datasets"
mkdir -p "$LOG_DIR"

for METHOD in moge_hires_ep3_v5_relative_seg moge_hires_ep3_v5origparams_relative_seg; do
    JID=$(/cluster/slurm/apps/bin/sbatch --parsable \
        --job-name="eval_${METHOD}_synthia_rerun" \
        --time=36:00:00 \
        --cpus-per-task=16 \
        --mem-per-cpu=4G \
        --output="$LOG_DIR/eval_${METHOD}_synthia_rerun_%j.out" \
        --error="$LOG_DIR/eval_${METHOD}_synthia_rerun_%j.err" \
        --wrap="bash -c '
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono
python $PROJECT_ROOT/evaluation/quantitative/evaluate_synthia_all_baselines.py --methods $METHOD
'")
    echo "Submitted $METHOD → job $JID"
done

echo ""
echo "After jobs complete, run to add synthia to the unified tables:"
echo "  cd /cluster/home/aoezkan/planeseg/PixelwisePlanarity"
echo "  python planamono/evaluation/quantitative/create_unified_tables.py --datasets synthia"

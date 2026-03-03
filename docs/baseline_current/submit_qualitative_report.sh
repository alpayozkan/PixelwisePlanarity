#!/bin/bash
#SBATCH --job-name=qual_report
#SBATCH --output=logs/qual_report_%j.log
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=0

conda activate planamono
cd /cluster/home/aoezkan/planeseg/PixelwisePlanarity

# Regenerate hypersim only (fix: use backproject_mcam instead of pinhole)
# Reuse existing scannetpp + synthia + vkitti2 PNGs
# Combine all 4 datasets into project_results_qual.pdf/png
python docs/baseline_current/generate_qualitative_report.py \
    --n-samples 5 --pdf --png \
    --datasets all \
    --regenerate hypersim

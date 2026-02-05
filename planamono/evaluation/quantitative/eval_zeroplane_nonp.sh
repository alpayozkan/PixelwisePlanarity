#!/bin/bash
# Evaluate ZeroPlane mixed models with non-planar label handling

# Optional SLURM directives (uncomment if submitting via sbatch)
#SBATCH --job-name=eval_zeroplane_nonp
#SBATCH --output=logs/eval_zeroplane_nonp_%j.out
#SBATCH --error=logs/eval_zeroplane_nonp_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=0

cd /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative

echo "Starting ZeroPlane nonp evaluation at $(date)"
echo "Methods: zeroplane_mixed, zeroplane_mixed_dust3r"

python evaluate_all_baselines_nonp.py --methods zeroplane_mixed zeroplane_mixed_dust3r

echo "Evaluation complete at $(date)"
echo ""
echo "Results saved to:"
echo "  - /cluster/scratch/aoezkan/planeseg/scannetpp/eval_v4_nonp/zeroplane_mixed/"
echo "  - /cluster/scratch/aoezkan/planeseg/scannetpp/eval_v4_nonp/zeroplane_mixed_dust3r/"

#!/bin/bash
# Submit MoGe HiRes ep2 (v6 seg) inference + evaluation pipeline:
#   VKITTI2  infer ──→ VKITTI2  eval
#   Synthia  infer ──→ Synthia  eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative"

mkdir -p /cluster/scratch/aoezkan/planeseg/logs/inference
mkdir -p /cluster/scratch/aoezkan/planeseg/logs/eval_vkitti2
mkdir -p /cluster/scratch/aoezkan/planeseg/logs/eval_synthia

echo "============================================================"
echo "Submitting MoGe HiRes ep2 (v6 seg): VKITTI2 + Synthia"
echo "============================================================"

# ── 1. Submit VKITTI2 inference (GPU) ──
JOB_VK_INFER=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_ep2_v6_vkitti2.sh")
echo "Submitted VKITTI2  inference: $JOB_VK_INFER"

# ── 2. Submit Synthia inference (GPU) ──
JOB_SYN_INFER=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_ep2_v6_synthia.sh")
echo "Submitted Synthia  inference: $JOB_SYN_INFER"

# ── 3. Submit VKITTI2 evaluation (CPU, depends on inference) ──
JOB_VK_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_VK_INFER <<EVAL_VK
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/eval_vkitti2/eval_moge_hires_ep2_v6_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/eval_vkitti2/eval_moge_hires_ep2_v6_%j.err

set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "VKITTI2 evaluation (moge_hires_ep2_v6)"
python $EVAL_DIR/evaluate_vkitti2_all_baselines.py --methods moge_hires_ep2_v6 --split test

echo "[DONE] VKITTI2 eval"
EVAL_VK
)
echo "Submitted VKITTI2  eval:      $JOB_VK_EVAL (after $JOB_VK_INFER)"

# ── 4. Submit Synthia evaluation (CPU, depends on inference) ──
JOB_SYN_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_SYN_INFER <<EVAL_SYN
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/eval_synthia/eval_moge_hires_ep2_v6_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/eval_synthia/eval_moge_hires_ep2_v6_%j.err

set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Synthia evaluation (moge_hires_ep2_v6)"
python $EVAL_DIR/evaluate_synthia_all_baselines.py --methods moge_hires_ep2_v6 --split test

echo "[DONE] Synthia eval"
EVAL_SYN
)
echo "Submitted Synthia  eval:      $JOB_SYN_EVAL (after $JOB_SYN_INFER)"

echo ""
echo "Pipeline submitted:"
echo "  VKITTI2  infer ($JOB_VK_INFER)  → eval ($JOB_VK_EVAL)"
echo "  Synthia  infer ($JOB_SYN_INFER) → eval ($JOB_SYN_EVAL)"
echo ""
echo "Monitor: squeue -u \$USER"

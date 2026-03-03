#!/bin/bash
# Submit MoGe HiRes ep1 + ep2 inference + evaluation pipeline:
#   ScanNet++ infer ep1 ─┐
#   ScanNet++ infer ep2 ─┼──→ ScanNet++ eval (both methods)
#   Hypersim  infer ep1 ─┐
#   Hypersim  infer ep2 ─┼──→ Hypersim  eval (both methods)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/inference"
mkdir -p "$LOG_DIR"

# ── 1. Submit inference jobs (GPU) ──
JOB_SNP_EP1=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_ep1.sh")
JOB_SNP_EP2=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_ep2.sh")
JOB_HYP_EP1=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_ep1_hypersim.sh")
JOB_HYP_EP2=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_ep2_hypersim.sh")

echo "Submitted ScanNet++ infer ep1: $JOB_SNP_EP1"
echo "Submitted ScanNet++ infer ep2: $JOB_SNP_EP2"
echo "Submitted Hypersim  infer ep1: $JOB_HYP_EP1"
echo "Submitted Hypersim  infer ep2: $JOB_HYP_EP2"

# ── 2. Submit ScanNet++ evaluation after BOTH ScanNet++ inferences finish ──
JOB_SNP_EVAL=$(sbatch --parsable --dependency=afterok:${JOB_SNP_EP1}:${JOB_SNP_EP2} <<'EVAL_SCANNETPP'
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_all_scannetpp_eval_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_all_scannetpp_eval_%j.err

set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "ScanNet++ evaluation (moge_hires_ep1 + moge_hires_ep2)"
python /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative/evaluate_all_baselines.py \
    --methods moge_hires_ep1 moge_hires_ep2

echo "[DONE] ScanNet++ eval"
EVAL_SCANNETPP
)
echo "Submitted ScanNet++ eval:      $JOB_SNP_EVAL (after $JOB_SNP_EP1, $JOB_SNP_EP2)"

# ── 3. Submit Hypersim evaluation after BOTH Hypersim inferences finish ──
JOB_HYP_EVAL=$(sbatch --parsable --dependency=afterok:${JOB_HYP_EP1}:${JOB_HYP_EP2} <<'EVAL_HYPERSIM'
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_all_hypersim_eval_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_all_hypersim_eval_%j.err

set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Hypersim evaluation (moge_hires_ep1 + moge_hires_ep2)"
python /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative/evaluate_hypersim_all_baselines.py \
    --methods moge_hires_ep1 moge_hires_ep2

echo "[DONE] Hypersim eval"
EVAL_HYPERSIM
)
echo "Submitted Hypersim  eval:      $JOB_HYP_EVAL (after $JOB_HYP_EP1, $JOB_HYP_EP2)"

echo ""
echo "Pipeline submitted:"
echo "  ScanNet++ infer ep1 ($JOB_SNP_EP1) ─┐"
echo "  ScanNet++ infer ep2 ($JOB_SNP_EP2) ─┼→ eval ($JOB_SNP_EVAL)"
echo "  Hypersim  infer ep1 ($JOB_HYP_EP1) ─┐"
echo "  Hypersim  infer ep2 ($JOB_HYP_EP2) ─┼→ eval ($JOB_HYP_EVAL)"
echo ""
echo "Monitor: squeue -u \$USER"

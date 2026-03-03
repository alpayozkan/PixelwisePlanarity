#!/bin/bash
# Submit MoGe HiRes ep3 inference + evaluation pipeline:
#   ScanNet++ infer ──→ ScanNet++ eval
#   Hypersim  infer ──→ Hypersim  eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/inference"
mkdir -p "$LOG_DIR"

# ── 1. Submit ScanNet++ inference (GPU) ──
JOB_SCANNETPP=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_ep3.sh")
echo "Submitted ScanNet++ inference: $JOB_SCANNETPP"

# ── 2. Submit Hypersim inference (GPU) ──
JOB_HYPERSIM=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_ep3_hypersim.sh")
echo "Submitted Hypersim  inference: $JOB_HYPERSIM"

# ── 3. Submit ScanNet++ evaluation (CPU+GPU, depends on inference) ──
JOB_SCANNETPP_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_SCANNETPP <<'EVAL_SCANNETPP'
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_ep3_scannetpp_eval_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_ep3_scannetpp_eval_%j.err

set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "ScanNet++ evaluation (moge_hires_ep3)"
python /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative/evaluate_all_baselines.py \
    --methods moge_hires_ep3

echo "[DONE] ScanNet++ eval"
EVAL_SCANNETPP
)
echo "Submitted ScanNet++ eval:      $JOB_SCANNETPP_EVAL (after $JOB_SCANNETPP)"

# ── 4. Submit Hypersim evaluation (CPU+GPU, depends on inference) ──
JOB_HYPERSIM_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_HYPERSIM <<'EVAL_HYPERSIM'
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_ep3_hypersim_eval_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_ep3_hypersim_eval_%j.err

set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Hypersim evaluation (moge_hires_ep3)"
python /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative/evaluate_hypersim_all_baselines.py \
    --methods moge_hires_ep3

echo "[DONE] Hypersim eval"
EVAL_HYPERSIM
)
echo "Submitted Hypersim  eval:      $JOB_HYPERSIM_EVAL (after $JOB_HYPERSIM)"

echo ""
echo "Pipeline submitted:"
echo "  ScanNet++ infer ($JOB_SCANNETPP) → eval ($JOB_SCANNETPP_EVAL)"
echo "  Hypersim  infer ($JOB_HYPERSIM)  → eval ($JOB_HYPERSIM_EVAL)"
echo ""
echo "Monitor: squeue -u \$USER"

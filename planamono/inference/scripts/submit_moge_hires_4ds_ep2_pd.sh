#!/bin/bash
# Submit MoGe HiRes 4DS ep2 inference + evaluation on Parallel Domain:
#   PD infer (GPU) ──→ PD moge eval (CPU)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/inference"
mkdir -p "$LOG_DIR"

METHOD="moge_hires_4ds_ep2"

echo "============================================================"
echo "Submitting $METHOD inference + evaluation on Parallel Domain"
echo "============================================================"
echo ""

# ── 1. Submit PD inference (GPU) ──
JOB_PD=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_4ds_ep2_pd.sh")
echo "Submitted PD inference:        $JOB_PD"

# ── 2. Submit PD moge evaluation (CPU, depends on inference) ──
JOB_PD_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_PD <<'EVAL_PD'
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_4ds_ep2_pd_eval_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_4ds_ep2_pd_eval_%j.err

set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Parallel Domain evaluation (moge_hires_4ds_ep2)"
python /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative/evaluate_pd_all_baselines.py \
    --methods moge_hires_4ds_ep2 --split val

echo "[DONE] PD moge eval"
EVAL_PD
)
echo "Submitted PD moge eval:        $JOB_PD_EVAL (after $JOB_PD)"

echo ""
echo "============================================================"
echo "Pipeline submitted:"
echo "  PD infer ($JOB_PD) -> moge eval ($JOB_PD_EVAL)"
echo ""
echo "Monitor: squeue -u $USER"
echo "============================================================"

#!/bin/bash
# Submit MoGe HiRes 4DS ep2 inference + evaluation pipeline on all 5 datasets:
#   ScanNet++ infer ──→ ScanNet++ eval
#   Hypersim  infer ──→ Hypersim  eval
#   VKITTI2   infer ──→ VKITTI2   eval
#   Synthia   infer ──→ Synthia   eval
#   PD        infer ──→ PD        eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/inference"
mkdir -p "$LOG_DIR"

METHOD="moge_hires_4ds_ep2"

echo "============================================================"
echo "Submitting $METHOD inference + evaluation on ALL 5 datasets"
echo "============================================================"
echo ""

# ── 1. Submit ScanNet++ inference (GPU) ──
JOB_SCANNETPP=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_4ds_ep2.sh")
echo "Submitted ScanNet++ inference: $JOB_SCANNETPP"

# ── 2. Submit Hypersim inference (GPU) ──
JOB_HYPERSIM=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_4ds_ep2_hypersim.sh")
echo "Submitted Hypersim  inference: $JOB_HYPERSIM"

# ── 3. Submit VKITTI2 inference (GPU) ──
JOB_VKITTI2=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_4ds_ep2_vkitti2.sh")
echo "Submitted VKITTI2   inference: $JOB_VKITTI2"

# ── 4. Submit Synthia inference (GPU) ──
JOB_SYNTHIA=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_4ds_ep2_synthia.sh")
echo "Submitted Synthia   inference: $JOB_SYNTHIA"

echo ""

# ── 5. Submit ScanNet++ evaluation (CPU, depends on inference) ──
JOB_SCANNETPP_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_SCANNETPP <<'EVAL_SCANNETPP'
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_4ds_ep2_scannetpp_eval_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_4ds_ep2_scannetpp_eval_%j.err

set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "ScanNet++ evaluation (moge_hires_4ds_ep2)"
python /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative/evaluate_all_baselines.py \
    --methods moge_hires_4ds_ep2

echo "[DONE] ScanNet++ eval"
EVAL_SCANNETPP
)
echo "Submitted ScanNet++ eval:      $JOB_SCANNETPP_EVAL (after $JOB_SCANNETPP)"

# ── 6. Submit Hypersim evaluation (CPU, depends on inference) ──
JOB_HYPERSIM_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_HYPERSIM <<'EVAL_HYPERSIM'
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_4ds_ep2_hypersim_eval_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_4ds_ep2_hypersim_eval_%j.err

set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Hypersim evaluation (moge_hires_4ds_ep2)"
python /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative/evaluate_hypersim_all_baselines.py \
    --methods moge_hires_4ds_ep2

echo "[DONE] Hypersim eval"
EVAL_HYPERSIM
)
echo "Submitted Hypersim  eval:      $JOB_HYPERSIM_EVAL (after $JOB_HYPERSIM)"

# ── 7. Submit VKITTI2 evaluation (CPU, depends on inference) ──
JOB_VKITTI2_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_VKITTI2 <<'EVAL_VKITTI2'
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_4ds_ep2_vkitti2_eval_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_4ds_ep2_vkitti2_eval_%j.err

set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "VKITTI2 evaluation (moge_hires_4ds_ep2)"
python /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative/evaluate_vkitti2_all_baselines.py \
    --methods moge_hires_4ds_ep2 --split test

echo "[DONE] VKITTI2 eval"
EVAL_VKITTI2
)
echo "Submitted VKITTI2   eval:      $JOB_VKITTI2_EVAL (after $JOB_VKITTI2)"

# ── 8. Submit Synthia evaluation (CPU, depends on inference) ──
JOB_SYNTHIA_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_SYNTHIA <<'EVAL_SYNTHIA'
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_4ds_ep2_synthia_eval_%j.out
#SBATCH --error=/cluster/scratch/aoezkan/planeseg/logs/inference/moge_hires_4ds_ep2_synthia_eval_%j.err

set -euo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planamono

echo "Synthia evaluation (moge_hires_4ds_ep2)"
python /cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative/evaluate_synthia_all_baselines.py \
    --methods moge_hires_4ds_ep2 --split test

echo "[DONE] Synthia eval"
EVAL_SYNTHIA
)
echo "Submitted Synthia   eval:      $JOB_SYNTHIA_EVAL (after $JOB_SYNTHIA)"

# ── 9. Submit Parallel Domain inference (GPU) ──
JOB_PD=$(sbatch --parsable "$SCRIPT_DIR/run_moge_hires_4ds_ep2_pd.sh")
echo "Submitted PD        inference: $JOB_PD"

# ── 10. Submit Parallel Domain evaluation (CPU, depends on inference) ──
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

echo "[DONE] PD eval"
EVAL_PD
)
echo "Submitted PD        eval:      $JOB_PD_EVAL (after $JOB_PD)"

echo ""
echo "============================================================"
echo "Pipeline submitted:"
echo "  ScanNet++ infer ($JOB_SCANNETPP) -> eval ($JOB_SCANNETPP_EVAL)"
echo "  Hypersim  infer ($JOB_HYPERSIM)  -> eval ($JOB_HYPERSIM_EVAL)"
echo "  VKITTI2   infer ($JOB_VKITTI2)   -> eval ($JOB_VKITTI2_EVAL)"
echo "  Synthia   infer ($JOB_SYNTHIA)   -> eval ($JOB_SYNTHIA_EVAL)"
echo "  PD        infer ($JOB_PD)        -> eval ($JOB_PD_EVAL)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "============================================================"

#!/bin/bash
# Submit all 4 jobs with dependencies:
#   ScanNet++ infer ──→ ScanNet++ eval
#   Hypersim  infer ──→ Hypersim  eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Submit inference jobs (GPU)
JOB_SCANNETPP=$(sbatch --parsable "$SCRIPT_DIR/run_moge_mixed_bce_476644.sh")
JOB_HYPERSIM=$(sbatch --parsable "$SCRIPT_DIR/run_moge_mixed_bce_476644_hypersim.sh")

echo "Submitted ScanNet++ inference: $JOB_SCANNETPP"
echo "Submitted Hypersim  inference: $JOB_HYPERSIM"

# Submit eval jobs (CPU-only, depend on inference)
JOB_SCANNETPP_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_SCANNETPP "$SCRIPT_DIR/eval_moge_mixed_bce_476644.sh")
JOB_HYPERSIM_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_HYPERSIM "$SCRIPT_DIR/eval_moge_mixed_bce_476644_hypersim.sh")

echo "Submitted ScanNet++ eval:      $JOB_SCANNETPP_EVAL (after $JOB_SCANNETPP)"
echo "Submitted Hypersim  eval:      $JOB_HYPERSIM_EVAL (after $JOB_HYPERSIM)"

echo ""
echo "Monitor: squeue -u \$USER"

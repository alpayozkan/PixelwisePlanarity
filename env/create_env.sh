#!/bin/bash
# Create the conda environment for this repository and install it as an
# editable library.
#
# Usage:
#   bash env/create_env.sh [env_name]
#
# Steps:
#   1. conda env create from env/environment.yml (name defaults to pxwplanar)
#   2. pip install -e <repo root>  (single pxwplanar package: shared, evaluation,
#      inference, gt_creation, paths — importable from anywhere)
#   3. git submodule update --init (MoGe fork; skipped with a warning if the
#      repository is not accessible)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="${1:-pxwplanar}"

# Locate conda
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    echo "[ERROR] conda not found" >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[1/3] Creating conda env '$ENV_NAME' from $SCRIPT_DIR/environment.yml ..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "      env '$ENV_NAME' already exists — skipping creation"
else
    conda env create -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.yml"
fi

echo "[2/3] Installing repo as editable library ..."
conda run -n "$ENV_NAME" pip install -e "$REPO_ROOT" --no-deps

echo "[3/3] Initializing MoGe submodule ..."
if git -C "$REPO_ROOT" submodule update --init MoGe; then
    echo "      MoGe submodule initialized"
else
    echo "[WARN] Could not initialize MoGe submodule (see the git error above)."
    echo "       Inference and training will not run"
    echo "       until MoGe/ is populated. Evaluation (evaluate_all_baselines.py)"
    echo "       works without it."
fi

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
conda run -n "$ENV_NAME" python -c "import pxwplanar.shared, pxwplanar.evaluation, pxwplanar.inference, pxwplanar.gt_creation, pxwplanar.paths; print('Editable install verified:', pxwplanar.__file__)"

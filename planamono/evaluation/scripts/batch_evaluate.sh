#!/bin/bash
# Batch Evaluation Script
# Evaluates multiple methods on the test set

# Configuration
SPLIT="${1:-test}"
OUTPUT_ROOT="${2:-./results}"

# Methods to evaluate
METHODS=("gt" "moge" "planercnn" "monoplane")

echo "[INFO] Batch evaluation for split: $SPLIT"
echo "[INFO] Methods: ${METHODS[@]}"

# Create logs directory
mkdir -p logs

# Evaluate each method
for method in "${METHODS[@]}"; do
    echo "================================================================"
    echo "[INFO] Evaluating method: $method"

    output_dir="${OUTPUT_ROOT}/${method}_${SPLIT}"
    mkdir -p "$output_dir"

    # Submit job (or run locally)
    if command -v sbatch &> /dev/null; then
        # SLURM cluster
        sbatch \
            --job-name="eval_${method}" \
            --output="logs/eval_${method}.out" \
            --error="logs/eval_${method}.err" \
            run_evaluation.sh "$method" "$SPLIT" "$output_dir"
        echo "[INFO] Job submitted: eval_${method}"
    else
        # Local execution
        ./run_evaluation.sh "$method" "$SPLIT" "$output_dir" > "logs/eval_${method}.out" 2>&1 &
        echo "[INFO] Running locally: eval_${method}"
    fi
done

echo "================================================================"
echo "[INFO] Batch evaluation jobs started"
echo "[INFO] Monitor with: tail -f logs/eval_*.out"

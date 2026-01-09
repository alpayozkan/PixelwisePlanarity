#!/bin/bash
# Run all fast evaluation scripts sequentially and save output to baseline_output.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="$SCRIPT_DIR/baseline_output.txt"

echo "Running all fast evaluation scripts..."
echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# Clear/create output file
> "$OUTPUT_FILE"

# List of scripts to run in order
SCRIPTS=(
    "evaluate_scannetpp_fast.py"
    "evaluate_scannetpp_gtseg_fast.py"
    "evaluate_scannetpp_gtplanarity_ourseg_fast.py"
    "evaluate_scannetpp_ourplanarity_gtseg_fast.py"
    "evaluate_scannetpp_zeroplane_fast.py"
)

# Run each script
for script in "${SCRIPTS[@]}"; do
    echo "========================================" | tee -a "$OUTPUT_FILE"
    echo "Running: $script" | tee -a "$OUTPUT_FILE"
    echo "Started at: $(date)" | tee -a "$OUTPUT_FILE"
    echo "========================================" | tee -a "$OUTPUT_FILE"

    python "$SCRIPT_DIR/$script" 2>&1 | tee -a "$OUTPUT_FILE"

    echo "" | tee -a "$OUTPUT_FILE"
    echo "Finished: $script at $(date)" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
done

# Aggregate results
echo "========================================" | tee -a "$OUTPUT_FILE"
echo "Aggregating results..." | tee -a "$OUTPUT_FILE"
echo "Started at: $(date)" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

python "$SCRIPT_DIR/agg_results_baselines.py" --output_dir "$SCRIPT_DIR" 2>&1 | tee -a "$OUTPUT_FILE"

echo "" | tee -a "$OUTPUT_FILE"
echo "Finished aggregation at: $(date)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "========================================" | tee -a "$OUTPUT_FILE"
echo "All scripts completed at: $(date)" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

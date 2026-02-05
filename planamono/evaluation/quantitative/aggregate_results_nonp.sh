#!/bin/bash
# Aggregate results from all baseline evaluations (run after all eval jobs complete)

SCRIPT_DIR="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/evaluation/quantitative"
OUTPUT_DIR="${SCRIPT_DIR}"
VERSION_TAG="v4_nonp"

echo "=========================================="
echo "Aggregating baseline results (${VERSION_TAG})"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

cd ${SCRIPT_DIR}

# Run aggregation only (no evaluation)
python evaluate_all_baselines_nonp.py --aggregate-only --output-dir "${OUTPUT_DIR}"

# Rename files to include version tag
if [ -f "${OUTPUT_DIR}/table_precision_recall_baselines.csv" ]; then
    mv "${OUTPUT_DIR}/table_precision_recall_baselines.csv" \
       "${OUTPUT_DIR}/table_precision_recall_baselines_${VERSION_TAG}.csv"
    echo "Renamed: table_precision_recall_baselines_${VERSION_TAG}.csv"
fi

if [ -f "${OUTPUT_DIR}/table_segmentation_baselines.csv" ]; then
    mv "${OUTPUT_DIR}/table_segmentation_baselines.csv" \
       "${OUTPUT_DIR}/table_segmentation_baselines_${VERSION_TAG}.csv"
    echo "Renamed: table_segmentation_baselines_${VERSION_TAG}.csv"
fi

if [ -f "${OUTPUT_DIR}/table_combined_baselines.csv" ]; then
    mv "${OUTPUT_DIR}/table_combined_baselines.csv" \
       "${OUTPUT_DIR}/table_combined_baselines_${VERSION_TAG}.csv"
    echo "Renamed: table_combined_baselines_${VERSION_TAG}.csv"
fi

echo ""
echo "=========================================="
echo "Aggregation complete!"
echo "Generated tables:"
echo "  - ${OUTPUT_DIR}/table_precision_recall_baselines_${VERSION_TAG}.csv"
echo "  - ${OUTPUT_DIR}/table_segmentation_baselines_${VERSION_TAG}.csv"
echo "  - ${OUTPUT_DIR}/table_combined_baselines_${VERSION_TAG}.csv"
echo "=========================================="

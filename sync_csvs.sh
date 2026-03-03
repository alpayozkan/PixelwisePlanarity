#!/usr/bin/env bash
# Sync all CSV and TEX files from planeseg scratch on Euler to local machine.
# Preserves directory hierarchy under a local target folder.
#
# Usage (run from your LOCAL machine):
#   ./sync_csvs.sh [local_dest]
#
# Default local destination: ./scratch_csvs/

set -euo pipefail

REMOTE="euler"
REMOTE_ROOT="/cluster/scratch/aoezkan/planeseg"
LOCAL_DEST="${1:-./scratch_csvs}"

echo "=== Scanning CSV and TEX files on ${REMOTE}:${REMOTE_ROOT} ==="

# Get file count and total size in one pass
STATS=$(ssh "$REMOTE" "find '${REMOTE_ROOT}' \( -name '*.csv' -o -name '*.tex' \) -type f -printf '%s %f\n'" 2>/dev/null)

if [ -z "$STATS" ]; then
    echo "No CSV/TEX files found under ${REMOTE}:${REMOTE_ROOT}"
    exit 0
fi

NUM_CSV=$(echo "$STATS" | grep -c '\.csv$' || true)
NUM_TEX=$(echo "$STATS" | grep -c '\.tex$' || true)
NUM_FILES=$((NUM_CSV + NUM_TEX))
TOTAL_BYTES=$(echo "$STATS" | awk '{s+=$1} END {print s}')
TOTAL_GB=$(echo "scale=3; ${TOTAL_BYTES} / 1073741824" | bc)

echo ""
echo "  CSV files:  ${NUM_CSV}"
echo "  TEX files:  ${NUM_TEX}"
echo "  Total:      ${NUM_FILES} files, ${TOTAL_GB} GB"
echo "  Remote:     ${REMOTE}:${REMOTE_ROOT}"
echo "  Local:      ${LOCAL_DEST}/"
echo ""
read -rp "Proceed with download? [y/N] " answer

if [[ ! "$answer" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "=== Syncing ==="

rsync -avz --relative \
    --include='*/' \
    --include='*.csv' \
    --include='*.tex' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_ROOT}/./" \
    "${LOCAL_DEST}/"

echo ""
echo "=== Done. Files saved to ${LOCAL_DEST}/ ==="

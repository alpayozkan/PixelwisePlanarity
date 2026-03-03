#!/bin/bash
# Test download speed from Apple's Hypersim servers.
# Downloads first 10MB of one scene zip to /dev/null (nothing saved).
# Usage: bash planamono/shared/datasets/test_download_speed.sh

URL="https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_001.zip"

echo "Testing download speed (10MB range request)..."
SPEED=$(curl -so /dev/null -w '%{speed_download}' -L -r 0-10485759 "$URL")
SPEED_MB=$(echo "$SPEED" | awk '{printf "%.1f", $1/1048576}')
echo "Speed: ${SPEED_MB} MB/s (${SPEED} bytes/sec)"

# Estimate total download time for ~177GB
TOTAL_GB=177
if [ "$(echo "$SPEED > 0" | bc)" -eq 1 ]; then
    HOURS=$(echo "$TOTAL_GB * 1024 / ($SPEED / 1048576) / 3600" | bc -l)
    printf "Estimated time for full dataset (~%dGB): %.1f hours\n" "$TOTAL_GB" "$HOURS"
fi

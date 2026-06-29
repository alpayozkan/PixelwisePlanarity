#!/bin/bash
# Copy the scannetpp dataset (~63G) from ayavuz scratch to aoezkan scratch
# Destination: /cluster/scratch/aoezkan/ayavuz_copy/dataset/scannetpp/
#
# Uses rsync so it's resumable — safe to re-run if interrupted; it will
# skip files that already match and only transfer what's missing/changed.

set -euo pipefail

SRC="/cluster/scratch/ayavuz/dataset/scannetpp/"
DST="/cluster/scratch/aoezkan/ayavuz_copy/dataset/scannetpp/"

mkdir -p "$DST"

echo "Copying:"
echo "  from: $SRC"
echo "  to:   $DST"
echo

rsync -avh --info=progress2 --partial \
      --exclude='._*' --exclude='.DS_Store' \
      "$SRC" "$DST"

echo
echo "Done. Final size:"
du -sh "$DST"

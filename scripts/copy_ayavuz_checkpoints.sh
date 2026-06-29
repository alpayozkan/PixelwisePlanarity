#!/bin/bash
# Copy all checkpoint files (*.pt, *.pth) from ayavuz scratch to aoezkan scratch
# under ayavuz_copy/, preserving directory structure.

set -euo pipefail

SRC="/cluster/scratch/ayavuz"
DST="/cluster/scratch/aoezkan/ayavuz_copy"

mkdir -p "$DST"

# Find all checkpoint files, skipping macOS resource-fork sidecars (._*)
cd "$SRC"
find . -type f \( -name "*.pt" -o -name "*.pth" \) ! -name "._*" -print0 \
  | while IFS= read -r -d '' f; do
      rel="${f#./}"
      target="$DST/$rel"
      mkdir -p "$(dirname "$target")"
      echo "Copying: $rel"
      cp -p "$f" "$target"
    done

echo
echo "Done. Files copied to: $DST"
du -sh "$DST"

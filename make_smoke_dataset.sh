#!/usr/bin/env bash
# Build a tiny train/val/test dataset from sample_dataset/ for smoke tests.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT_DIR/sample_dataset"
DST="$ROOT_DIR/smoke_dataset"
rm -rf "$DST"
for split in train val test; do
    for cls in 0 1 2 3 4; do
        mkdir -p "$DST/$split/$cls"
        # smoke: reuse the 8 sample images per class in every split
        for img in "$SRC/$cls"/*; do
            ln -sf "$img" "$DST/$split/$cls/$(basename "$img")"
        done
    done
done
echo "✅ smoke_dataset ready (40 symlinks per split):"
find "$DST" -maxdepth 2 -type d | sort

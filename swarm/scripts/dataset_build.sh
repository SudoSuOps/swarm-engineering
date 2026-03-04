#!/usr/bin/env bash
# dataset_build.sh — Assemble and validate a dataset
# Usage: ./dataset_build.sh <source_dir> <output_dir> <dataset_name>
set -euo pipefail

SOURCE_DIR="${1:?Usage: dataset_build.sh <source_dir> <output_dir> <dataset_name>}"
OUTPUT_DIR="${2:?}"
DATASET_NAME="${3:?}"

echo "=== Swarm Dataset Build ==="
echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Name:   $DATASET_NAME"

mkdir -p "$OUTPUT_DIR"

python3 -m swarm.pipelines.dataset_factory \
    --sources "$SOURCE_DIR" \
    --output "$OUTPUT_DIR" \
    --name "$DATASET_NAME" \
    2>&1 | tee "$OUTPUT_DIR/build.log"

echo "=== Dataset build complete ==="
echo "Files:"
ls -lh "$OUTPUT_DIR/${DATASET_NAME}"_*.jsonl 2>/dev/null || echo "  (no output files found)"

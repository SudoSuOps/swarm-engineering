#!/usr/bin/env bash
# run_eval.sh — Run evaluation suite against a model
# Usage: ./run_eval.sh <model_path> <eval_data> <output_dir>
set -euo pipefail

MODEL_PATH="${1:?Usage: run_eval.sh <model_path> <eval_data> <output_dir>}"
EVAL_DATA="${2:?}"
OUTPUT_DIR="${3:?}"

echo "=== Swarm Eval ==="
echo "Model:  $MODEL_PATH"
echo "Data:   $EVAL_DATA"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

python3 -m swarm.pipelines.eval_runner \
    --model "$MODEL_PATH" \
    --eval-data "$EVAL_DATA" \
    --output "$OUTPUT_DIR/eval_results.jsonl" \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

echo "=== Eval complete ==="

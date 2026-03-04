#!/usr/bin/env bash
# run_training.sh — Launch a training run on swarmrails
# Usage: ./run_training.sh <model> <dataset> <output_dir> [gpu_id]
set -euo pipefail

MODEL="${1:?Usage: run_training.sh <model> <dataset> <output_dir> [gpu_id]}"
DATASET="${2:?}"
OUTPUT_DIR="${3:?}"
GPU_ID="${4:-1}"

echo "=== Swarm Training Launch ==="
echo "Model:      $MODEL"
echo "Dataset:    $DATASET"
echo "Output:     $OUTPUT_DIR"
echo "GPU:        $GPU_ID"

mkdir -p "$OUTPUT_DIR"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
WANDB_MODE=disabled \
python3 -u "$OUTPUT_DIR/train.py" \
    --base-model "$MODEL" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "=== Training complete ==="

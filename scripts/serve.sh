#!/usr/bin/env bash
# Serve SwarmSignal-9B Q4_K_M via llama-server
#
# Usage:
#   ./scripts/serve.sh                          # default: GPU 0, port 8090
#   ./scripts/serve.sh --gpu 1 --port 8091      # custom GPU and port
#   ./scripts/serve.sh --model /path/to/gguf     # custom model path

set -euo pipefail

MODEL="${MODEL:-swarmsignal-9b-q4km.gguf}"
PORT="${PORT:-8090}"
GPU="${GPU:-0}"
CTX="${CTX:-4096}"
NGL="${NGL:-99}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)  MODEL="$2"; shift 2 ;;
        --port)   PORT="$2"; shift 2 ;;
        --gpu)    GPU="$2"; shift 2 ;;
        --ctx)    CTX="$2"; shift 2 ;;
        *)        echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$MODEL" ]]; then
    echo "Model not found: $MODEL"
    echo "Download the Q4_K_M GGUF or set MODEL=/path/to/swarmsignal-9b-q4km.gguf"
    exit 1
fi

echo "SwarmSignal-9B Server"
echo "  Model: $MODEL"
echo "  GPU:   $GPU"
echo "  Port:  $PORT"
echo "  CTX:   $CTX"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    llama-server \
    -m "$MODEL" \
    -ngl "$NGL" \
    -c "$CTX" \
    --port "$PORT"

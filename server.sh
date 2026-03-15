#!/bin/bash
# Launch BitNet as a web server with chat UI and live stats
# Usage: ./server.sh [model_dir] [threads] [port]
#   ./server.sh                                    # default 2B model, 4 threads, port 8080
#   ./server.sh models/BitNet-b1.58-2B-4T 8 9090  # specify model, threads, port

MODEL_DIR="${1:-models/BitNet-b1.58-2B-4T}"
THREADS="${2:-4}"
PORT="${3:-8080}"
MODEL_FILE="$MODEL_DIR/ggml-model-i2_s.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Model not found: $MODEL_FILE"
    echo "Available models:"
    find models -name "*.gguf" 2>/dev/null
    exit 1
fi

export SDKROOT=$(xcrun --show-sdk-path)
export LIBRARY_PATH="$SDKROOT/usr/lib:${LIBRARY_PATH:-}"

echo "Starting BitNet server on http://localhost:$PORT"
echo "Model: $MODEL_FILE"
echo "Threads: $THREADS"
echo ""

exec ./build/bin/llama-server \
    -m "$MODEL_FILE" \
    -t "$THREADS" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --metrics \
    -c 4096

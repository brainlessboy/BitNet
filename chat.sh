#!/bin/bash
# Launch interactive chat with BitNet model
# Usage: ./chat.sh [model_dir] [threads]
#   ./chat.sh                                    # default 2B model, 4 threads
#   ./chat.sh models/BitNet-b1.58-2B-4T 8       # specify model and threads

MODEL_DIR="${1:-models/BitNet-b1.58-2B-4T}"
THREADS="${2:-4}"
MODEL_FILE="$MODEL_DIR/ggml-model-i2_s.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Model not found: $MODEL_FILE"
    echo "Available models:"
    find models -name "*.gguf" 2>/dev/null
    exit 1
fi

export SDKROOT=$(xcrun --show-sdk-path)
export LIBRARY_PATH="$SDKROOT/usr/lib:${LIBRARY_PATH:-}"

exec ./build/bin/llama-cli \
    -m "$MODEL_FILE" \
    -t "$THREADS" \
    -n 512 \
    -c 4096 \
    --temp 0.7 \
    --top-p 0.9 \
    --repeat-penalty 1.1 \
    -cnv \
    -p "You are a helpful, concise assistant."

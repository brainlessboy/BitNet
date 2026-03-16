# BitNet + BitDistill

Fork of [microsoft/BitNet](https://github.com/microsoft/BitNet) with a training pipeline to distill any full-precision LLM into a 1.58-bit ternary model. Based on the [BitNet Distillation](https://arxiv.org/abs/2510.13998) paper.

**What this does**: Takes a normal model (e.g. Qwen2.5-0.5B), distills it to 1.58-bit weights ({-1, 0, +1}), and runs it in BitNet's optimized C++ inference engine at ~100+ tokens/sec on CPU.

## Installation

**Requirements**: macOS Apple Silicon (or Linux), Miniconda, LLVM 18, CMake >= 3.22

```bash
# Install deps (macOS)
brew install miniconda llvm@18

# Clone
git clone --recursive https://github.com/brainlessboy/BitNet.git
cd BitNet

# Python environment
conda create -n bitnet-cpp python=3.9 -y
conda activate bitnet-cpp
pip install -r requirements.txt
pip install datasets accelerate

# Build inference engine (macOS needs SDK paths for Homebrew clang)
export SDKROOT=$(xcrun --show-sdk-path)
export LIBRARY_PATH="$SDKROOT/usr/lib:${LIBRARY_PATH:-}"

cmake -B build -DBITNET_ARM_TL1=OFF \
    -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@18/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@18/bin/clang++ \
    -DCMAKE_EXE_LINKER_FLAGS="-L$SDKROOT/usr/lib -F$SDKROOT/System/Library/Frameworks" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$SDKROOT/usr/lib -F$SDKROOT/System/Library/Frameworks"

cmake --build build --config Release -j8
```

## Distill a Model

Four steps: train, export, convert, chat.

### Step 1: Train

Distills Qwen2.5-0.5B (FP16) into a 1.58-bit student using BitLinear + SubLN + knowledge distillation.

```bash
conda activate bitnet-cpp

# Quick test (5 min)
python distill/distill.py --max_steps 50 --device mps --save_every 50

# Real training (~4 hours, recommended 2000 steps)
python distill/distill.py --max_steps 2000 --device mps --save_every 500
```

For machines with less memory (8GB), use CPU with shorter sequences:
```bash
python distill/distill.py --max_steps 2000 --device cpu --max_length 256 --batch_size 1 --accumulation_steps 32
```

### Step 2: Export

Converts the trained checkpoint to HuggingFace format with ternary-quantized weights.

```bash
python distill/export_bitnet.py \
    --checkpoint distill/checkpoints/final.pt \
    --output models/distilled-qwen-bitnet
```

### Step 3: Convert to GGUF + Quantize

```bash
# Convert to GGUF
python utils/convert-hf-to-gguf-bitnet.py models/distilled-qwen-bitnet/ --outtype f32

# Quantize (1.9 GB -> 224 MB)
./build/bin/llama-quantize \
    models/distilled-qwen-bitnet/ggml-model-f32.gguf \
    models/distilled-qwen-bitnet/ggml-model-i2_s.gguf I2_S 1
```

### Step 4: Chat

```bash
# Terminal chat
./chat.sh models/distilled-qwen-bitnet

# Web UI with live stats at http://localhost:8080
./server.sh models/distilled-qwen-bitnet
```

## Chat with Pre-built Models

You can also download and run existing BitNet models without training:

```bash
# Download Microsoft's official 2B model
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T

# Build it
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# Chat
./chat.sh models/BitNet-b1.58-2B-4T
```

## How It Works

The distillation pipeline implements three techniques from the [BitDistill paper](https://arxiv.org/abs/2510.13998):

1. **BitLinear**: Replaces all linear layers with ternary-quantized versions (weights in {-1, 0, +1}) using Straight-Through Estimator for gradient flow
2. **SubLN**: Inserts RMSNorm before the attention output and FFN down projections to stabilize quantized training
3. **Three-loss distillation**: Cross-entropy + KL-divergence on softened logits + MiniLM attention relation matching

```
Qwen2.5-0.5B (FP16, 1GB) --> distill.py --> export --> GGUF --> I2_S (224MB) --> bitnet.cpp (~107 tok/s)
```

## References

- [BitNet Distillation](https://arxiv.org/abs/2510.13998) - the paper this implements
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) - foundational BitNet b1.58 paper
- [microsoft/BitNet](https://github.com/microsoft/BitNet) - upstream inference framework

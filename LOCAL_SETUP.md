# BitNet Local Setup (macOS Apple Silicon)

Local setup notes for running BitNet on this machine (Apple M1, 16GB RAM, macOS).

## Prerequisites Installed

- **Miniconda** (`brew install miniconda`) - conda 26.1.1
- **LLVM 18** (`brew install llvm@18`) - Homebrew clang 18.1.8 at `/opt/homebrew/opt/llvm@18/bin/clang`
- **CMake** 4.2.0
- **Python** 3.9 (via conda env `bitnet-cpp`)

## One-Time Setup (Already Done)

```bash
# 1. Clone with submodules
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet

# 2. Accept conda TOS (first-time only)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 3. Create conda environment
conda create -n bitnet-cpp python=3.9 -y
conda activate bitnet-cpp
pip install -r requirements.txt
pip install datasets accelerate  # for distillation

# 4. Download the 2B model (pre-quantized GGUF, ~900MB)
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
    --local-dir models/BitNet-b1.58-2B-4T

# 5. Build — requires SDK paths for Homebrew clang to find macOS frameworks
export SDKROOT=$(xcrun --show-sdk-path)
export LIBRARY_PATH="$SDKROOT/usr/lib:${LIBRARY_PATH:-}"

cmake -B build -DBITNET_ARM_TL1=OFF \
    -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@18/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@18/bin/clang++ \
    -DCMAKE_EXE_LINKER_FLAGS="-L$SDKROOT/usr/lib -F$SDKROOT/System/Library/Frameworks" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$SDKROOT/usr/lib -F$SDKROOT/System/Library/Frameworks"

cmake --build build --config Release -j8
```

### Homebrew Clang Gotcha

The standard `python setup_env.py` uses plain `clang`/`clang++` which resolves to Homebrew's LLVM 18.
Homebrew clang can't find macOS system libraries (`libSystem`, `Accelerate.framework`) without
setting `SDKROOT`, `LIBRARY_PATH`, and framework search paths. The cmake invocation above handles this.
If you ever need to rebuild, always set those env vars first:

```bash
export SDKROOT=$(xcrun --show-sdk-path)
export LIBRARY_PATH="$SDKROOT/usr/lib:${LIBRARY_PATH:-}"
```

## Quick Start

```bash
conda activate bitnet-cpp
cd ~/Documents/BitNet
```

### Web UI with Live Stats (recommended)

```bash
./server.sh
```

Opens a chat server at **http://localhost:8080** with a browser-based chat interface.
Shows tokens/sec, timing, and generation stats live after each response.
Prometheus metrics available at `/metrics`. Ctrl-C to stop.

```bash
# Custom options: ./server.sh [model_dir] [threads] [port]
./server.sh models/BitNet-b1.58-2B-4T 8 9090
```

### Terminal Chat

```bash
./chat.sh
```

Interactive multi-turn chat in the terminal. Performance stats (tokens/sec) print when you Ctrl-C to quit.

```bash
# Custom options: ./chat.sh [model_dir] [threads]
./chat.sh models/BitNet-b1.58-2B-4T 8
```

### Single Prompt (non-interactive)

```bash
./build/bin/llama-cli \
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -t 4 -n 200 --temp 0.7 \
    -p "<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Your question here<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
```

### Benchmark

```bash
python utils/e2e_benchmark.py \
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -t 8 -n 128 -p 512
```

## Observed Performance (Apple M1, 4 threads)

| Metric | Value |
|--------|-------|
| Model size | 1.10 GiB (I2_S, 3.91 BPW) |
| KV cache | 150 MiB |
| Prompt eval | ~195 tokens/sec |
| Generation | ~33 tokens/sec |
| Load time | ~600ms (warm) / ~5.5s (cold) |

## Available Models

### Installed

| Model | Params | Size | Path |
|-------|--------|------|------|
| BitNet-b1.58-2B-4T | 2.4B | ~1.1 GiB | `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf` |

### Ready to Download

These models all support ARM I2_S and will work on this machine.

**Official (Microsoft):**

| Model | Params | Format | Download |
|-------|--------|--------|----------|
| [BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | 2.4B | Pre-quantized GGUF | Already installed |

**Community (need conversion via `setup_env.py`):**

| Model | Params | Est. Size | Download & Build |
|-------|--------|-----------|------------------|
| [bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large) | 0.7B | ~350 MiB | `python setup_env.py -hr 1bitLLM/bitnet_b1_58-large -q i2_s` |
| [bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) | 3.3B | ~1.5 GiB | See note below |
| [Llama3-8B-1.58](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens) | 8.0B | ~3 GiB | `python setup_env.py -hr HF1BitLLM/Llama3-8B-1.58-100B-tokens -q i2_s` |
| [Falcon3-1B-1.58bit](https://huggingface.co/tiiuae/Falcon3-1B-Instruct-1.58bit) | 1B | ~500 MiB | `python setup_env.py -hr tiiuae/Falcon3-1B-Instruct-1.58bit -q i2_s` |
| [Falcon3-3B-1.58bit](https://huggingface.co/tiiuae/Falcon3-3B-Instruct-1.58bit) | 3B | ~1.5 GiB | `python setup_env.py -hr tiiuae/Falcon3-3B-Instruct-1.58bit -q i2_s` |
| [Falcon3-7B-1.58bit](https://huggingface.co/tiiuae/Falcon3-7B-Instruct-1.58bit) | 7B | ~3 GiB | `python setup_env.py -hr tiiuae/Falcon3-7B-Instruct-1.58bit -q i2_s` |
| [Falcon3-10B-1.58bit](https://huggingface.co/tiiuae/Falcon3-10B-Instruct-1.58bit) | 10B | ~4 GiB | `python setup_env.py -hr tiiuae/Falcon3-10B-Instruct-1.58bit -q i2_s` |

> **Note on bitnet_b1_58-3B**: This model only supports TL1 on ARM (not I2_S). To use it:
> ```bash
> export SDKROOT=$(xcrun --show-sdk-path)
> export LIBRARY_PATH="$SDKROOT/usr/lib"
> python setup_env.py -hr 1bitLLM/bitnet_b1_58-3B -q tl1
> ```

> **Note on community models**: `setup_env.py -hr` downloads, converts to GGUF, quantizes, and rebuilds
> automatically. However, because Homebrew clang needs the SDK paths, you may need to set the env vars
> first or use the manual cmake approach from the setup section above.

### Download a model manually (without `setup_env.py`)

```bash
# Example: Llama3-8B-1.58
huggingface-cli download HF1BitLLM/Llama3-8B-1.58-100B-tokens \
    --local-dir models/Llama3-8B-1.58

# Convert and quantize
python utils/convert-helper-bitnet.py models/Llama3-8B-1.58

# Then chat with it
./chat.sh models/Llama3-8B-1.58
```

## Useful llama-cli Flags

| Flag | Description |
|------|-------------|
| `-cnv` | Conversation/chat mode (multi-turn) |
| `-p "..."` | System prompt (in `-cnv` mode) or user prompt |
| `-n 512` | Max tokens to generate |
| `-t 4` | Number of CPU threads (try 4-8) |
| `--temp 0.7` | Temperature (0.0 = deterministic, 1.0 = creative) |
| `--top-p 0.9` | Nucleus sampling threshold |
| `--repeat-penalty 1.1` | Penalize repetition |
| `-c 4096` | Context window size (max 4096 for BitNet-2B-4T, 32768 for Qwen-based) |
| `--no-warmup` | Skip warmup run (faster startup) |

---

## BitDistill: Distill Any Model → 1.58-bit BitNet

Implementation of [BitNet Distillation](https://arxiv.org/abs/2510.13998) (Microsoft, Oct 2025).
Takes a full-precision LLM and distills it into a 1.58-bit ternary model runnable in bitnet.cpp.

No official open-source training code exists — this is our own implementation based on the paper.

### How It Works

BitDistill converts a standard FP16 model (teacher) into a 1.58-bit model (student) using three
techniques applied simultaneously:

1. **BitLinear layers**: Replace all `nn.Linear` projections with ternary-quantized versions.
   Weights are constrained to {-1, 0, +1} during forward pass using absmean quantization
   with Straight-Through Estimator (STE) for gradient flow.

2. **SubLN (Sub-Layer Normalization)**: Insert RMSNorm at two points per transformer block:
   - Before the attention output projection (`o_proj`)
   - Before the FFN down projection (`down_proj`)

   This stabilizes activations entering quantized layers, critical for training convergence.

3. **Three-loss distillation**:
   - **L_CE**: Standard cross-entropy on task labels
   - **L_LD**: KL-divergence on softened logits (temperature τ=5.0) between teacher and student
   - **L_AD**: MiniLM-style self-relation matching on Q/K/V projections from the last transformer layer

The paper's ablation shows distillation alone (skipping continued pre-training) achieves nearly
the full benefit: 88.04% vs 88.17% on MNLI. Our implementation skips Stage 2 accordingly.

### Current Setup: Qwen2.5-0.5B → BitNet

| | Teacher | Student |
|--|---------|---------|
| Model | Qwen2.5-0.5B (FP16) | Qwen2.5-0.5B (1.58-bit) |
| Parameters | 494M | 494M |
| Size | ~1 GB | ~224 MB (I2_S quantized) |
| Precision | FP16 | W1.58A8 (ternary weights, INT8 activations) |
| Inference speed | — | ~107 tok/s on M1 |

### Pipeline

```
                    distill.py                export_bitnet.py
Qwen2.5-0.5B ──→ BitLinear + SubLN ──→ HuggingFace safetensors ──→ GGUF ──→ I2_S ──→ bitnet.cpp
  (teacher)      + distillation              (ternary weights)
                 (training)                     (export)              (convert)  (quantize) (inference)
```

### Training

```bash
conda activate bitnet-cpp

# Smoke test (5 min, CPU)
python distill/distill.py --max_steps 50 --device cpu --log_every 10

# Short run (30 min, MPS/Metal)
python distill/distill.py --max_steps 500 --device mps

# Full run (8-12 hours, MPS/Metal)
python distill/distill.py --epochs 2 --device mps
```

**Training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `mps` | `mps` (Metal GPU), `cpu`, or `cuda` |
| `--batch_size` | `4` | Micro-batch size per step |
| `--accumulation_steps` | `8` | Gradient accumulation (effective batch = 32) |
| `--max_length` | `512` | Sequence length |
| `--epochs` | `2` | Training epochs |
| `--max_steps` | None | Override epochs, stop after N optimizer steps |
| `--lr` | `1e-4` | Learning rate (cosine schedule with warmup) |
| `--tau` | `5.0` | Distillation temperature |
| `--lambda_ld` | `10.0` | Logits distillation loss weight |
| `--gamma_ad` | `1e-5` | Attention distillation loss weight |
| `--save_every` | `500` | Save checkpoint every N steps |
| `--output_dir` | `distill` | Where to save checkpoints |

**Dataset**: [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) (51k instruction examples, auto-downloaded).

**Memory**: ~6 GB (teacher 1GB + student 1GB + optimizer 2GB + activations 2GB). Fits in 16GB.

### Export & Convert

```bash
# 1. Export trained checkpoint to HuggingFace format
python distill/export_bitnet.py \
    --checkpoint distill/checkpoints/final.pt \
    --output models/distilled-qwen-bitnet

# 2. Convert to GGUF
python utils/convert-hf-to-gguf-bitnet.py models/distilled-qwen-bitnet/ --outtype f32

# 3. Quantize to I2_S (1885 MB → 224 MB)
./build/bin/llama-quantize \
    models/distilled-qwen-bitnet/ggml-model-f32.gguf \
    models/distilled-qwen-bitnet/ggml-model-i2_s.gguf I2_S 1

# 4. Chat with your distilled model
./chat.sh models/distilled-qwen-bitnet

# Or use the web UI
./server.sh models/distilled-qwen-bitnet
```

### What the Export Does

1. Loads the training checkpoint
2. Hard-quantizes all projection weights to final ternary values: `w = round(w / mean(|w|)).clamp(-1, 1)`
3. Renames tensors to match GGUF expectations (`o_proj.norm` → `inner_attn_ln`, `down_proj.norm` → `ffn_layernorm`)
4. Strips bias tensors (BitNet architecture doesn't use them)
5. Skips `lm_head.weight` (tied to `embed_tokens`)
6. Saves as safetensors + writes BitNet-compatible `config.json`
7. Copies Qwen2.5 tokenizer files

### Patches to Upstream Code

Two small patches were applied to `utils/convert-hf-to-gguf-bitnet.py`:

1. **Tokenizer fallback**: `BitnetModel.set_vocab()` now falls back to `_set_vocab_gpt2()` when
   `tokenizer.model` (SentencePiece) doesn't exist, enabling BPE-based models like Qwen.

2. **Qwen2 tokenizer hash**: Added the Qwen2.5 pre-tokenizer hash so the converter recognizes it.

### Files

```
distill/
  distill.py          Training script: BitLinear + SubLN + 3-loss distillation (~300 lines)
  export_bitnet.py    Export checkpoint → HuggingFace format for GGUF conversion (~130 lines)
  checkpoints/        Training checkpoints (created during training)
```

### References

- [BitNet Distillation (BitDistill)](https://arxiv.org/abs/2510.13998) — the paper this implements
- [Continual Quantization-Aware Pre-Training](https://arxiv.org/abs/2502.11895) — when to switch from 16-bit to 1.58-bit
- [gpu/model.py](gpu/model.py) — reference BitLinear and SubLN implementation from Microsoft

---

## Architecture Overview

```
Python layer (orchestration)
  setup_env.py            build orchestrator: codegen -> convert -> quantize -> compile
  run_inference.py        wrapper that invokes compiled llama-cli
  utils/convert-*.py      HuggingFace safetensors -> GGUF format conversion
  utils/codegen_tl1.py    generates ARM NEON lookup table kernel headers
  utils/codegen_tl2.py    generates x86 AVX lookup table kernel headers

C++ layer (compiled into llama-cli)
  src/ggml-bitnet-mad.cpp   I2_S kernel: ternary multiply-accumulate with SIMD
  src/ggml-bitnet-lut.cpp   TL1/TL2 kernels: lookup-table-based matmul
  include/ggml-bitnet.h     C API header
  include/gemm-config.h     tuning: block sizes, parallelism level

3rdparty/llama.cpp          modified llama.cpp fork with BitNet integration
gpu/                        CUDA-only GPU kernels (not usable on Mac)
preset_kernels/             pre-tuned LUT kernels for known model architectures

distill/                    BitDistill training pipeline (our addition)
```

### How 1.58-bit Quantization Works

Weights are **ternary**: each weight is one of {-1, 0, +1}, stored as {0, 1, 2} in 2 bits.

- **I2_S format**: packs 64 weights per block (ARM) with a float32 scale factor. Matmul uses NEON
  SIMD intrinsics (`vdotq_s32` with DOTPROD, `vmlal_s8` fallback).
- **TL1 format**: precomputes lookup tables from 8-bit activations x ternary weights. Kernels are
  code-generated per model layer dimensions for maximum performance.
- **W2A8**: 2-bit weights x 8-bit activations. Activations are quantized on-the-fly during inference.

### Key Papers

- [The Era of 1-bit LLMs (Feb 2024)](https://arxiv.org/abs/2402.17764) — foundational BitNet b1.58 paper
- [BitNet.cpp: Efficient Edge Inference (Feb 2025)](https://arxiv.org/abs/2502.11880) — this inference framework
- [BitNet a4.8: 4-bit Activations (Nov 2024)](https://arxiv.org/abs/2411.04965) — reduced activation precision
- [BitNet b1.58 2B4T Technical Report](https://huggingface.co/papers/2504.12285) — the 2B model training details
- [BitNet Distillation (Oct 2025)](https://arxiv.org/abs/2510.13998) — distilling FP16 models to 1.58-bit

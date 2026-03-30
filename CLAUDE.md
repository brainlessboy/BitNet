# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BitNet.cpp is Microsoft's official inference framework for 1-bit LLMs (BitNet b1.58). It provides fast, lossless inference of ternary-quantized models ({-1, 0, 1}) on CPU and GPU, built on a modified fork of llama.cpp.

## Build & Run

**Prerequisites**: Python >= 3.9, CMake >= 3.22, Clang >= 18 (not GCC), Conda recommended.

```bash
# Environment setup
conda create -n bitnet-cpp python=3.9 && conda activate bitnet-cpp
pip install -r requirements.txt

# Download the 2B model (smallest official model)
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T

# Build (handles codegen, model conversion, quantization, and compilation)
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# Interactive chat
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are helpful" -cnv

# Benchmark
python utils/e2e_benchmark.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -t 8 -n 128 -p 512
```

**Quantization types by architecture:**
- ARM64: `i2_s` (default), `tl1` (LUT-based, optimized for Apple Silicon)
- x86_64: `i2_s` (default), `tl2` (LUT-based)

**CMake flags**: `-DBITNET_ARM_TL1=ON` for ARM TL1, `-DBITNET_X86_TL2=ON` for x86 TL2.

The build outputs `build/bin/llama-cli` (inference), `build/bin/llama-quantize`, and `build/bin/llama-bench`.

## Architecture

```
Python orchestration layer
  setup_env.py          – end-to-end build orchestrator (codegen → convert → quantize → compile)
  run_inference.py      – thin wrapper that invokes compiled llama-cli binary
  utils/convert-*.py    – HuggingFace → GGUF model conversion
  utils/codegen_tl1.py  – generates ARM LUT kernel headers
  utils/codegen_tl2.py  – generates x86 LUT kernel headers

C++ inference layer (compiled into llama-cli via llama.cpp)
  src/ggml-bitnet-mad.cpp  – I2_S kernel: multiply-accumulate with SIMD (AVX2/NEON)
  src/ggml-bitnet-lut.cpp  – TL1/TL2 kernels: lookup-table-based matmul
  include/ggml-bitnet.h    – C API header
  include/gemm-config.h    – runtime tuning: block sizes, parallelism

3rdparty/llama.cpp         – modified llama.cpp fork with BitNet hooks
gpu/                       – CUDA-only GPU kernels (W2A8, dp4a-based)
preset_kernels/            – pre-tuned LUT kernel headers for known model architectures
```

## Key Concepts

**I2_S quantization** (`ggml-bitnet-mad.cpp`): Weights are ternary {-1, 0, 1} stored as {0, 1, 2} in 2 bits. Block size is 128 on x86, 64 on ARM. Each block stores packed 2-bit values + a float32 scale. Matmul is done via multiply-accumulate dot products using SIMD intrinsics.

**TL1/TL2 quantization** (`ggml-bitnet-lut.cpp`): Lookup-table approach where activations are quantized to 8-bit, then ternary weight × activation products are precomputed in tables. Kernels are code-generated per model layer dimensions by `codegen_tl1.py`/`codegen_tl2.py`.

**Activation quantization**: W2A8 format — 2-bit weights × 8-bit activations. Embedding layers can optionally use Q6_K quantization (`--quant-embd` flag) for memory savings with minimal quality loss.

## GPU Kernels (CUDA only)

Located in `gpu/`. Separate conda env, separate build:
```bash
cd gpu && pip install -r requirements.txt
cd bitnet_kernels && bash compile.sh
python test.py
```

## Tuning

`include/gemm-config.h` controls `ROW_BLOCK_SIZE`, `COL_BLOCK_SIZE`, and `PARALLEL_SIZE`. Use `utils/tune_gemm_config.py` for automated tuning. Preset configs exist in `preset_kernels/` for known models.

## BitDistill: Distillation Pipeline

Distills a standard HuggingFace model into a 1.58-bit BitNet model using knowledge distillation (arxiv 2510.13998). The pipeline: train → export → GGUF convert → quantize → inference.

**Prerequisites**: `pip install -r distill/requirements.txt`

### Quick Start (Local)

```bash
# Smoke test (CPU, ~5 min)
python distill/distill.py --max_steps 50 --device cpu

# Local MPS run (Apple Silicon, ~10 hours for 1 epoch)
python distill/distill.py --epochs 1 --device mps

# Full local run
python distill/distill.py --epochs 2 --device mps --save_every 200
```

### Cloud GPU (Recommended for Quality Results)

```bash
# Qwen2.5-0.5B validation run (H100, ~45 min, good for testing pipeline)
python distill/distill.py \
  --device cuda \
  --teacher_model Qwen/Qwen2.5-0.5B \
  --dataset alpaca \
  --batch_size 8 \
  --accumulation_steps 4 \
  --max_length 256 \
  --max_steps 5000 \
  --lr 5e-4 \
  --tau 2.0 \
  --save_every 500

# Qwen2.5-3B, SlimOrca dataset, H100 80GB
python distill/distill.py \
  --device cuda \
  --teacher_model Qwen/Qwen2.5-3B \
  --dataset slimorca \
  --batch_size 4 \
  --accumulation_steps 8 \
  --max_length 1024 \
  --max_steps 10000 \
  --lr 5e-4 \
  --tau 2.0 \
  --save_every 500

# Resume from checkpoint (critical for spot instances)
python distill/distill.py \
  --device cuda \
  --teacher_model Qwen/Qwen2.5-3B \
  --dataset slimorca \
  --resume_from distill/checkpoints/step_1000.pt
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `mps` | `mps`, `cpu`, or `cuda` |
| `--teacher_model` | `Qwen/Qwen2.5-0.5B` | HuggingFace teacher model |
| `--student_model` | `Qwen/Qwen2.5-0.5B` | HuggingFace student base model |
| `--dataset` | `alpaca` | `alpaca` (51k), `slimorca` (500k), `openhermes` (1M) |
| `--batch_size` | `2` | Micro-batch size |
| `--accumulation_steps` | `16` | Gradient accumulation steps |
| `--max_length` | `512` | Max sequence length |
| `--epochs` | `2` | Training epochs |
| `--max_steps` | None | Override epochs, stop after N optimizer steps |
| `--lr` | `1e-4` | Learning rate (cosine decay) |
| `--tau` | `5.0` | Distillation temperature (recommended: 2.0) |
| `--save_every` | `500` | Checkpoint save interval (steps) |
| `--resume_from` | None | Resume from checkpoint path |

### Export & Inference Pipeline

One command does everything (export → GGUF → quantize → chat):

```bash
python distill/deploy.py distill/checkpoints/final.pt
```

Options:
```bash
# Custom tokenizer and output dir
python distill/deploy.py final.pt --tokenizer Qwen/Qwen2.5-3B -o models/my-bitnet

# Just build, don't launch chat
python distill/deploy.py final.pt --no-chat
```

<details>
<summary>Manual step-by-step (equivalent)</summary>

```bash
# 1. Export checkpoint to HuggingFace format
python distill/export_bitnet.py \
  --checkpoint distill/checkpoints/final.pt \
  --output models/distilled-bitnet \
  --tokenizer_model Qwen/Qwen2.5-0.5B  # use teacher model if different

# 2. Convert to GGUF
python utils/convert-hf-to-gguf-bitnet.py models/distilled-bitnet/ --outtype f32

# 3. Quantize to I2_S
./build/bin/llama-quantize \
  models/distilled-bitnet/ggml-model-f32.gguf \
  models/distilled-bitnet/ggml-model-i2_s.gguf I2_S 1

# 4. Run inference
python run_inference.py -m models/distilled-bitnet/ggml-model-i2_s.gguf -p "You are helpful" -cnv
```
</details>

### GPU VRAM Requirements

| Teacher Model | FP16 VRAM | Total Pipeline | Recommended GPU |
|--------------|-----------|----------------|-----------------|
| Qwen2.5-0.5B | ~1 GB | ~10 GB | MPS 16GB / A100 40GB |
| Qwen2.5-1.5B | ~3 GB | ~14 GB | A100 40GB |
| Qwen2.5-3B | ~6 GB | ~18 GB | A100 40GB / 80GB |
| Qwen2.5-7B | ~14 GB | ~28 GB | A100 80GB |
| Qwen2.5-14B | ~28 GB | ~42 GB | A100 80GB |

### Architecture

```
distill/
  distill.py              – Main distillation training script
  export_bitnet.py        – Export checkpoint → HuggingFace safetensors format
  deploy.py               – One-command pipeline: export → GGUF → quantize → chat
  dashboard.py            – Live curses terminal dashboard (reads training_log.jsonl)
  plot.py                 – Generate PNG training curve plots (requires matplotlib)
  test_checkpoint.py      – Test checkpoint quality directly in PyTorch (no GGUF)
  inspect_checkpoint.py   – Inspect weight statistics and ternary analysis
  inspect_logits.py       – Compare student vs teacher logit distributions
  requirements.txt        – Python dependencies for distillation
```

### Monitoring Training

`distill.py` writes `distill/training_log.jsonl` (append-only, one JSON line per step). Three ways to monitor:

```bash
tail -f distill_log.txt                                    # raw scrolling log
python distill/dashboard.py distill/training_log.jsonl     # live color terminal dashboard (q to quit)
python distill/plot.py distill/training_log.jsonl           # PNG plot → distill/training_plot.png
```

### Testing Checkpoints

Test model quality at any point without stopping training or converting to GGUF:

```bash
# Test generation (use -t and -r for better output)
python distill/test_checkpoint.py distill/checkpoints/step_1000.pt -n 50 -t 0.7 -r 1.3

# Test with Alpaca format (matches training data format)
python distill/test_checkpoint.py distill/checkpoints/step_1000.pt -t 0.7 -r 1.3 \
  --prompt "### Instruction:\nWhat is the capital of Switzerland?\n\n### Response:\n"

# Compare student vs teacher logit distributions
python distill/inspect_logits.py distill/checkpoints/step_1000.pt

# Inspect weight statistics
python distill/inspect_checkpoint.py distill/checkpoints/step_1000.pt

# Test base model without BitLinear (teacher quality reference)
python distill/test_checkpoint.py "base:Qwen/Qwen2.5-0.5B" -n 30
```

### GGUF Export Patches

The GGUF export requires patches to `3rdparty/llama.cpp` for Qwen2.5-based distilled models. Run after any fresh install:

```bash
python patch_llama_cpp.py
cmake --build build --config Release -j$(nproc)
```

This applies three fixes: BITNET_B158 architecture support, SiLU activation (Qwen2.5 uses SiLU, not ReluSquared), and output layer handling for tied embeddings.

### Distillation Architecture

The distillation uses three loss components:
- **CE**: Cross-entropy on next-token prediction
- **LD**: KL divergence between student/teacher logit distributions (weight: λ=10, padding tokens excluded)
- **AD**: MiniLM-style attention self-relation distillation (weight: γ=1e-5)

Student modifications (BitLinear + SubLN surgery):
- All projection layers (q/k/v/o_proj, gate/up/down_proj) → BitLinear (ternary weights via STE)
- SubLN (RMSNorm) inserted before o_proj and down_proj
- Embeddings frozen during training

### Key Findings from Experiments

**Critical parameters:**
- `--tau 2.0` works much better than `5.0` (sharper teacher signal through ternary bottleneck)
- `--lr 5e-4` recommended over `1e-4` (stronger gradient signal to escape repetition traps)
- Effective batch size 32 (e.g., `batch_size 8 × accumulation_steps 4`)

**Training behavior:**
- BitLinear surgery immediately destroys pretrained model quality (complete gibberish at step 0)
- CE loss drops fast in first ~300 steps then flattens — CE alone is misleading because it includes easy formatting/instruction tokens
- Use `inspect_logits.py` to compare student vs teacher distributions — KL divergence is a better quality signal than CE
- Coherent generation typically requires many thousands of steps
- Repetition penalty (`-r 1.3`) and temperature (`-t 0.7`) are essential for generation quality

**Checkpoint saving:**
- Periodic checkpoints exclude optimizer state (~12GB vs ~33GB for 3B model)
- Auto-cleanup keeps only last 2 periodic checkpoints
- Final checkpoint saves full state

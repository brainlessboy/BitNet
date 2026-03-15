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

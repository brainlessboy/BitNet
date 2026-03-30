#!/usr/bin/env python3
"""
Patch 3rdparty/llama.cpp for BitDistill compatibility.

Applies fixes needed for distilled Qwen2.5-based BitNet models:
1. Add BITNET_B158 architecture to Python gguf library
2. Change FFN activation from ReluSquared to SiLU in build_bitnet_158()
3. Use model.output when available instead of tok_embd

Run after setup_env.py or any fresh install/update of 3rdparty/llama.cpp.

Usage:
    python patch_llama_cpp.py
"""

import os
import sys


def patch_file(path, patches, description):
    """Apply text replacements to a file. Returns True if any changes made."""
    if not os.path.exists(path):
        print(f"  SKIP {path} (not found)")
        return False

    with open(path) as f:
        content = f.read()

    original = content
    for old, new, marker in patches:
        # Skip if already patched (check for unique marker string)
        if marker in content:
            continue
        if old not in content:
            print(f"  WARN Could not find patch target in {path}")
            print(f"       Looking for: {old[:80]}...")
            return False
        content = content.replace(old, new, 1)

    if content == original:
        print(f"  OK   {description} (already patched)")
        return False

    with open(path, "w") as f:
        f.write(content)
    print(f"  DONE {description}")
    return True


def patch_gguf_constants(path):
    """Add BITNET_B158 architecture to gguf constants.py."""
    patches = [
        # Add enum value
        (
            "    BITNET_25    = auto()\n    T5",
            "    BITNET_25    = auto()\n    BITNET_B158  = auto()\n    T5",
            "BITNET_B158",
        ),
        # Add name mapping
        (
            '    MODEL_ARCH.BITNET_25:      "bitnet-25",',
            '    MODEL_ARCH.BITNET_25:      "bitnet-25",\n    MODEL_ARCH.BITNET_B158:    "bitnet-b1.58",',
            "bitnet-b1.58",
        ),
        # Add tensor list (after BITNET_25 closing bracket, before T5)
        (
            "        MODEL_TENSOR.FFN_SUB_NORM,\n    ],\n    MODEL_ARCH.T5:",
            "        MODEL_TENSOR.FFN_SUB_NORM,\n    ],\n"
            "    MODEL_ARCH.BITNET_B158: [\n"
            "        MODEL_TENSOR.TOKEN_EMBD,\n"
            "        MODEL_TENSOR.OUTPUT_NORM,\n"
            "        MODEL_TENSOR.OUTPUT,\n"
            "        MODEL_TENSOR.ROPE_FREQS,\n"
            "        MODEL_TENSOR.ATTN_NORM,\n"
            "        MODEL_TENSOR.ATTN_Q,\n"
            "        MODEL_TENSOR.ATTN_K,\n"
            "        MODEL_TENSOR.ATTN_V,\n"
            "        MODEL_TENSOR.ATTN_OUT,\n"
            "        MODEL_TENSOR.ATTN_ROT_EMBD,\n"
            "        MODEL_TENSOR.FFN_GATE_INP,\n"
            "        MODEL_TENSOR.FFN_NORM,\n"
            "        MODEL_TENSOR.FFN_GATE,\n"
            "        MODEL_TENSOR.FFN_DOWN,\n"
            "        MODEL_TENSOR.FFN_UP,\n"
            "        MODEL_TENSOR.FFN_GATE_EXP,\n"
            "        MODEL_TENSOR.FFN_DOWN_EXP,\n"
            "        MODEL_TENSOR.FFN_UP_EXP,\n"
            "        MODEL_TENSOR.ATTN_SUB_NORM,\n"
            "        MODEL_TENSOR.FFN_SUB_NORM,\n"
            "    ],\n    MODEL_ARCH.T5:",
            "BITNET_B158: [",
        ),
    ]
    return patch_file(path, patches, f"gguf constants: {os.path.basename(os.path.dirname(os.path.dirname(path)))}")


def patch_llama_cpp(path):
    """Patch llama.cpp: SiLU activation + output layer fix."""
    patches = [
        # Fix 1: Change ReluSquared to SiLU in build_bitnet_158
        (
            "LLM_FFN_RELU_SQR, LLM_FFN_PAR, cb, il);\n"
            "            cb(cur, \"ffn_out\", il);\n"
            "\n"
            "            cur = llm_build_norm(ctx0, cur, hparams,\n"
            "                            model.layers[il].ffn_sub_norm",
            "LLM_FFN_SILU, LLM_FFN_PAR, cb, il);\n"
            "            cb(cur, \"ffn_out\", il);\n"
            "\n"
            "            cur = llm_build_norm(ctx0, cur, hparams,\n"
            "                            model.layers[il].ffn_sub_norm",
            # Marker: check if SiLU is already used in the bitnet_158 ffn block
            "LLM_FFN_SILU, LLM_FFN_PAR, cb, il);\n            cb(cur, \"ffn_out\", il);\n\n            cur = llm_build_norm(ctx0, cur, hparams,\n                            model.layers[il].ffn_sub_norm",
        ),
        # Fix 2: Use model.output when available
        (
            "cur = llm_build_lora_mm(lctx, ctx0, model.tok_embd, cur);\n"
            "\n"
            "        cb(cur, \"result_output\", -1);\n"
            "\n"
            "        ggml_build_forward_expand(gf, cur);\n"
            "\n"
            "        return gf;\n"
            "    }",
            "cur = llm_build_lora_mm(lctx, ctx0, model.output ? model.output : model.tok_embd, cur);\n"
            "\n"
            "        cb(cur, \"result_output\", -1);\n"
            "\n"
            "        ggml_build_forward_expand(gf, cur);\n"
            "\n"
            "        return gf;\n"
            "    }",
            "model.output ? model.output : model.tok_embd",
        ),
    ]
    return patch_file(path, patches, "llama.cpp: SiLU + output layer")


def main():
    root = os.path.dirname(os.path.abspath(__file__))

    print("Patching 3rdparty/llama.cpp for BitDistill...\n")

    changed = False

    # 1. Patch gguf constants (local 3rdparty copy)
    gguf_constants = os.path.join(root, "3rdparty", "llama.cpp", "gguf-py", "gguf", "constants.py")
    changed |= patch_gguf_constants(gguf_constants)

    # 2. Patch system-installed gguf package (if exists)
    try:
        import gguf
        system_constants = os.path.join(os.path.dirname(gguf.__file__), "constants.py")
        if os.path.abspath(system_constants) != os.path.abspath(gguf_constants):
            changed |= patch_gguf_constants(system_constants)
    except ImportError:
        pass

    # 3. Patch llama.cpp source
    llama_cpp = os.path.join(root, "3rdparty", "llama.cpp", "src", "llama.cpp")
    changed |= patch_llama_cpp(llama_cpp)

    print()
    if changed:
        print("Patches applied. Rebuild with:")
        print("  cmake --build build --config Release -j$(nproc)")
    else:
        print("All patches already applied.")


if __name__ == "__main__":
    main()

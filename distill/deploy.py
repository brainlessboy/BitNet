#!/usr/bin/env python3
"""
Deploy a BitDistill checkpoint: export → GGUF → quantize → chat.

Takes a final.pt checkpoint and runs the full pipeline in one command.

Usage:
    python distill/deploy.py distill/checkpoints/final.pt

    # Custom output dir and tokenizer
    python distill/deploy.py distill/checkpoints/final.pt \
        --output models/my-bitnet \
        --tokenizer Qwen/Qwen2.5-3B

    # Skip chat, just build the quantized model
    python distill/deploy.py distill/checkpoints/final.pt --no-chat
"""

import argparse
import os
import subprocess
import sys


def run(cmd, desc):
    print(f"\n\033[36m{'=' * 60}\033[0m")
    print(f"\033[1m  {desc}\033[0m")
    print(f"\033[36m{'=' * 60}\033[0m\n")
    print(f"  \033[2m$ {' '.join(cmd)}\033[0m\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n\033[31mFailed: {desc}\033[0m")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Deploy BitDistill checkpoint: export → GGUF → quantize → chat",
    )
    parser.add_argument("checkpoint", help="Path to .pt checkpoint (e.g. distill/checkpoints/final.pt)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: models/distilled-bitnet)")
    parser.add_argument("--tokenizer", "-t", default=None,
                        help="HuggingFace tokenizer model (default: auto-detect from checkpoint)")
    parser.add_argument("--quant-type", "-q", default="i2_s",
                        help="Quantization type (default: i2_s)")
    parser.add_argument("--no-chat", action="store_true",
                        help="Skip launching interactive chat")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    output = args.output or os.path.join(root, "models", "distilled-bitnet")

    # Auto-detect tokenizer from checkpoint if not specified
    tokenizer = args.tokenizer
    if tokenizer is None:
        import torch
        print("  Auto-detecting tokenizer from checkpoint...")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = ckpt.get("config", None)
        if config and hasattr(config, "_name_or_path"):
            tokenizer = config._name_or_path
            print(f"  Detected: {tokenizer}")
        else:
            print("  Could not detect tokenizer. Use --tokenizer to specify.")
            sys.exit(1)
        del ckpt

    # Paths
    f32_gguf = os.path.join(output, "ggml-model-f32.gguf")
    quant_gguf = os.path.join(output, f"ggml-model-{args.quant_type}.gguf")
    quantize_bin = os.path.join(root, "build", "bin", "llama-quantize")

    if not os.path.exists(quantize_bin):
        print(f"\n\033[31mllama-quantize not found at {quantize_bin}\033[0m")
        print("Run setup_env.py first to build the inference engine.")
        sys.exit(1)

    # Step 1: Export to HuggingFace format
    run([
        sys.executable, os.path.join(root, "distill", "export_bitnet.py"),
        "--checkpoint", args.checkpoint,
        "--output", output,
        "--tokenizer_model", tokenizer,
    ], "Step 1/3: Export checkpoint → HuggingFace format")

    # Step 2: Convert to GGUF
    run([
        sys.executable, os.path.join(root, "utils", "convert-hf-to-gguf-bitnet.py"),
        output, "--outtype", "f32",
    ], "Step 2/3: Convert → GGUF")

    # Step 3: Quantize
    run([
        quantize_bin, f32_gguf, quant_gguf, args.quant_type.upper(), "1",
    ], f"Step 3/3: Quantize → {args.quant_type.upper()}")

    # Summary
    size_mb = os.path.getsize(quant_gguf) / (1024 * 1024)
    print(f"\n\033[32m{'=' * 60}\033[0m")
    print(f"\033[1m\033[32m  Done! Quantized model ready.\033[0m")
    print(f"\033[32m{'=' * 60}\033[0m")
    print(f"\n  Model: {quant_gguf}")
    print(f"  Size:  {size_mb:.0f} MB")
    print(f"\n  Run manually:")
    print(f"    python run_inference.py -m {quant_gguf} -p \"You are helpful\" -cnv\n")

    # Step 4: Chat (optional)
    if not args.no_chat:
        run([
            sys.executable, os.path.join(root, "run_inference.py"),
            "-m", quant_gguf,
            "-p", "You are helpful",
            "-cnv",
        ], "Launching interactive chat")


if __name__ == "__main__":
    main()

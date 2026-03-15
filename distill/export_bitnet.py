"""
Export a BitDistill-trained student checkpoint to HuggingFace format
suitable for GGUF conversion and inference in bitnet.cpp.

Usage:
    python distill/export_bitnet.py \
        --checkpoint distill/checkpoints/final.pt \
        --output models/distilled-qwen-bitnet

Then convert to GGUF:
    python utils/convert-hf-to-gguf-bitnet.py models/distilled-qwen-bitnet/ --outtype f32
    ./build/bin/llama-quantize \\
        models/distilled-qwen-bitnet/ggml-model-f32.gguf \\
        models/distilled-qwen-bitnet/ggml-model-i2_s.gguf I2_S 1
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer


def weight_quant(weight: torch.Tensor) -> torch.Tensor:
    """Hard-quantize weight to ternary {-1, 0, +1} with absmean scale."""
    dtype = weight.dtype
    w = weight.float()
    s = 1.0 / w.abs().mean().clamp(min=1e-5)
    result = (w * s).round().clamp(-1, 1) / s
    return result.to(dtype)


def flatten_state_dict(state_dict: dict) -> dict:
    """
    Flatten NormedLinear wrappers back to the tensor names expected by
    the GGUF converter (tensor_mapping.py).

    Training names                          → Export names
    ---------------------------------------------------------------
    layers.{i}.self_attn.o_proj.norm.weight → layers.{i}.self_attn.inner_attn_ln.weight
    layers.{i}.self_attn.o_proj.linear.*    → layers.{i}.self_attn.o_proj.*
    layers.{i}.mlp.down_proj.norm.weight    → layers.{i}.mlp.ffn_layernorm.weight
    layers.{i}.mlp.down_proj.linear.*       → layers.{i}.mlp.down_proj.*
    """
    new_sd = {}

    for key, value in state_dict.items():
        new_key = key

        # Attention SubLN: o_proj.norm → inner_attn_ln
        if ".self_attn.o_proj.norm." in key:
            new_key = key.replace(".self_attn.o_proj.norm.", ".self_attn.inner_attn_ln.")
        # Attention o_proj: o_proj.linear → o_proj
        elif ".self_attn.o_proj.linear." in key:
            new_key = key.replace(".self_attn.o_proj.linear.", ".self_attn.o_proj.")

        # FFN SubLN: down_proj.norm → ffn_layernorm
        elif ".mlp.down_proj.norm." in key:
            new_key = key.replace(".mlp.down_proj.norm.", ".mlp.ffn_layernorm.")
        # FFN down_proj: down_proj.linear → down_proj
        elif ".mlp.down_proj.linear." in key:
            new_key = key.replace(".mlp.down_proj.linear.", ".mlp.down_proj.")

        # Skip lm_head when using tied embeddings (shared with embed_tokens)
        if new_key == "lm_head.weight":
            continue

        # Skip bias tensors — BitNet architecture doesn't use them
        if new_key.endswith(".bias"):
            continue

        new_sd[new_key] = value

    return new_sd


QUANTIZE_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)


def export(args):
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    config = ckpt["config"]

    # Flatten NormedLinear wrappers
    print("Flattening state dict...")
    state_dict = flatten_state_dict(state_dict)

    # Hard-quantize projection weights to ternary
    print("Quantizing weights to ternary...")
    n_quantized = 0
    for key in list(state_dict.keys()):
        if key.endswith(QUANTIZE_SUFFIXES):
            state_dict[key] = weight_quant(state_dict[key])
            n_quantized += 1
    print(f"  Quantized {n_quantized} tensors")

    # Convert all to float16 for storage
    for key in state_dict:
        if state_dict[key].dtype == torch.float32:
            state_dict[key] = state_dict[key].to(torch.float16)

    # Write output directory
    os.makedirs(args.output, exist_ok=True)

    # Save weights
    safetensors_path = os.path.join(args.output, "model.safetensors")
    print(f"Saving weights to {safetensors_path}")
    save_file(state_dict, safetensors_path)

    # Write config.json for BitNet architecture
    bitnet_config = {
        "architectures": ["BitnetForCausalLM"],
        "model_type": "bitnet",
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "vocab_size": config.vocab_size,
        "rms_norm_eps": config.rms_norm_eps,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", True),
        "hidden_act": getattr(config, "hidden_act", "silu"),
        "torch_dtype": "float16",
    }
    config_path = os.path.join(args.output, "config.json")
    with open(config_path, "w") as f:
        json.dump(bitnet_config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Copy tokenizer from Qwen2.5-0.5B
    print("Copying tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizer.save_pretrained(args.output)

    # Print summary
    total_params = sum(v.numel() for v in state_dict.values())
    total_size_mb = sum(v.numel() * v.element_size() for v in state_dict.values()) / (1024 * 1024)
    print(f"\nExport complete:")
    print(f"  Output dir: {args.output}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: {total_size_mb:.1f} MB (FP16 safetensors)")
    print(f"  Config: {config_path}")

    # Print tensor names for verification
    print(f"\nTensor names ({len(state_dict)}):")
    for key in sorted(state_dict.keys())[:20]:
        print(f"  {key}: {list(state_dict[key].shape)}")
    if len(state_dict) > 20:
        print(f"  ... and {len(state_dict) - 20} more")

    print(f"\nNext steps:")
    print(f"  1. python utils/convert-hf-to-gguf-bitnet.py {args.output} --outtype f32")
    print(f"  2. ./build/bin/llama-quantize {args.output}/ggml-model-f32.gguf {args.output}/ggml-model-i2_s.gguf I2_S 1")
    print(f"  3. ./chat.sh {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Export BitDistill checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Output directory for HF-format model")
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()

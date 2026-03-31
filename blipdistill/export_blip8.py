"""
Export BLIP-8 format: 8 ternary planes per projection layer.

Saves each plane as a {-1, 0, +1} int8 matrix with its power-of-2 scale.
This is the hardware-ready format for analog crossbar arrays.
"""

import json
import os

import torch
from blipdistill.binary_transform import decompose_matrix


def export_blip8(teacher_state_dict: dict, config, output_dir: str, k: int = 8):
    """
    Export model weights in BLIP-8 format.

    Output structure:
        output_dir/
            metadata.json       — model info, architecture, scales
            embeddings.pt       — embedding weights (float16, unchanged)
            norms.pt            — all RMSNorm weights (float32, unchanged)
            layer_00/
                q_proj_planes.pt  — {planes: int8 tensor (k, out, in), scales: float tensor (k,)}
                k_proj_planes.pt
                v_proj_planes.pt
                o_proj_planes.pt
                gate_proj_planes.pt
                up_proj_planes.pt
                down_proj_planes.pt
            layer_01/
                ...
    """
    os.makedirs(output_dir, exist_ok=True)

    n_layers = config.num_hidden_layers
    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    total_planes = 0
    total_ternary_weights = 0

    # Export projection layers as ternary planes
    for layer_idx in range(n_layers):
        layer_dir = os.path.join(output_dir, f"layer_{layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        for proj_name in proj_names:
            if proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                key = f"model.layers.{layer_idx}.self_attn.{proj_name}.weight"
            else:
                key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"

            W = teacher_state_dict[key].float().cpu()
            planes, scales, delta = decompose_matrix(W, k=k)

            # Convert planes to int8 {-1, 0, +1}
            planes_int8 = torch.stack([p.to(torch.int8) for p in planes])  # (k, out, in)

            torch.save({
                "planes": planes_int8,
                "scales": scales,
                "delta": delta,
                "shape": list(W.shape),
            }, os.path.join(layer_dir, f"{proj_name}_planes.pt"))

            total_planes += k
            total_ternary_weights += planes_int8.numel()

        if (layer_idx + 1) % 4 == 0:
            print(f"    Layer {layer_idx + 1}/{n_layers} exported")

    # Export embeddings (unchanged, float16)
    embed_data = {}
    embed_data["embed_tokens"] = teacher_state_dict["model.embed_tokens.weight"].cpu().half()
    if "lm_head.weight" in teacher_state_dict:
        embed_data["lm_head"] = teacher_state_dict["lm_head.weight"].cpu().half()
    torch.save(embed_data, os.path.join(output_dir, "embeddings.pt"))

    # Export norms (unchanged, float32)
    norm_data = {}
    norm_data["output_norm"] = teacher_state_dict["model.norm.weight"].cpu().float()
    for layer_idx in range(n_layers):
        norm_data[f"layer_{layer_idx}.input_layernorm"] = \
            teacher_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"].cpu().float()
        norm_data[f"layer_{layer_idx}.post_attention_layernorm"] = \
            teacher_state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"].cpu().float()
    torch.save(norm_data, os.path.join(output_dir, "norms.pt"))

    # Metadata
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None and hasattr(config, "rope_parameters"):
        rope_theta = config.rope_parameters.get("rope_theta", 10000.0)

    metadata = {
        "format": "BLIP-8",
        "description": "Binary Layered Inference Planes, 8 ternary",
        "source_model": config._name_or_path,
        "n_planes": k,
        "n_layers": n_layers,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "vocab_size": config.vocab_size,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": rope_theta,
        "total_crossbar_arrays": total_planes,
        "total_ternary_weights": total_ternary_weights,
        "projections_per_layer": proj_names,
        "plane_scales": "power-of-2: delta × [0.5, 0.25, 0.125, ..., 0.5^k]",
        "ternary_values": "{-1, 0, +1} stored as int8",
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    total_size = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(output_dir) for fn in fns
    )

    print(f"\n  BLIP-8 Export Complete:")
    print(f"    Crossbar arrays:     {total_planes:,}")
    print(f"    Ternary weights:     {total_ternary_weights:,}")
    print(f"    Storage:             {total_size / (1024**2):.0f} MB")
    print(f"    Output:              {output_dir}")

    return metadata

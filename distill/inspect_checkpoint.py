#!/usr/bin/env python3
"""
Inspect a BitDistill checkpoint: weight statistics, ternary analysis.

Usage:
    python distill/inspect_checkpoint.py distill/checkpoints/step_100.pt
"""

import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect BitDistill checkpoint")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    args = parser.parse_args()

    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    print(f"Step: {ckpt.get('step', '?')}")
    print(f"Tensors: {len(sd)}")

    # Categorize tensors
    proj_keys = [k for k in sd if any(p in k for p in
                 ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
                 and "weight" in k and "norm" not in k]
    norm_keys = [k for k in sd if "norm" in k or "layernorm" in k]
    embed_keys = [k for k in sd if "embed" in k or "lm_head" in k]

    print(f"\nProjection layers: {len(proj_keys)}")
    print(f"Norm layers: {len(norm_keys)}")
    print(f"Embedding layers: {len(embed_keys)}")

    # Analyze projection weights (these should be quasi-ternary from BitLinear)
    print(f"\n{'=' * 70}")
    print("PROJECTION WEIGHT ANALYSIS (should be quasi-ternary)")
    print(f"{'=' * 70}")

    all_near_ternary = []
    for i, k in enumerate(proj_keys[:6]):  # first 6 layers
        w = sd[k].float()
        delta = w.abs().mean().clamp(min=1e-5)
        w_normalized = w / delta

        # Check how close to ternary
        rounded = w_normalized.round().clamp(-1, 1)
        residual = (w_normalized - rounded).abs().mean()

        # Distribution of normalized weights
        near_neg1 = (w_normalized < -0.5).float().mean() * 100
        near_zero = ((w_normalized >= -0.5) & (w_normalized <= 0.5)).float().mean() * 100
        near_pos1 = (w_normalized > 0.5).float().mean() * 100

        pct_ternary = near_neg1 + near_zero + near_pos1  # should be ~100%

        print(f"\n  {k}")
        print(f"    shape: {list(w.shape)}")
        print(f"    scale (absmean): {delta:.6f}")
        print(f"    weight range: [{w.min():.4f}, {w.max():.4f}]")
        print(f"    unique values: {w.unique().numel()}")
        print(f"    normalized distribution: -1:{near_neg1:.1f}%  0:{near_zero:.1f}%  +1:{near_pos1:.1f}%")
        print(f"    residual from ternary: {residual:.4f} (0=perfectly ternary)")
        all_near_ternary.append(residual.item())

    avg_residual = sum(all_near_ternary) / len(all_near_ternary)
    print(f"\n  Average ternary residual: {avg_residual:.4f}")
    if avg_residual < 0.1:
        print("  Weights are close to ternary (expected)")
    else:
        print("  WARNING: Weights are NOT ternary - STE may not be working")

    # Analyze embeddings (should be frozen = unchanged from pretrained)
    print(f"\n{'=' * 70}")
    print("EMBEDDING ANALYSIS (should be frozen)")
    print(f"{'=' * 70}")
    for k in embed_keys[:2]:
        w = sd[k].float()
        print(f"\n  {k}")
        print(f"    shape: {list(w.shape)}")
        print(f"    range: [{w.min():.4f}, {w.max():.4f}]")
        print(f"    mean: {w.mean():.6f}, std: {w.std():.4f}")

    # Analyze norms
    print(f"\n{'=' * 70}")
    print("NORM ANALYSIS (SubLN weights)")
    print(f"{'=' * 70}")
    sub_norms = [k for k in norm_keys if "sub_norm" in k or "inner_attn_ln" in k or "ffn_layernorm" in k]
    regular_norms = [k for k in norm_keys if k not in sub_norms]

    for k in sub_norms[:4]:
        w = sd[k].float()
        print(f"  {k}: mean={w.mean():.4f} std={w.std():.4f} range=[{w.min():.4f}, {w.max():.4f}]")

    # Compare student output logits distribution
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    total_params = sum(v.numel() for v in sd.values())
    proj_params = sum(sd[k].numel() for k in proj_keys)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Projection parameters: {proj_params:,} ({proj_params/total_params*100:.1f}%)")
    print(f"  Average ternary residual: {avg_residual:.4f}")


if __name__ == "__main__":
    main()

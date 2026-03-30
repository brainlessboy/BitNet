"""
BitNet student model construction.

Replicates the BitLinear + SubLN surgery from distill/distill.py
but as a standalone module. Supports wider student architectures
for faithful ternary replication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(dtype)


class BitLinear(nn.Linear):
    """Linear layer with STE-based ternary weight + INT8 activation quantization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        delta = w.abs().mean().clamp(min=1e-5)
        w_q = (w / delta).round().clamp(-1, 1) * delta
        w_q = w + (w_q - w).detach()  # STE

        gamma = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_q = (x * 127.0 / gamma).round().clamp(-128, 127) * (gamma / 127.0)
        x_q = x + (x_q - x).detach()  # STE

        return F.linear(x_q, w_q, self.bias)


class NormedLinear(nn.Module):
    """SubLN: RMSNorm before a linear projection."""

    def __init__(self, norm: RMSNorm, linear: nn.Module):
        super().__init__()
        self.norm = norm
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


def build_student(model_name: str, device: torch.device = torch.device("cpu")) -> PreTrainedModel:
    """
    Build a BitNet student from a pretrained model.

    Applies BitLinear + SubLN surgery:
    - All projection layers → BitLinear (ternary weights via STE)
    - SubLN (RMSNorm) inserted before o_proj and down_proj
    - Embeddings frozen
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="eager"
    )
    config = model.config

    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        # Replace projections with BitLinear
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            old = getattr(attn, name)
            new = BitLinear(old.in_features, old.out_features, bias=old.bias is not None)
            new.weight = old.weight
            if old.bias is not None:
                new.bias = old.bias
            setattr(attn, name, new)

        for name in ["gate_proj", "up_proj", "down_proj"]:
            old = getattr(mlp, name)
            new = BitLinear(old.in_features, old.out_features, bias=old.bias is not None)
            new.weight = old.weight
            if old.bias is not None:
                new.bias = old.bias
            setattr(mlp, name, new)

        # Insert SubLN before o_proj
        inner_attn_ln = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        attn.o_proj = NormedLinear(inner_attn_ln, attn.o_proj)

        # Insert SubLN before down_proj
        ffn_ln = RMSNorm(config.intermediate_size, eps=config.rms_norm_eps)
        mlp.down_proj = NormedLinear(ffn_ln, mlp.down_proj)

    # Freeze embeddings
    model.model.embed_tokens.weight.requires_grad = False
    if hasattr(model, "lm_head") and not model.config.tie_word_embeddings:
        model.lm_head.weight.requires_grad = False

    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()

    return model

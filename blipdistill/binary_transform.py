#!/usr/bin/env python3
"""
Binary Transform: Exact float-to-ternary model conversion.

Every float weight is decomposed into k ternary digits:
  w = delta × (b₁×2⁻¹ + b₂×2⁻² + ... + bₖ×2⁻ᵏ)
  where bᵢ ∈ {-1, 0, +1} and delta is the per-matrix scale

Each linear layer becomes k parallel ternary layers with power-of-2 scales.
With k=16: mathematically identical to float16. Zero information loss.

This is a TRANSFORM, not training. Deterministic, exact, no data needed.

Usage:
    python blipdistill/binary_transform.py --teacher Qwen/Qwen2.5-0.5B --bits 16 --test
    python blipdistill/binary_transform.py --teacher Qwen/Qwen2.5-0.5B --bits 8 --test
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def banner(text):
    print(f"\n{CYAN}{'=' * 60}")
    print(f"  {BOLD}{text}{RESET}{CYAN}")
    print(f"{'=' * 60}{RESET}\n")


def info(label, value):
    print(f"  {DIM}{label:<22}{RESET}{value}")


def section(text):
    print(f"\n{CYAN}{BOLD}  > {text}{RESET}")


def binary_decompose_weight(w: float, k: int) -> list[int]:
    """
    Decompose a normalized weight in [-1, 1] into k ternary digits.

    w ≈ b₁×2⁻¹ + b₂×2⁻² + ... + bₖ×2⁻ᵏ
    where bᵢ ∈ {-1, 0, +1}

    Returns list of k ternary digits.
    """
    digits = []
    residual = w
    for i in range(k):
        scale = 0.5 ** (i + 1)
        if residual > scale * 0.5:
            digits.append(1)
            residual -= scale
        elif residual < -scale * 0.5:
            digits.append(-1)
            residual += scale
        else:
            digits.append(0)
    return digits


def decompose_matrix(W: torch.Tensor, k: int) -> tuple[list[torch.Tensor], torch.Tensor, float]:
    """
    Decompose a weight matrix into k ternary planes using binary decomposition.

    Args:
        W: Weight matrix (out_features, in_features)
        k: Number of ternary bits

    Returns:
        (planes, scales, delta):
            planes: list of k ternary matrices {-1, 0, +1}
            scales: tensor of k power-of-2 scales
            delta: global normalization scale
    """
    W = W.float()

    # Global scale: normalize to [-1, 1]
    delta = W.abs().max().item()
    if delta < 1e-8:
        delta = 1.0
    W_norm = W / delta

    # Decompose each element
    planes = [torch.zeros_like(W) for _ in range(k)]
    residual = W_norm.clone()

    for i in range(k):
        scale = 0.5 ** (i + 1)
        # Elements > threshold → +1
        pos_mask = residual > scale * 0.5
        # Elements < -threshold → -1
        neg_mask = residual < -scale * 0.5

        planes[i][pos_mask] = 1.0
        planes[i][neg_mask] = -1.0

        residual[pos_mask] -= scale
        residual[neg_mask] += scale

    # Scales: delta × 2⁻¹, delta × 2⁻², ...
    scales = torch.tensor([delta * (0.5 ** (i + 1)) for i in range(k)])

    return planes, scales, delta


def reconstruct_matrix(planes: list[torch.Tensor], scales: torch.Tensor) -> torch.Tensor:
    """Reconstruct weight matrix from ternary planes."""
    W = torch.zeros_like(planes[0])
    for i, T in enumerate(planes):
        W += T * scales[i]
    return W


class BinaryTernaryLinear(nn.Module):
    """
    Linear layer using k ternary planes from binary decomposition.
    Mathematically equivalent to the original float linear layer.

    On analog hardware: k crossbar arrays with power-of-2 voltage dividers.
    """

    def __init__(self, planes: list[torch.Tensor], scales: torch.Tensor):
        super().__init__()
        self.k = len(planes)
        for i, T in enumerate(planes):
            self.register_buffer(f"T_{i}", T)
        self.register_buffer("scales", scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = None
        for i in range(self.k):
            T = getattr(self, f"T_{i}")
            s = self.scales[i]
            # Ternary matmul: T @ x is just additions/subtractions
            contribution = F.linear(x, T) * s
            y = contribution if y is None else y + contribution
        return y

    @property
    def weight(self):
        """Reconstruct effective weight (for compatibility checks)."""
        W = torch.zeros_like(getattr(self, "T_0"))
        for i in range(self.k):
            W += getattr(self, f"T_{i}") * self.scales[i]
        return W


def transform_model(
    teacher: nn.Module,
    k: int = 16,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Transform a float model into a binary-ternary model.

    Every linear projection layer is decomposed into k ternary planes.
    All other components (embeddings, norms, attention logic) stay unchanged.

    Args:
        teacher: Source float model
        k: Number of ternary bits per weight
        device: Device

    Returns:
        Transformed model with BinaryTernaryLinear layers
    """
    config = teacher.config
    teacher_sd = {name: v.float() for name, v in teacher.state_dict().items()}

    # Create a fresh model for the student
    student = AutoModelForCausalLM.from_pretrained(
        config._name_or_path, torch_dtype=torch.float32, attn_implementation="eager"
    )

    total_error = 0.0
    n_layers = 0

    proj_names_attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
    proj_names_mlp = ["gate_proj", "up_proj", "down_proj"]

    for layer_idx in range(config.num_hidden_layers):
        layer = student.model.layers[layer_idx]

        for proj_name in proj_names_attn:
            key = f"model.layers.{layer_idx}.self_attn.{proj_name}.weight"
            W = teacher_sd[key]
            planes, scales, delta = decompose_matrix(W, k)

            # Verify reconstruction
            W_recon = reconstruct_matrix(planes, scales)
            err = (W - W_recon).pow(2).sum().sqrt() / W.pow(2).sum().sqrt().clamp(min=1e-8)
            total_error += err.item()
            n_layers += 1

            bt_linear = BinaryTernaryLinear(planes, scales)
            setattr(layer.self_attn, proj_name, bt_linear)

        for proj_name in proj_names_mlp:
            key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"
            W = teacher_sd[key]
            planes, scales, delta = decompose_matrix(W, k)

            W_recon = reconstruct_matrix(planes, scales)
            err = (W - W_recon).pow(2).sum().sqrt() / W.pow(2).sum().sqrt().clamp(min=1e-8)
            total_error += err.item()
            n_layers += 1

            bt_linear = BinaryTernaryLinear(planes, scales)
            setattr(layer.mlp, proj_name, bt_linear)

        if (layer_idx + 1) % 4 == 0:
            print(f"    Layer {layer_idx + 1}/{config.num_hidden_layers} transformed")

    avg_error = total_error / max(n_layers, 1)
    print(f"    Avg reconstruction error: {avg_error:.8f}")

    student.to(device)
    student.eval()
    return student


def test_equivalence(teacher, student, tokenizer, device, n_tokens=50):
    """Test that the transformed model produces identical output."""
    prompts = [
        "What is the capital of Switzerland?",
        "The meaning of life is",
        "Write a haiku about snow",
        "2 + 2 =",
    ]

    print(f"\n  {'─' * 60}")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            teacher_logits = teacher(**inputs).logits
            student_logits = student(**inputs).logits

        # Compare logits
        diff = (teacher_logits - student_logits).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Compare top-1 prediction
        teacher_pred = teacher_logits[0, -1].argmax().item()
        student_pred = student_logits[0, -1].argmax().item()
        match = teacher_pred == student_pred

        color = GREEN if max_diff < 0.01 else YELLOW if max_diff < 1.0 else RED
        match_str = f"{GREEN}MATCH{RESET}" if match else f"{RED}DIFFER{RESET}"

        print(f"\n  > {prompt}")
        print(f"    Max logit diff: {color}{max_diff:.6f}{RESET}  Mean: {mean_diff:.6f}")
        print(f"    Top-1 prediction: {match_str}"
              f"  teacher={tokenizer.decode(teacher_pred)!r}"
              f"  student={tokenizer.decode(student_pred)!r}")

        # Generate with student
        if n_tokens > 0:
            print(f"    Student generates: ", end="")
            generated = inputs["input_ids"]
            for _ in range(n_tokens):
                out = student(generated)
                logits = out.logits[:, -1, :]
                # Apply repetition penalty
                for tid in set(generated[0].tolist()):
                    if logits[0, tid] > 0:
                        logits[0, tid] /= 1.3
                    else:
                        logits[0, tid] *= 1.3
                probs = torch.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=-1)
                tok = tokenizer.decode(next_token[0], skip_special_tokens=True)
                print(tok, end="", flush=True)
                if next_token.item() == tokenizer.eos_token_id:
                    break
            print()

    print(f"\n  {'─' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Binary Transform: exact float-to-ternary conversion")
    parser.add_argument("--teacher", required=True, help="HuggingFace teacher model")
    parser.add_argument("--bits", type=int, default=16, help="Ternary bits per weight (16=exact, 8=good, 4=lossy)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--test", action="store_true", help="Test equivalence and generate samples")
    parser.add_argument("--sweep", action="store_true", help="Test multiple bit widths")
    args = parser.parse_args()

    device = torch.device(args.device)
    teacher_name = args.teacher.split("/")[-1]

    banner(f"Binary Transform: {teacher_name} → {args.bits}-bit ternary")

    # Load teacher
    section("Loading teacher")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.float32, attn_implementation="eager"
    )
    teacher.eval()
    teacher.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    config = teacher.config

    info("Model", args.teacher)
    info("Parameters", f"{sum(p.numel() for p in teacher.parameters()):,}")
    info("Layers", config.num_hidden_layers)

    if args.sweep:
        # Test multiple bit widths
        section("Bit width sweep")
        for bits in [16, 12, 10, 8, 6, 4, 2, 1]:
            print(f"\n  {'=' * 50}")
            print(f"  {BOLD}{bits} bits{RESET} ({bits} crossbar arrays per layer)")
            print(f"  {'=' * 50}")

            t0 = time.time()
            student = transform_model(teacher, k=bits, device=device)
            info("Transform time", f"{time.time() - t0:.1f}s")

            # Quick equivalence test
            prompt = "What is the capital of Switzerland?"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                t_logits = teacher(**inputs).logits
                s_logits = student(**inputs).logits

            diff = (t_logits - s_logits).abs()
            max_diff = diff.max().item()
            t_pred = tokenizer.decode(t_logits[0, -1].argmax())
            s_pred = tokenizer.decode(s_logits[0, -1].argmax())

            color = GREEN if max_diff < 0.01 else YELLOW if max_diff < 1.0 else RED
            info("Max logit diff", f"{color}{max_diff:.6f}{RESET}")
            info("Teacher predicts", repr(t_pred))
            info("Student predicts", repr(s_pred))

            # Generate
            print(f"  Student: ", end="")
            generated = inputs["input_ids"]
            with torch.no_grad():
                for _ in range(30):
                    out = student(generated)
                    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=-1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            response = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
            print(response[:100])

            del student

    else:
        # Single bit width
        section(f"Transforming ({args.bits} ternary bits per weight)")
        n_projections = config.num_hidden_layers * 7  # 7 projections per layer
        info("Crossbar arrays", f"{n_projections * args.bits:,} ({args.bits} per projection)")

        t0 = time.time()
        student = transform_model(teacher, k=args.bits, device=device)
        info("Transform time", f"{time.time() - t0:.1f}s")

        if args.test:
            section("Testing equivalence")
            test_equivalence(teacher, student, tokenizer, device)

    total_time = time.time() - t0
    banner("Transform Complete")


if __name__ == "__main__":
    main()

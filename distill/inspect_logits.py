#!/usr/bin/env python3
"""
Inspect logits from a BitDistill checkpoint vs the teacher model.

Shows top-10 predicted tokens and entropy for both models side by side.

Usage:
    python distill/inspect_logits.py distill/checkpoints/step_500.pt
    python distill/inspect_logits.py distill/checkpoints/step_500.pt --prompt "Hello, my name is"
"""

import argparse
import importlib.util
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import modify_student
_spec = importlib.util.spec_from_file_location(
    "distill_module", os.path.join(os.path.dirname(os.path.abspath(__file__)), "distill.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
modify_student = _mod.modify_student


def analyze_logits(logits, tokenizer, label):
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, 10)
    entropy = -(probs * (probs + 1e-10).log()).sum()

    print(f"\n  {label}")
    print(f"  {'─' * 40}")
    print(f"  Entropy: {entropy:.2f}  (low=confident, high=uncertain)")
    print(f"  Max prob: {probs.max():.4f}")
    print(f"  Top 10 predictions:")
    for p, idx in zip(topk.values, topk.indices):
        token_str = repr(tokenizer.decode(idx))
        print(f"    {p:.4f}  {token_str}")


def main():
    parser = argparse.ArgumentParser(description="Inspect logits from student vs teacher")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--prompt", "-p", default="What is the capital of Switzerland?")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = ckpt["config"]._name_or_path
    step = ckpt.get("step", "?")
    print(f"Base model: {model_name}, step: {step}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load teacher
    print("Loading teacher (original model)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="eager"
    )
    teacher.eval()

    # Load student
    print("Loading student (BitLinear + trained)...")
    student = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="eager"
    )
    student = modify_student(student)
    student.load_state_dict(ckpt["model_state_dict"])
    student.eval()
    del ckpt

    # Also test BitLinear with no training (step 0 baseline)
    print("Loading student (BitLinear + NO training)...")
    baseline = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="eager"
    )
    baseline = modify_student(baseline)
    baseline.eval()

    prompts = [
        args.prompt,
        "The meaning of life is",
        "2 + 2 =",
    ]

    for prompt in prompts:
        print(f"\n{'=' * 50}")
        print(f"  Prompt: {repr(prompt)}")
        print(f"{'=' * 50}")

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            t_out = teacher(**inputs)
            s_out = student(**inputs)
            b_out = baseline(**inputs)

        # Analyze last token position
        analyze_logits(t_out.logits[0, -1], tokenizer, "TEACHER (original Qwen2.5)")
        analyze_logits(b_out.logits[0, -1], tokenizer, "STUDENT step 0 (BitLinear, no training)")
        analyze_logits(s_out.logits[0, -1], tokenizer, f"STUDENT step {step} (BitLinear, trained)")

        # KL divergence between teacher and student
        t_probs = torch.softmax(t_out.logits[0, -1], dim=-1)
        s_probs = torch.softmax(s_out.logits[0, -1], dim=-1)
        b_probs = torch.softmax(b_out.logits[0, -1], dim=-1)
        kl_trained = (t_probs * (t_probs / (s_probs + 1e-10)).log()).sum()
        kl_baseline = (t_probs * (t_probs / (b_probs + 1e-10)).log()).sum()
        print(f"\n  KL(teacher || step 0):    {kl_baseline:.2f}")
        print(f"  KL(teacher || step {step}): {kl_trained:.2f}")
        print(f"  {'Improved!' if kl_trained < kl_baseline else 'No improvement'}")


if __name__ == "__main__":
    main()

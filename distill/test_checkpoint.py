#!/usr/bin/env python3
"""
Test a BitDistill checkpoint directly in PyTorch (no GGUF conversion).

Usage:
    python distill/test_checkpoint.py distill/checkpoints/step_6500.pt
    python distill/test_checkpoint.py distill/checkpoints/step_6500.pt --prompt "Explain quantum computing"
"""

import argparse
import sys
import os

import importlib.util

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import modify_student directly to avoid package conflicts
_spec = importlib.util.spec_from_file_location(
    "distill_module", os.path.join(os.path.dirname(os.path.abspath(__file__)), "distill.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
modify_student = _mod.modify_student


def main():
    parser = argparse.ArgumentParser(description="Test a BitDistill checkpoint")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint (or 'base:MODEL' to test unmodified base)")
    parser.add_argument("--prompt", "-p", default="What is the capital of Switzerland?")
    parser.add_argument("--max_tokens", "-n", type=int, default=100)
    parser.add_argument("--temperature", "-t", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", "-r", type=float, default=1.0)
    args = parser.parse_args()

    if args.checkpoint.startswith("base:"):
        # Test unmodified base model (no BitLinear, no training)
        model_name = args.checkpoint.split(":", 1)[1]
        print(f"Loading BASE model (no BitLinear): {model_name}")
        student = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, attn_implementation="eager"
        )
        student.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("This is the unmodified teacher — should produce coherent text.")
    elif args.checkpoint.startswith("bitlinear:"):
        # Test base model WITH BitLinear but NO training (step 0)
        model_name = args.checkpoint.split(":", 1)[1]
        print(f"Loading model with BitLinear surgery (NO training): {model_name}")
        student = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, attn_implementation="eager"
        )
        student = modify_student(student)
        student.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("This shows the quality BEFORE distillation — expect gibberish.")
    else:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        model_name = config._name_or_path
        print(f"Base model: {model_name}")
        print(f"Step: {ckpt.get('step', '?')}")

        print("Loading base model + BitLinear surgery...")
        student = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, attn_implementation="eager"
        )
        student = modify_student(student)
        student.load_state_dict(ckpt["model_state_dict"])
        student.eval()
        del ckpt

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = [
        args.prompt,
        "What is 2 + 2?",
        "The meaning of life is",
        "Write a haiku about snow",
    ]

    print(f"\n{'=' * 60}")
    for prompt in prompts:
        print(f"\n> {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            generated = inputs["input_ids"]
            for i in range(args.max_tokens):
                out = student(generated)
                next_token_logits = out.logits[:, -1, :]
                # Apply repetition penalty
                if args.repetition_penalty != 1.0:
                    for token_id in set(generated[0].tolist()):
                        if next_token_logits[0, token_id] > 0:
                            next_token_logits[0, token_id] /= args.repetition_penalty
                        else:
                            next_token_logits[0, token_id] *= args.repetition_penalty
                if args.temperature > 0:
                    probs = torch.softmax(next_token_logits / args.temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
                print(token_str, end="", flush=True)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        print(f"\n{'-' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
BLIP-8 Chat: Interactive chat with a BLIP-8 ternary model.

Loads a teacher model, transforms it to BLIP-8 (8 ternary planes),
and provides an interactive chat interface.

Usage:
    python blipdistill/chat.py --teacher Qwen/Qwen2.5-0.5B
    python blipdistill/chat.py --teacher Qwen/Qwen2.5-3B --device cuda
    python blipdistill/chat.py --teacher Qwen/Qwen2.5-0.5B --bits 6
"""

import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from blipdistill.binary_transform import decompose_matrix, reconstruct_matrix


DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def build_blip8_model(teacher, k=8):
    """Transform teacher into BLIP-8 model (reconstructed weights in nn.Linear)."""
    student = copy.deepcopy(teacher).cpu().float()
    config = teacher.config
    teacher_sd = {name: v.float().cpu() for name, v in teacher.state_dict().items()}

    for li in range(config.num_hidden_layers):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            key = f"model.layers.{li}.self_attn.{proj}.weight"
            W = teacher_sd[key]
            planes, scales, _ = decompose_matrix(W, k=k)
            getattr(student.model.layers[li].self_attn, proj).weight.data = reconstruct_matrix(planes, scales)

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            key = f"model.layers.{li}.mlp.{proj}.weight"
            W = teacher_sd[key]
            planes, scales, _ = decompose_matrix(W, k=k)
            getattr(student.model.layers[li].mlp, proj).weight.data = reconstruct_matrix(planes, scales)

    student.eval()
    return student


def chat_loop(model, tokenizer, device, max_tokens=200, temperature=0.7, repetition_penalty=1.2):
    """Interactive chat loop."""
    print(f"\n{CYAN}{'─' * 60}{RESET}")
    print(f"  {BOLD}BLIP-8 Chat{RESET}")
    print(f"  {DIM}Type your message and press Enter. Type 'quit' to exit.{RESET}")
    print(f"  {DIM}Commands: /clear, /temp <value>, /max <tokens>{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}\n")

    history = []

    while True:
        try:
            user_input = input(f"{GREEN}> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Bye!")
            break

        if user_input.startswith("/clear"):
            history.clear()
            print(f"  {DIM}History cleared.{RESET}")
            continue

        if user_input.startswith("/temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"  {DIM}Temperature: {temperature}{RESET}")
            except (ValueError, IndexError):
                print(f"  {DIM}Usage: /temp 0.7{RESET}")
            continue

        if user_input.startswith("/max "):
            try:
                max_tokens = int(user_input.split()[1])
                print(f"  {DIM}Max tokens: {max_tokens}{RESET}")
            except (ValueError, IndexError):
                print(f"  {DIM}Usage: /max 200{RESET}")
            continue

        # Build prompt
        history.append({"role": "user", "content": user_input})

        # Format as simple prompt (works with base models)
        prompt = ""
        for msg in history:
            if msg["role"] == "user":
                prompt += f"### User:\n{msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"### Assistant:\n{msg['content']}\n\n"
        prompt += "### Assistant:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated = inputs["input_ids"]

        # Generate
        print(f"{CYAN}", end="")
        response_tokens = []
        t0 = time.time()

        with torch.no_grad():
            for i in range(max_tokens):
                out = model(generated)
                logits = out.logits[:, -1, :].float()

                # Repetition penalty
                if repetition_penalty != 1.0:
                    for tid in set(generated[0][-100:].tolist()):
                        if logits[0, tid] > 0:
                            logits[0, tid] /= repetition_penalty
                        else:
                            logits[0, tid] *= repetition_penalty

                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=-1)
                tok_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
                print(tok_str, end="", flush=True)
                response_tokens.append(next_token.item())

                if next_token.item() == tokenizer.eos_token_id:
                    break

                # Stop at "### User:" (model trying to generate user turn)
                decoded_so_far = tokenizer.decode(response_tokens, skip_special_tokens=True)
                if "### User:" in decoded_so_far or "### user:" in decoded_so_far.lower():
                    break

        elapsed = time.time() - t0
        tps = len(response_tokens) / elapsed if elapsed > 0 else 0

        print(f"{RESET}")
        print(f"  {DIM}{len(response_tokens)} tokens, {tps:.1f} tok/s{RESET}\n")

        # Add to history
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        response = response.split("### User:")[0].strip()
        history.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(description="BLIP-8 Chat")
    parser.add_argument("--teacher", required=True, help="HuggingFace teacher model")
    parser.add_argument("--bits", type=int, default=8, help="Ternary planes (8=perfect)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)
    teacher_name = args.teacher.split("/")[-1]

    print(f"\n{CYAN}{'=' * 60}")
    print(f"  {BOLD}BLIP-8: {teacher_name} ({args.bits} ternary planes){RESET}{CYAN}")
    print(f"{'=' * 60}{RESET}")

    print(f"\n  {DIM}Loading teacher model...{RESET}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.float32, attn_implementation="eager"
    )
    teacher.eval()
    teacher.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)

    print(f"  {DIM}Transforming to BLIP-{args.bits}...{RESET}")
    t0 = time.time()
    model = build_blip8_model(teacher, k=args.bits)
    model.to(device)
    del teacher
    print(f"  {DIM}Ready in {time.time() - t0:.1f}s{RESET}")

    chat_loop(model, tokenizer, device, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()

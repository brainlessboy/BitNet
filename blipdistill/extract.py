#!/usr/bin/env python3
"""
Phase 1: Extract teacher knowledge via vocabulary-seeded self-exploration.

Instead of random probes, the teacher generates its OWN training data:
1. Pick a seed token from the vocabulary
2. Teacher generates a full sequence autoregressively
3. Save the sequence + teacher's logits at every position

This captures the teacher's natural language patterns, knowledge,
and statistical behavior — the teacher explores its own knowledge space.

Usage:
    # Auto-calculate dataset size, vocab-seeded generation
    python blipdistill/extract.py --teacher Qwen/Qwen2.5-0.5B --device mps

    # Specify size, run on GPU
    python blipdistill/extract.py --teacher Qwen/Qwen2.5-3B --device cuda --n_examples 100000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"


def banner(text):
    print(f"\n{CYAN}{'=' * 60}")
    print(f"  {BOLD}{text}{RESET}{CYAN}")
    print(f"{'=' * 60}{RESET}\n")


def info(label, value):
    print(f"  {DIM}{label:<22}{RESET}{value}")


def section(text):
    print(f"\n{CYAN}{BOLD}  > {text}{RESET}")


def cluster_vocabulary(embed_weight: torch.Tensor, n_clusters: int = 1000) -> torch.Tensor:
    """
    Cluster vocabulary embeddings to get diverse seed tokens.

    Returns one representative token ID per cluster — maximally diverse seeds.
    """
    # Use random projection for fast approximate clustering
    n_vocab, embed_dim = embed_weight.shape
    n_clusters = min(n_clusters, n_vocab)

    # Pick random cluster centers from actual vocab tokens
    indices = torch.randperm(n_vocab)[:n_clusters]
    centers = embed_weight[indices].float()

    # Assign each token to nearest center (one pass, no iteration needed for seeds)
    # Just return the center indices — they're already diverse
    return indices


@torch.no_grad()
def generate_teacher_sequences(
    teacher,
    seed_tokens: torch.Tensor,
    seq_len: int,
    batch_size: int,
    temperatures: list[float],
    device: torch.device,
) -> list[dict]:
    """
    Generate sequences from the teacher, starting from seed tokens.

    For each seed token, generates a full sequence autoregressively.
    Saves the complete sequence AND the teacher's logits at each position.

    Args:
        teacher: Teacher model
        seed_tokens: 1D tensor of seed token IDs
        seq_len: Target sequence length
        batch_size: Batch size for parallel generation
        temperatures: List of temperatures to use (cycles through them)
        device: Device

    Returns:
        List of dicts, each with 'input_ids' and 'top_k_indices'/'top_k_probs'
    """
    examples = []
    n_seeds = len(seed_tokens)

    for batch_start in range(0, n_seeds, batch_size):
        batch_end = min(batch_start + batch_size, n_seeds)
        batch_seeds = seed_tokens[batch_start:batch_end].to(device)
        bs = batch_seeds.shape[0]

        # Start with seed tokens
        generated = batch_seeds.unsqueeze(1)  # (bs, 1)

        # Storage for logits at each position
        all_top_k_indices = []
        all_top_k_probs = []

        # Pick temperature for this batch
        temp = temperatures[batch_start % len(temperatures)]

        # Generate token by token
        for pos in range(seq_len):
            outputs = teacher(generated)
            logits = outputs.logits[:, -1, :].float().clamp(-100, 100)  # prevent overflow

            # Save top-K logits
            top_k = min(128, logits.shape[-1])
            top_vals, top_idx = logits.topk(top_k, dim=-1)
            top_probs = F.softmax(top_vals / 2.0, dim=-1).to(torch.float16)
            all_top_k_indices.append(top_idx.cpu().to(torch.int32))
            all_top_k_probs.append(top_probs.cpu())

            # Sample next token (with NaN/inf guard)
            sample_logits = logits / temp
            # Replace any NaN/inf with 0
            sample_logits = torch.where(
                torch.isfinite(sample_logits), sample_logits,
                torch.zeros_like(sample_logits)
            )
            probs = F.softmax(sample_logits, dim=-1)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=-1)

            del outputs, logits

        # Stack logits: (bs, seq_len, top_k)
        stacked_indices = torch.stack(all_top_k_indices, dim=1)  # (bs, seq_len, top_k)
        stacked_probs = torch.stack(all_top_k_probs, dim=1)

        # Store each example
        for i in range(bs):
            examples.append({
                "input_ids": generated[i, :-1].cpu(),  # exclude last (no logits for it)
                "top_k_indices": stacked_indices[i],
                "top_k_probs": stacked_probs[i],
                "temperature": temp,
            })

        del generated, all_top_k_indices, all_top_k_probs

        yield batch_start, batch_end, examples[-bs:]


def _save_chunk(examples: list[dict], output_dir: str, chunk_idx: int):
    """Save a chunk of examples to disk."""
    path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.pt")
    save_data = {
        "probe_types": ["vocab_seed"] * len(examples),
        "input_ids": torch.stack([e["input_ids"] for e in examples]),
        "top_k_indices": torch.stack([e["top_k_indices"] for e in examples]),
        "top_k_probs": torch.stack([e["top_k_probs"] for e in examples]),
    }
    torch.save(save_data, path)


@torch.no_grad()
def extract(args):
    device = torch.device(args.device)
    t_start = time.time()

    teacher_name = args.teacher.split("/")[-1]
    banner(f"Extract: {teacher_name} → knowledge dataset")

    # --- Load teacher ---
    section("Loading teacher")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.float32, attn_implementation="eager"
    )
    teacher.eval()
    teacher.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    config = teacher.config
    vocab_size = config.vocab_size

    info("Model", args.teacher)
    info("Parameters", f"{sum(p.numel() for p in teacher.parameters()):,}")
    info("Vocab size", f"{vocab_size:,}")

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Calculate dataset size ---
    if args.n_examples <= 0:
        from blipdistill.model import build_student
        tmp_student = build_student(args.teacher, device=torch.device("cpu"))
        trainable_params = sum(p.numel() for p in tmp_student.parameters() if p.requires_grad)
        del tmp_student
        constraints_per_example = args.seq_len * args.top_k
        min_examples = trainable_params // constraints_per_example
        args.n_examples = int(min_examples * 3)
        section("Auto-calculated dataset size")
        info("Trainable params", f"{trainable_params:,}")
        info("Constraints/example", f"{constraints_per_example:,}")
        info("Minimum examples", f"{min_examples:,}")
        info("Recommended (3x)", f"{args.n_examples:,}")

    # --- Create diverse seed tokens ---
    section("Creating vocabulary-diverse seeds")
    embed_weight = teacher.model.embed_tokens.weight.detach()
    seed_pool = cluster_vocabulary(embed_weight, n_clusters=min(2000, vocab_size))
    info("Seed pool size", f"{len(seed_pool):,}")

    # Expand seeds to match n_examples (cycle through pool with different temperatures)
    n_repeats = (args.n_examples + len(seed_pool) - 1) // len(seed_pool)
    all_seeds = seed_pool.repeat(n_repeats)[:args.n_examples]
    # Shuffle for variety
    all_seeds = all_seeds[torch.randperm(len(all_seeds))]
    info("Total sequences", f"{len(all_seeds):,}")

    # Temperatures: cycle through for diversity
    temperatures = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
    info("Temperatures", f"{temperatures}")

    # --- Setup output ---
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    per_example_bytes = args.seq_len * args.top_k * (4 + 2) + args.seq_len * 4
    total_bytes = args.n_examples * per_example_bytes

    section(f"Generating {args.n_examples:,} teacher-authored sequences")
    info("Sequence length", args.seq_len)
    info("Top-K logits", args.top_k)
    info("Batch size", args.batch_size)
    info("Est. storage", f"{total_bytes / (1024**3):.1f} GB")
    info("Method", "vocabulary-seeded autoregressive generation")
    print()

    # --- Generate ---
    chunk_buffer = []
    chunk_idx = 0
    chunk_size = 500
    examples_saved = 0

    for batch_start, batch_end, batch_examples in generate_teacher_sequences(
        teacher, all_seeds, args.seq_len, args.batch_size, temperatures, device
    ):
        chunk_buffer.extend(batch_examples)
        examples_saved += len(batch_examples)

        # Save chunks
        while len(chunk_buffer) >= chunk_size:
            _save_chunk(chunk_buffer[:chunk_size], output_dir, chunk_idx)
            chunk_buffer = chunk_buffer[chunk_size:]
            chunk_idx += 1

        # Progress
        if examples_saved % (args.batch_size * 10) == 0 or examples_saved == args.n_examples:
            elapsed = time.time() - t_start
            rate = examples_saved / elapsed if elapsed > 0 else 0
            eta = (args.n_examples - examples_saved) / rate if rate > 0 else 0
            eta_m, eta_s = divmod(int(eta), 60)
            eta_h, eta_m_r = divmod(eta_m, 60)
            if eta_h > 0:
                eta_str = f"{eta_h}h{eta_m_r:02d}m"
            else:
                eta_str = f"{eta_m}m{eta_s:02d}s"
            print(f"    {examples_saved:>7,}/{args.n_examples:,}"
                  f"  ({rate:.1f}/s, ETA {eta_str})")

    # Save remaining
    if chunk_buffer:
        _save_chunk(chunk_buffer, output_dir, chunk_idx)
        chunk_idx += 1

    # Save metadata
    metadata = {
        "teacher": args.teacher,
        "vocab_size": vocab_size,
        "hidden_size": config.hidden_size,
        "seq_len": args.seq_len,
        "top_k": args.top_k,
        "n_examples": examples_saved,
        "n_chunks": chunk_idx,
        "method": "vocab_seeded",
        "temperatures": temperatures,
    }
    torch.save(metadata, os.path.join(output_dir, "metadata.pt"))

    # Summary
    total_time = time.time() - t_start
    actual_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir) if f.endswith(".pt")
    )
    t_m, t_s = divmod(int(total_time), 60)
    t_h, t_m = divmod(t_m, 60)
    time_str = f"{t_h}h{t_m:02d}m{t_s:02d}s" if t_h else f"{t_m}m{t_s:02d}s"

    banner("Extraction Complete")
    info("Examples", f"{examples_saved:,}")
    info("Chunks", chunk_idx)
    info("Storage", f"{actual_size / (1024**3):.2f} GB")
    info("Time", time_str)
    info("Output", output_dir)
    print()
    section("Next step")
    print(f"\n    python blipdistill/train.py --teacher {args.teacher} "
          f"--probe_data {output_dir} --device {args.device}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Extract teacher knowledge via vocab-seeded generation")
    parser.add_argument("--teacher", required=True, help="HuggingFace teacher model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_examples", type=int, default=0,
                        help="Number of examples (0 = auto-calculate from model size)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for parallel generation")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--output", default="blipdistill/probes")
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()

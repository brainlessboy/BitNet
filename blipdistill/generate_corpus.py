#!/usr/bin/env python3
"""
Phase 1: Teacher generates its own training corpus.

The teacher writes text — this IS its knowledge made manifest.
We save only the token IDs (tiny: ~100MB for 100K sequences).
During training, logits are computed live (full vocab, no sparse approx).

Strategy for diverse, knowledge-covering generation:
1. Cluster vocabulary into ~2000 topic seeds
2. For each seed, teacher generates a sequence autoregressively
3. Multiple temperatures ensure coverage of confident + uncertain knowledge
4. Result: a corpus that comprehensively represents what the teacher knows

Usage:
    python blipdistill/generate_corpus.py --teacher Qwen/Qwen2.5-0.5B --device cuda
    python blipdistill/generate_corpus.py --teacher Qwen/Qwen2.5-3B --device cuda --n_sequences 200000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
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


def get_diverse_seeds(teacher, n_seeds: int = 2000) -> torch.Tensor:
    """
    Get diverse seed tokens by clustering the vocabulary.
    Returns token IDs that are maximally spread across the embedding space.
    """
    embed = teacher.model.embed_tokens.weight.float().detach().cpu()
    n_vocab = embed.shape[0]
    n_seeds = min(n_seeds, n_vocab)

    # Simple but effective: random subset of vocab
    # Each token represents a different concept/entry point
    indices = torch.randperm(n_vocab)[:n_seeds]
    return indices


@torch.no_grad()
def generate_batch(
    teacher,
    seed_tokens: torch.Tensor,
    seq_len: int,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate sequences from seed tokens using batched generation with KV cache.

    Returns: (batch_size, seq_len) tensor of token IDs
    """
    bs = seed_tokens.shape[0]
    generated = seed_tokens.unsqueeze(1).to(device)  # (bs, 1)

    past_key_values = None
    for pos in range(seq_len - 1):
        if past_key_values is None:
            outputs = teacher(generated, use_cache=True)
        else:
            outputs = teacher(generated[:, -1:], past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()

        # Clamp for numerical stability
        logits = logits.clamp(-100, 100)

        # Handle NaN
        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))

        # Sample
        probs = torch.softmax(logits / temperature, dim=-1)
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(probs, 1)

        generated = torch.cat([generated, next_token], dim=-1)

    return generated  # (bs, seq_len)


@torch.no_grad()
def generate_corpus(args):
    device = torch.device(args.device)
    t_start = time.time()

    teacher_name = args.teacher.split("/")[-1]
    banner(f"Generate Corpus: {teacher_name}")

    # Load teacher
    section("Loading teacher")
    print(f"  {DIM}Downloading/loading model weights...{RESET}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.float16, attn_implementation="eager"
    )
    teacher.eval()
    teacher.to(device)
    print(f"  {DIM}Model loaded and moved to {args.device}{RESET}")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    config = teacher.config

    info("Model", args.teacher)
    info("Parameters", f"{sum(p.numel() for p in teacher.parameters()):,}")
    info("Vocab size", f"{config.vocab_size:,}")

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        info("TF32", "enabled")

    # Auto-calculate corpus size if needed
    if args.n_sequences <= 0:
        h = config.hidden_size
        i = config.intermediate_size
        n = config.num_hidden_layers
        kv = config.num_key_value_heads
        heads = config.num_attention_heads
        head_dim = h // heads
        per_layer = h*h + kv*head_dim*h + kv*head_dim*h + h*h + i*h + i*h + h*i
        trainable_params = per_layer * n
        # Rule of thumb: ~10 tokens per parameter for good coverage
        total_tokens_needed = trainable_params * 10
        args.n_sequences = max(10000, total_tokens_needed // args.seq_len)
        section("Auto-calculated corpus size")
        info("Trainable params", f"{trainable_params:,}")
        info("Tokens needed", f"{total_tokens_needed:,}")
        info("Sequences", f"{args.n_sequences:,}")

    # Get diverse seeds
    section("Creating diverse seed tokens")
    seed_pool = get_diverse_seeds(teacher, n_seeds=min(5000, config.vocab_size))
    info("Seed pool", f"{len(seed_pool):,} diverse tokens")

    # Expand seeds to cover all sequences
    n_repeats = (args.n_sequences + len(seed_pool) - 1) // len(seed_pool)
    all_seeds = seed_pool.repeat(n_repeats)[:args.n_sequences]
    all_seeds = all_seeds[torch.randperm(len(all_seeds))]

    # Temperatures for diversity
    temperatures = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2]

    # Setup output
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Estimate storage
    storage_bytes = args.n_sequences * args.seq_len * 4  # int32 token IDs
    info("Est. storage", f"{storage_bytes / (1024**2):.0f} MB")

    section(f"Generating {args.n_sequences:,} sequences (len={args.seq_len})")
    info("Batch size", args.batch_size)
    info("Temperatures", str(temperatures))
    print()

    # Generate in batches, save in chunks
    chunk_size = 1000
    chunk_buffer = []
    chunk_idx = 0
    total_generated = 0

    for batch_start in range(0, args.n_sequences, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.n_sequences)
        bs = batch_end - batch_start

        batch_seeds = all_seeds[batch_start:batch_end]
        temp = temperatures[(batch_start // args.batch_size) % len(temperatures)]

        sequences = generate_batch(teacher, batch_seeds, args.seq_len, temp, device)
        chunk_buffer.append(sequences.cpu())
        total_generated += bs

        # Save chunks
        total_in_buffer = sum(s.shape[0] for s in chunk_buffer)
        if total_in_buffer >= chunk_size:
            all_seqs = torch.cat(chunk_buffer, dim=0)
            while all_seqs.shape[0] >= chunk_size:
                chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.pt")
                torch.save({"sequences": all_seqs[:chunk_size]}, chunk_path)
                all_seqs = all_seqs[chunk_size:]
                chunk_idx += 1
            chunk_buffer = [all_seqs] if all_seqs.shape[0] > 0 else []

        # Progress
        if total_generated % (args.batch_size * 5) == 0:
            elapsed = time.time() - t_start
            rate = total_generated / elapsed if elapsed > 0 else 0
            eta = (args.n_sequences - total_generated) / rate if rate > 0 else 0
            eta_m, eta_s = divmod(int(eta), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_s:02d}s"
            print(f"    {total_generated:>8,}/{args.n_sequences:,}"
                  f"  ({rate:.1f} seq/s, temp={temp:.1f}, ETA {eta_str})")

    # Save remaining
    if chunk_buffer:
        remaining = torch.cat(chunk_buffer, dim=0)
        if remaining.shape[0] > 0:
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.pt")
            torch.save({"sequences": remaining}, chunk_path)
            chunk_idx += 1

    # Save metadata
    metadata = {
        "teacher": args.teacher,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "seq_len": args.seq_len,
        "n_sequences": total_generated,
        "n_chunks": chunk_idx,
        "temperatures": temperatures,
        "method": "teacher_generated_corpus",
    }
    torch.save(metadata, os.path.join(output_dir, "metadata.pt"))

    # Summary
    total_time = time.time() - t_start
    actual_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir) if f.endswith(".pt")
    )
    t_m, t_s = divmod(int(total_time), 60)

    banner("Corpus Generation Complete")
    info("Sequences", f"{total_generated:,}")
    info("Total tokens", f"{total_generated * args.seq_len:,}")
    info("Chunks", chunk_idx)
    info("Storage", f"{actual_size / (1024**2):.0f} MB")
    info("Time", f"{t_m}m {t_s:02d}s")
    info("Output", output_dir)
    print()

    # Sample preview
    section("Sample sequences")
    sample_chunk = torch.load(os.path.join(output_dir, "chunk_0000.pt"), weights_only=False)
    for i in range(min(3, sample_chunk["sequences"].shape[0])):
        text = tokenizer.decode(sample_chunk["sequences"][i][:60], skip_special_tokens=True)
        print(f"  {DIM}[{i}]{RESET} {text[:100]}...")
    print()

    section("Next step")
    print(f"\n    python blipdistill/train.py --teacher {args.teacher}"
          f" --corpus {output_dir} --device {args.device}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate teacher training corpus")
    parser.add_argument("--teacher", required=True, help="HuggingFace teacher model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_sequences", type=int, default=0,
                        help="Number of sequences (0 = auto-calculate)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--output", default="blipdistill/corpus")
    args = parser.parse_args()
    generate_corpus(args)


if __name__ == "__main__":
    main()

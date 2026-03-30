#!/usr/bin/env python3
"""
Phase 2: Full-KL distillation on teacher-generated corpus.

Both teacher and student are in memory. The teacher's OWN text corpus
(from generate_corpus.py) provides structured, knowledge-rich inputs.
Full-vocab KL divergence ensures accurate gradient signal.

This is the approach proven to work (Alpaca distillation produced coherent
text), but with teacher-generated data instead of external datasets.

Usage:
    python blipdistill/train.py --teacher Qwen/Qwen2.5-0.5B --corpus blipdistill/corpus --device cuda
    python blipdistill/train.py --teacher Qwen/Qwen2.5-3B --corpus blipdistill/corpus_3b --device cuda --max_steps 10000
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from blipdistill.model import build_student


# Pretty output
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def banner(text):
    print(f"\n{CYAN}{'=' * 60}")
    print(f"  {BOLD}{text}{RESET}{CYAN}")
    print(f"{'=' * 60}{RESET}\n")


def info(label, value):
    print(f"  {DIM}{label:<22}{RESET}{value}")


def section(text):
    print(f"\n{CYAN}{BOLD}  > {text}{RESET}")


def empty_cache(device_str):
    if device_str == "cuda":
        torch.cuda.empty_cache()
    elif device_str == "mps":
        torch.mps.empty_cache()


class CorpusDataset:
    """Loads teacher-generated text corpus from disk."""

    def __init__(self, corpus_dir: str, device: torch.device):
        self.device = device
        metadata = torch.load(os.path.join(corpus_dir, "metadata.pt"), weights_only=False)
        self.seq_len = metadata["seq_len"]
        self.n_sequences = metadata["n_sequences"]

        # Load all chunks into one big tensor
        chunk_files = sorted(f for f in os.listdir(corpus_dir) if f.startswith("chunk_"))
        all_seqs = []
        for cf in chunk_files:
            chunk = torch.load(os.path.join(corpus_dir, cf), weights_only=False)
            all_seqs.append(chunk["sequences"])

        self.sequences = torch.cat(all_seqs, dim=0)  # (n_sequences, seq_len)
        print(f"  Loaded {self.sequences.shape[0]:,} sequences ({self.seq_len} tokens each)")

    def get_batch(self, batch_size: int) -> torch.Tensor:
        """Get a random batch of token sequences."""
        indices = torch.randint(0, self.sequences.shape[0], (batch_size,))
        return self.sequences[indices].to(self.device)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tau: float = 2.0,
) -> tuple[torch.Tensor, float]:
    """
    Full-vocab KL divergence loss.

    This is the PROVEN loss function — same as distill/distill.py.
    No sparse approximation, no shortcuts.
    """
    student_logits = student_logits.float().clamp(-100, 100)
    teacher_logits = teacher_logits.float().clamp(-100, 100)

    student_soft = F.log_softmax(student_logits / tau, dim=-1)
    teacher_soft = F.softmax(teacher_logits / tau, dim=-1)

    loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (tau ** 2)

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=loss.device, requires_grad=True), 0.0

    return loss, loss.item()


def train(args):
    device = torch.device(args.device)
    t_start = time.time()

    teacher_name = args.teacher.split("/")[-1]
    banner(f"BLIPDistill: {teacher_name} → 1.58-bit BitNet")

    # --- Load teacher ---
    section("Loading teacher")
    print(f"  {DIM}Downloading/loading model weights...{RESET}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.float16, attn_implementation="eager"
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.to(device)
    print(f"  {DIM}Teacher on {args.device}{RESET}")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    config = teacher.config

    info("Model", args.teacher)
    info("Parameters", f"{sum(p.numel() for p in teacher.parameters()):,}")
    info("Vocab size", f"{config.vocab_size:,}")

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        info("TF32", "enabled")

    # --- Load corpus ---
    section("Loading teacher-generated corpus")
    corpus = CorpusDataset(args.corpus, device)

    # --- Build student ---
    section("Building student (BitLinear + SubLN)")
    print(f"  {DIM}Loading and modifying model...{RESET}")
    student = build_student(
        args.teacher, device=device,
        use_gradient_checkpointing=(args.device != "cuda"),
    )

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    info("Total parameters", f"{total_params:,}")
    info("Trainable", f"{trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    total_steps = args.max_steps
    warmup_steps = min(200, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Output setup ---
    output_dir = args.output_dir
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "training_log.jsonl")
    with open(log_path, "w") as f:
        f.write(json.dumps({
            "type": "header",
            "total_steps": total_steps,
            "mode": "teacher_corpus",
            "teacher": args.teacher,
        }) + "\n")

    # --- Training ---
    section(f"Training ({total_steps} steps)")
    effective_batch = args.batch_size * args.accumulation_steps
    info("Batch size", args.batch_size)
    info("Accumulation", args.accumulation_steps)
    info("Effective batch", effective_batch)
    info("Sequence length", corpus.seq_len)
    info("Temperature", args.tau)
    info("Learning rate", f"{args.lr} (cosine decay)")
    info("Warmup", f"{warmup_steps} steps")
    info("Data source", "teacher-generated corpus (full-KL)")
    print()

    best_loss = float("inf")
    accum_loss = 0.0
    global_step = 0
    optimizer.zero_grad()

    for step_idx in range(total_steps * args.accumulation_steps):
        accum_pos = (step_idx % args.accumulation_steps) + 1

        # Get batch from corpus
        input_ids = corpus.get_batch(args.batch_size)

        # Teacher forward
        with torch.no_grad():
            teacher_out = teacher(input_ids)
            teacher_logits = teacher_out.logits.float().detach()
            del teacher_out
        empty_cache(args.device)

        # Student forward
        student_out = student(input_ids=input_ids)

        # Full-vocab KL loss
        loss, ld = distillation_loss(student_out.logits, teacher_logits, tau=args.tau)
        loss = loss / args.accumulation_steps

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            accum_loss = 0.0
            del student_out, teacher_logits
            empty_cache(args.device)
            continue

        loss.backward()
        accum_loss += loss.item()

        del student_out, teacher_logits

        # Optimizer step
        if accum_pos == args.accumulation_steps:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            avg_loss = accum_loss
            if avg_loss < best_loss:
                best_loss = avg_loss

            elapsed = time.time() - t_start
            sps = global_step / elapsed if elapsed > 0 else 0
            eta = (total_steps - global_step) / sps if sps > 0 else 0
            lr_now = scheduler.get_last_lr()[0]

            eta_m, eta_s = divmod(int(eta), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_s:02d}s"

            print(f"  {GREEN}{BOLD}Step {global_step}/{total_steps}{RESET}"
                  f"  {DIM}|{RESET} loss {YELLOW}{avg_loss:.2f}{RESET}"
                  f"  {DIM}|{RESET} lr {lr_now:.1e}"
                  f"  {DIM}ETA {eta_str}{RESET}")

            # Log
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "type": "step", "step": global_step,
                    "loss": avg_loss, "ce": 0.0, "ld": avg_loss, "ad": 0.0,
                    "lr": lr_now, "elapsed": elapsed, "eta": eta,
                }) + "\n")

            accum_loss = 0.0

            # Save checkpoint
            if global_step % args.save_every == 0:
                save_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                print(f"    {DIM}Saving...{RESET}", end="", flush=True)
                torch.save({
                    "step": global_step,
                    "model_state_dict": student.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": student.config,
                }, save_path)
                size_mb = os.path.getsize(save_path) / (1024 * 1024)
                print(f"\r    {GREEN}Saved{RESET} {save_path} {DIM}({size_mb:.0f} MB){RESET}")

                # Keep only last 2
                existing = sorted(
                    [f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pt")],
                    key=lambda f: int(f.split("_")[1].split(".")[0]),
                )
                for old in existing[:-2]:
                    os.remove(os.path.join(ckpt_dir, old))

        if step_idx % 50 == 0:
            empty_cache(args.device)

    # Save final
    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        "step": global_step,
        "model_state_dict": student.state_dict(),
        "config": student.config,
    }, final_path)

    # Summary
    total_time = time.time() - t_start
    t_h, rem = divmod(int(total_time), 3600)
    t_m, t_s = divmod(rem, 60)
    time_str = f"{t_h}h {t_m:02d}m {t_s:02d}s" if t_h else f"{t_m}m {t_s:02d}s"

    banner("Training Complete")
    info("Steps", global_step)
    info("Time", time_str)
    info("Best loss", f"{best_loss:.4f}")
    info("Final checkpoint", final_path)
    print()
    section("Next steps")
    print(f"\n  {DIM}# Test quality{RESET}")
    print(f"  python distill/test_checkpoint.py {final_path} -n 50 -t 0.7 -r 1.3")
    print()
    print(f"  {DIM}# Deploy{RESET}")
    print(f"  python distill/deploy.py {final_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description="BLIPDistill: Full-KL distillation on teacher corpus")
    parser.add_argument("--teacher", required=True, help="HuggingFace teacher model")
    parser.add_argument("--corpus", required=True, help="Path to teacher-generated corpus")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--output_dir", default="blipdistill")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

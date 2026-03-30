#!/usr/bin/env python3
"""
BLIPDistill: Dataset-free BitNet distillation.

The teacher IS the training data. No external dataset needed.
Training signal comes from:
  1. Random token probes → teacher responses (broad exploration)
  2. Adversarial probes → find and fix max disagreement (focused refinement)
  3. Teacher-continued probes → semi-structured inputs (in-distribution)

Usage:
    # Quick local test (0.5B, CPU/MPS)
    python blipdistill/train.py --teacher Qwen/Qwen2.5-0.5B --device mps --max_steps 500

    # Full run (3B, GPU)
    python blipdistill/train.py --teacher Qwen/Qwen2.5-3B --device cuda --max_steps 5000

    # Wider student for higher fidelity
    python blipdistill/train.py --teacher Qwen/Qwen2.5-0.5B --device cuda --max_steps 5000 --seq_len 256
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
from blipdistill.probing import random_probe, adversarial_probe, ProbeScheduler, EmbeddingSpaceProbe
from blipdistill.losses import combined_loss, QKCapture


class ProbeDataset:
    """Loads pre-extracted probe data from disk (from extract.py)."""

    def __init__(self, probe_dir: str, device: torch.device):
        self.device = device
        metadata = torch.load(os.path.join(probe_dir, "metadata.pt"), weights_only=False)
        self.vocab_size = metadata["vocab_size"]
        self.top_k = metadata["top_k"]
        self.seq_len = metadata["seq_len"]
        self.n_examples = metadata["n_examples"]

        # Load all chunks
        self.chunks = []
        chunk_files = sorted(f for f in os.listdir(probe_dir) if f.startswith("chunk_"))
        for cf in chunk_files:
            self.chunks.append(torch.load(os.path.join(probe_dir, cf), weights_only=False))

        # Build flat index: (chunk_idx, position_in_chunk)
        self.index = []
        for ci, chunk in enumerate(self.chunks):
            n = len(chunk["probe_types"])
            for i in range(n):
                self.index.append((ci, i))

        print(f"  Loaded {len(self.index):,} probe examples from {len(self.chunks)} chunks")

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """
        Get a random batch of (inputs_embeds_or_None, input_ids_or_None, teacher_logits).

        Teacher logits are reconstructed from top-K sparse format to full vocab size.
        """
        indices = torch.randint(0, len(self.index), (batch_size,))

        all_ids = []
        all_logits = []

        for idx in indices:
            ci, pos = self.index[idx.item()]
            chunk = self.chunks[ci]

            # Get input_ids (works for both vocab_seed and rand probes)
            if "input_ids" in chunk:
                all_ids.append(chunk["input_ids"][pos])

            # Reconstruct teacher soft targets from top-K
            # These are already probabilities (after softmax with tau during extraction)
            # Store as probabilities directly — the loss function will use them as targets
            top_k_idx = chunk["top_k_indices"][pos]    # (seq_len, top_k) int32
            top_k_probs = chunk["top_k_probs"][pos]    # (seq_len, top_k) float16

            # Create sparse probability tensor (zeros for non-top-K tokens)
            probs_float = top_k_probs.float()
            # Replace NaN with uniform
            nan_mask = probs_float.isnan()
            if nan_mask.any():
                probs_float[nan_mask] = 1.0 / self.top_k
            full_probs = torch.zeros(self.seq_len, self.vocab_size)
            for s in range(self.seq_len):
                full_probs[s, top_k_idx[s].long()] = probs_float[s]
            all_logits.append(full_probs)

        teacher_logits = torch.stack(all_logits).to(self.device)

        input_ids = None
        if all_ids:
            input_ids = torch.stack(all_ids).to(self.device)

        return None, input_ids, teacher_logits


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


def train(args):
    device = torch.device(args.device)
    t_start = time.time()

    teacher_name = args.teacher.split("/")[-1]
    offline_mode = args.probe_data is not None

    if offline_mode:
        banner(f"BLIPDistill: {teacher_name} → 1.58-bit BitNet (offline probes)")
    else:
        banner(f"BLIPDistill: {teacher_name} → 1.58-bit BitNet (dataset-free)")

    # --- Load teacher (only if online mode) ---
    teacher = None
    if offline_mode:
        section("Loading pre-extracted probe data")
        probe_dataset = ProbeDataset(args.probe_data, device)
        vocab_size = probe_dataset.vocab_size
        info("Probe examples", f"{probe_dataset.n_examples:,}")
        info("Top-K", probe_dataset.top_k)
    else:
        section("Loading teacher")
        t0 = time.time()
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=torch.float16, attn_implementation="eager"
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.to(device)
        vocab_size = teacher.config.vocab_size

        info("Parameters", f"{sum(p.numel() for p in teacher.parameters()):,}")
        info("Loaded in", f"{time.time() - t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    info("Model", args.teacher)
    info("Vocab size", f"{vocab_size:,}")
    info("Mode", "offline (probes from disk)" if offline_mode else "online (teacher live)")

    # Enable TF32 on CUDA
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        info("TF32", "enabled")

    # --- Build student ---
    section("Building student (BitLinear + SubLN surgery)")
    t0 = time.time()
    student = build_student(args.teacher, device=device)

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    info("Total parameters", f"{total_params:,}")
    info("Trainable", f"{trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")
    info("Built in", f"{time.time() - t0:.1f}s")

    # --- Register QKV hooks for attention distillation (online mode only) ---
    n_layers = student.config.num_hidden_layers
    student_cap = None
    teacher_cap = None
    if not offline_mode:
        student_cap = QKCapture()
        teacher_cap = QKCapture()
        student_cap.register(student.model.layers[n_layers - 1].self_attn)
        teacher_cap.register(teacher.model.layers[n_layers - 1].self_attn)

    # --- Optimizer and scheduler ---
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

    probe_scheduler = ProbeScheduler(total_steps, adversarial_warmup=args.adversarial_warmup)

    # Initialize embedding space probe (online mode only)
    embed_probe = None
    if not offline_mode:
        section("Analyzing teacher embedding space")
        embed_probe = EmbeddingSpaceProbe(teacher)
        info("Embedding dim", embed_probe.embed_dim)
        info("Principal components", embed_probe.components.shape[0])

    # --- Training log ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.jsonl")
    with open(log_path, "w") as f:
        f.write(json.dumps({"type": "header", "total_steps": total_steps, "mode": "dataset-free"}) + "\n")

    # --- Training loop ---
    section(f"Training ({total_steps} steps, dataset-free)")
    info("Batch size", args.batch_size)
    info("Sequence length", args.seq_len)
    info("Accumulation", args.accumulation_steps)
    info("Temperature", args.tau)
    info("Adversarial warmup", f"{args.adversarial_warmup} steps")
    info("Probe modes", "random + teacher-continued + adversarial")
    print()

    best_loss = float("inf")
    accum_loss = 0.0
    accum_ld = 0.0
    accum_ad = 0.0
    global_step = 0
    optimizer.zero_grad()

    for step_idx in range(total_steps * args.accumulation_steps):
        accum_pos = (step_idx % args.accumulation_steps) + 1

        # --- Generate probe input + get teacher response ---
        if offline_mode:
            # Load from pre-extracted probes
            inputs_embeds, input_ids, teacher_logits = probe_dataset.get_batch(args.batch_size)
            use_embeds = inputs_embeds is not None
            probe_type = "DISK"
        else:
            # Online: generate probes and run teacher live
            adv_ratio = probe_scheduler.get_adversarial_ratio(global_step)

            if adv_ratio > 0 and torch.rand(1).item() < adv_ratio:
                input_ids = adversarial_probe(
                    teacher, student, vocab_size,
                    args.batch_size, args.seq_len, device,
                    n_steps=args.adversarial_steps,
                )
                probe_type = "ADV"
                use_embeds = False
            elif embed_probe is not None and torch.rand(1).item() < 0.5:
                inputs_embeds = embed_probe.sample(args.batch_size, args.seq_len, device)
                probe_type = "EMBED"
                use_embeds = True
            else:
                input_ids = random_probe(vocab_size, args.batch_size, args.seq_len, device)
                probe_type = "RAND"
                use_embeds = False

            with torch.no_grad():
                if use_embeds:
                    teacher_out = teacher(inputs_embeds=inputs_embeds.to(
                        next(teacher.parameters()).dtype))
                else:
                    teacher_out = teacher(input_ids)
                teacher_logits = teacher_out.logits.float().detach()
                del teacher_out
            empty_cache(args.device)

        # --- Student forward ---
        if use_embeds and inputs_embeds is not None:
            student_out = student(inputs_embeds=inputs_embeds)
        elif input_ids is not None:
            student_out = student(input_ids=input_ids)
        else:
            # Fallback: random tokens
            input_ids = random_probe(vocab_size, args.batch_size, args.seq_len, device)
            student_out = student(input_ids=input_ids)
            probe_type = "RAND"

        # --- Loss ---
        loss, ld, ad = combined_loss(
            student_out.logits, teacher_logits,
            student_cap, teacher_cap,
            tau=args.tau, lambda_ld=1.0, gamma_ad=args.gamma_ad,
            teacher_is_probs=offline_mode,
        )
        loss = loss / args.accumulation_steps

        # Skip NaN batches (can happen with extreme random inputs)
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_ld = 0.0
            accum_ad = 0.0
            del student_out, teacher_logits
            empty_cache(args.device)
            continue

        loss.backward()

        accum_loss += loss.item()
        accum_ld += ld
        accum_ad += ad

        del student_out, teacher_logits

        # --- Progress ---
        if accum_pos == args.accumulation_steps:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            avg_loss = accum_loss
            avg_ld = accum_ld / args.accumulation_steps
            avg_ad = accum_ad / args.accumulation_steps

            if avg_loss < best_loss:
                best_loss = avg_loss

            elapsed = time.time() - t_start
            sps = global_step / elapsed if elapsed > 0 else 0
            eta = (total_steps - global_step) / sps if sps > 0 else 0
            lr_now = scheduler.get_last_lr()[0]

            # Print progress
            eta_m, eta_s = divmod(int(eta), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_s:02d}s"

            print(f"  {GREEN}{BOLD}Step {global_step}/{total_steps}{RESET}"
                  f"  {DIM}|{RESET} loss {YELLOW}{avg_loss:.2f}{RESET}"
                  f"  {DIM}|{RESET} LD {avg_ld:.1f}  AD {avg_ad:.1f}"
                  f"  {DIM}|{RESET} lr {lr_now:.1e}"
                  f"  {DIM}|{RESET} {probe_type}"
                  f"  {DIM}ETA {eta_str}{RESET}")

            # Log
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "type": "step", "step": global_step,
                    "loss": avg_loss, "ce": 0.0, "ld": avg_ld, "ad": avg_ad,
                    "lr": lr_now, "elapsed": elapsed, "eta": eta,
                    "probe": probe_type,
                }) + "\n")

            accum_loss = 0.0
            accum_ld = 0.0
            accum_ad = 0.0

            # Save checkpoint
            if global_step % args.save_every == 0:
                save_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                print(f"    {DIM}Saving checkpoint...{RESET}", end="", flush=True)
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

    # --- Save final ---
    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        "step": global_step,
        "model_state_dict": student.state_dict(),
        "config": student.config,
    }, final_path)

    if student_cap:
        student_cap.remove()
    if teacher_cap:
        teacher_cap.remove()

    # --- Summary ---
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
    section("Next step")
    print(f"\n    python distill/deploy.py {final_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="BLIPDistill: Dataset-free BitNet distillation",
    )
    parser.add_argument("--teacher", required=True, help="HuggingFace teacher model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tau", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--gamma_ad", type=float, default=0.01, help="Attention distillation weight")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--output_dir", default="blipdistill")
    parser.add_argument("--adversarial_warmup", type=int, default=200,
                        help="Steps before adversarial probing starts")
    parser.add_argument("--adversarial_steps", type=int, default=5,
                        help="Gradient steps per adversarial probe")
    parser.add_argument("--probe_data", default=None,
                        help="Path to pre-extracted probe data (from extract.py). "
                             "If provided, teacher is not loaded — trains from disk only.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

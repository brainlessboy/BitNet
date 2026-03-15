"""
BitDistill PoC: Distill Qwen2.5-0.5B -> 1.58-bit BitNet model.

Based on "BitNet Distillation" (arxiv 2510.13998).
Implements Stage 1 (arch modification) and Stage 3 (distillation).
Stage 2 (continued pre-training) is skipped per ablation results.

Usage:
    # Smoke test (5 min)
    python distill/distill.py --max_steps 50 --device cpu

    # Short run (30 min)
    python distill/distill.py --max_steps 500 --device mps

    # Full run (8-12 hours)
    python distill/distill.py --epochs 2 --device mps
"""

import argparse
import math
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Pretty CLI output
# ---------------------------------------------------------------------------
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"
BG_BLUE = "\033[44m"
CLEAR_LINE = "\033[2K\r"


def get_term_width():
    return shutil.get_terminal_size((80, 24)).columns


def banner(text):
    w = get_term_width()
    pad = (w - len(text) - 4) // 2
    print(f"\n{CYAN}{'=' * w}")
    print(f"{' ' * pad}  {BOLD}{text}{RESET}{CYAN}  ")
    print(f"{'=' * w}{RESET}\n")


def section(text):
    print(f"\n{CYAN}{BOLD}  > {text}{RESET}")


def info(label, value):
    print(f"    {DIM}{label:<20}{RESET}{WHITE}{value}{RESET}")


def progress_bar(current, total, width=30, label="", extra=""):
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = f"{'█' * filled}{'░' * (width - filled)}"
    pct = f"{frac * 100:5.1f}%"
    line = f"  {CYAN}{bar}{RESET} {WHITE}{pct}{RESET} {DIM}{label}{RESET}"
    if extra:
        line += f"  {extra}"
    sys.stdout.write(CLEAR_LINE + line)
    sys.stdout.flush()


def step_summary(step, total, loss, ce, ld, ad, lr, elapsed, eta):
    w = get_term_width()
    step_str = f"Step {step}/{total}"
    print(f"\r{' ' * w}", end="")
    print(f"\r  {GREEN}{BOLD}{step_str}{RESET}", end="")
    print(f"  {DIM}|{RESET} loss {YELLOW}{loss:.2f}{RESET}", end="")
    print(f"  {DIM}|{RESET} CE {ce:.3f}  LD {ld:.0f}  AD {ad:.1f}", end="")
    print(f"  {DIM}|{RESET} lr {MAGENTA}{lr:.1e}{RESET}", end="")
    print(f"  {DIM}|{RESET} {elapsed:.0f}s", end="")
    if eta > 0:
        eta_m, eta_s = divmod(int(eta), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        if eta_h > 0:
            print(f"  {DIM}ETA {eta_h}h{eta_m:02d}m{RESET}", end="")
        else:
            print(f"  {DIM}ETA {eta_m}m{eta_s:02d}s{RESET}", end="")
    print()


# ---------------------------------------------------------------------------
# RMSNorm (manual impl for PyTorch < 2.4 compatibility)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# BitLinear: nn.Linear with ternary weight + INT8 activation quantization
# ---------------------------------------------------------------------------
class BitLinear(nn.Linear):
    """Linear layer with STE-based ternary weight and INT8 activation quantization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Weight quantization: absmean -> ternary {-1, 0, +1}
        w = self.weight
        delta = w.abs().mean().clamp(min=1e-5)
        w_q = (w / delta).round().clamp(-1, 1) * delta
        w_q = w + (w_q - w).detach()  # STE

        # Activation quantization: per-token absmax -> INT8
        gamma = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_q = (x * 127.0 / gamma).round().clamp(-128, 127) * (gamma / 127.0)
        x_q = x + (x_q - x).detach()  # STE

        return F.linear(x_q, w_q, self.bias)


# ---------------------------------------------------------------------------
# NormedLinear: wraps SubLN + projection for clean insertion
# ---------------------------------------------------------------------------
class NormedLinear(nn.Module):
    """SubLN norm applied before a linear projection."""

    def __init__(self, norm: RMSNorm, linear: nn.Module):
        super().__init__()
        self.norm = norm
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


# ---------------------------------------------------------------------------
# Model surgery: Qwen2.5 -> BitNet student
# ---------------------------------------------------------------------------
PROJECTION_NAMES_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
PROJECTION_NAMES_MLP = ["gate_proj", "up_proj", "down_proj"]


def replace_linear_with_bitlinear(module: nn.Module, name: str) -> BitLinear:
    old = getattr(module, name)
    new = BitLinear(old.in_features, old.out_features, bias=old.bias is not None)
    new.weight = old.weight
    if old.bias is not None:
        new.bias = old.bias
    setattr(module, name, new)
    return new


def modify_student(model: nn.Module) -> nn.Module:
    """Transform Qwen2.5 into a BitNet student with BitLinear + SubLN."""
    config = model.config

    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        # Replace all projections with BitLinear
        for name in PROJECTION_NAMES_ATTN:
            replace_linear_with_bitlinear(attn, name)
        for name in PROJECTION_NAMES_MLP:
            replace_linear_with_bitlinear(mlp, name)

        # Insert SubLN before o_proj (attention output)
        inner_attn_ln = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        attn.o_proj = NormedLinear(inner_attn_ln, attn.o_proj)

        # Insert SubLN before down_proj (FFN output)
        ffn_layernorm = RMSNorm(config.intermediate_size, eps=config.rms_norm_eps)
        mlp.down_proj = NormedLinear(ffn_layernorm, mlp.down_proj)

    return model


# ---------------------------------------------------------------------------
# QKV hook capture for attention distillation
# ---------------------------------------------------------------------------
class QKVCapture:
    def __init__(self):
        self.q = None
        self.k = None
        self.v = None
        self._hooks = []

    def register(self, layer_attn: nn.Module):
        """Register hooks on q_proj, k_proj, v_proj of an attention layer."""
        def make_hook(key):
            def hook(_module, _input, output):
                setattr(self, key, output)
            return hook

        for name in ["q", "k", "v"]:
            proj = getattr(layer_attn, f"{name}_proj")
            h = proj.register_forward_hook(make_hook(name))
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def compute_attention_distillation_loss(
    student_cap: QKVCapture, teacher_cap: QKVCapture, temperature: float = 1.0
) -> torch.Tensor:
    """MiniLM-style self-relation distillation on Q, K, V."""
    loss = torch.tensor(0.0, device=student_cap.q.device)

    for key in ["q", "k", "v"]:
        s = getattr(student_cap, key)
        t = getattr(teacher_cap, key)

        s = F.normalize(s.float(), dim=-1)
        t = F.normalize(t.float(), dim=-1)

        s_rel = torch.bmm(s, s.transpose(-1, -2)) / temperature
        t_rel = torch.bmm(t, t.transpose(-1, -2)) / temperature

        s_log_prob = F.log_softmax(s_rel, dim=-1)
        t_prob = F.softmax(t_rel, dim=-1)
        loss = loss + F.kl_div(s_log_prob, t_prob, reduction="batchmean")

    return loss / 3.0


# ---------------------------------------------------------------------------
# Combined distillation loss
# ---------------------------------------------------------------------------
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    student_cap: QKVCapture,
    teacher_cap: QKVCapture,
    tau: float = 5.0,
    lambda_ld: float = 10.0,
    gamma_ad: float = 1e-5,
) -> tuple:
    """Combined loss: L_CE + lambda * L_LD + gamma * L_AD"""
    loss_ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    student_soft = F.log_softmax(student_logits.float() / tau, dim=-1)
    teacher_soft = F.softmax(teacher_logits.float() / tau, dim=-1)
    loss_ld = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (tau ** 2)

    loss_ad = compute_attention_distillation_loss(student_cap, teacher_cap)

    total = loss_ce + lambda_ld * loss_ld + gamma_ad * loss_ad
    return total, loss_ce.item(), loss_ld.item(), loss_ad.item()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_dataloader(tokenizer, batch_size: int, max_length: int):
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    def format_and_tokenize(example):
        if example["input"]:
            text = (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}"
            )
        else:
            text = (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['output']}"
            )
        enc = tokenizer(
            text, truncation=True, max_length=max_length, padding="max_length"
        )
        enc["labels"] = enc["input_ids"].copy()
        enc["labels"] = [
            -100 if tok == tokenizer.pad_token_id else tok for tok in enc["labels"]
        ]
        return enc

    dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)
    dataset.set_format("torch")

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    banner("BitDistill: Qwen2.5-0.5B -> 1.58-bit BitNet")

    section("Configuration")
    info("Device", args.device)
    info("Effective batch", f"{args.batch_size} x {args.accumulation_steps} = {args.batch_size * args.accumulation_steps}")
    info("Sequence length", args.max_length)
    info("Learning rate", f"{args.lr} (cosine decay)")
    info("Temperature", args.tau)
    info("Loss weights", f"lambda_ld={args.lambda_ld}, gamma_ad={args.gamma_ad}")
    info("Max steps", args.max_steps if args.max_steps else "unlimited")

    device = torch.device(args.device)

    # Load tokenizer
    section("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    info("Vocab size", f"{tokenizer.vocab_size:,}")

    # Load teacher (frozen FP16)
    section("Loading teacher model (Qwen2.5-0.5B FP16)")
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.to(device)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    info("Parameters", f"{teacher_params:,}")
    info("Loaded in", f"{time.time() - t0:.1f}s")

    # Load student and modify
    section("Loading student model + BitLinear + SubLN surgery")
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    student = modify_student(student)
    student.model.embed_tokens.weight.requires_grad = False
    if hasattr(student, "lm_head") and not student.config.tie_word_embeddings:
        student.lm_head.weight.requires_grad = False
    student.to(device)
    student.train()
    student.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    info("Total parameters", f"{total_params:,}")
    info("Trainable", f"{trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    info("Frozen", f"{total_params - trainable_params:,} (embeddings)")
    info("Modified in", f"{time.time() - t0:.1f}s")

    # Register QKV hooks
    last_layer_idx = len(student.model.layers) - 1
    student_cap = QKVCapture()
    teacher_cap = QKVCapture()
    student_cap.register(student.model.layers[last_layer_idx].self_attn)
    teacher_cap.register(teacher.model.layers[last_layer_idx].self_attn)
    info("Distill layer", f"layer {last_layer_idx} (last)")

    # Build dataloader
    section("Loading dataset (yahma/alpaca-cleaned)")
    t0 = time.time()
    dataloader = build_dataloader(tokenizer, args.batch_size, args.max_length)
    info("Samples", f"{len(dataloader) * args.batch_size:,}")
    info("Batches/epoch", f"{len(dataloader):,}")
    info("Loaded in", f"{time.time() - t0:.1f}s")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    total_steps = args.max_steps or (len(dataloader) * args.epochs // args.accumulation_steps)
    warmup_steps = min(200, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Training ---
    banner(f"Training  ({total_steps} steps)")
    info("Warmup", f"{warmup_steps} steps")
    info("Save every", f"{args.save_every} steps")
    info("Checkpoints", ckpt_dir)
    print()

    global_step = 0
    accum_loss = 0.0
    accum_ce = 0.0
    accum_ld = 0.0
    accum_ad = 0.0
    best_loss = float("inf")
    optimizer.zero_grad()
    t_start = time.time()

    for epoch in range(args.epochs):
        print(f"  {CYAN}{BOLD}Epoch {epoch + 1}/{args.epochs}{RESET}")
        print()

        for batch_idx, batch in enumerate(dataloader):
            micro_step = batch_idx + 1
            accum_pos = ((batch_idx) % args.accumulation_steps) + 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Teacher forward
            with torch.no_grad():
                teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_out.logits.float().detach()
            del teacher_out
            if args.device == "mps":
                torch.mps.empty_cache()

            # Student forward
            student_out = student(input_ids=input_ids, attention_mask=attention_mask)

            # Loss + backward
            loss, ce, ld, ad = distillation_loss(
                student_out.logits, teacher_logits, labels,
                student_cap, teacher_cap,
                tau=args.tau, lambda_ld=args.lambda_ld, gamma_ad=args.gamma_ad,
            )
            loss = loss / args.accumulation_steps
            loss.backward()

            accum_loss += loss.item()
            accum_ce += ce
            accum_ld += ld
            accum_ad += ad

            del student_out, teacher_logits
            if args.device == "mps":
                torch.mps.empty_cache()

            # Progress bar for accumulation
            elapsed = time.time() - t_start
            sps = global_step / elapsed if elapsed > 0 and global_step > 0 else None
            eta_str = ""
            if sps and sps > 0:
                eta = (total_steps - global_step) / sps
                eta_m, eta_s = divmod(int(eta), 60)
                eta_h, eta_m = divmod(eta_m, 60)
                eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_s:02d}s"

            extra = (
                f"{DIM}CE{RESET} {ce:.2f}  "
                f"{DIM}LD{RESET} {ld:.0f}  "
                f"{DIM}AD{RESET} {ad:.1f}"
            )
            progress_bar(
                accum_pos, args.accumulation_steps,
                width=20,
                label=f"step {global_step + 1}/{total_steps}  {DIM}micro {accum_pos}/{args.accumulation_steps}{RESET}",
                extra=extra,
            )

            # Optimizer step
            if accum_pos == args.accumulation_steps:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = accum_loss
                avg_ce = accum_ce / args.accumulation_steps
                avg_ld = accum_ld / args.accumulation_steps
                avg_ad = accum_ad / args.accumulation_steps

                if avg_loss < best_loss:
                    best_loss = avg_loss

                elapsed = time.time() - t_start
                sps = global_step / elapsed if elapsed > 0 else 0
                eta = (total_steps - global_step) / sps if sps > 0 else 0

                # Clear progress bar and print step summary
                print()  # newline after progress bar
                step_summary(global_step, total_steps, avg_loss, avg_ce, avg_ld, avg_ad,
                             scheduler.get_last_lr()[0], elapsed, eta)

                accum_loss = 0.0
                accum_ce = 0.0
                accum_ld = 0.0
                accum_ad = 0.0

                # Save checkpoint
                if global_step % args.save_every == 0:
                    save_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                    print(f"    {DIM}Saving checkpoint...{RESET}", end="", flush=True)
                    torch.save(
                        {
                            "step": global_step,
                            "model_state_dict": student.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": student.config,
                        },
                        save_path,
                    )
                    size_mb = os.path.getsize(save_path) / (1024 * 1024)
                    print(f"\r    {GREEN}Saved{RESET} {save_path} {DIM}({size_mb:.0f} MB){RESET}")

                if args.max_steps and global_step >= args.max_steps:
                    break

            if args.device == "mps" and batch_idx % 50 == 0:
                torch.mps.empty_cache()

        if args.max_steps and global_step >= args.max_steps:
            break

        print(f"\n  {GREEN}Epoch {epoch + 1} complete{RESET}\n")

    # Save final checkpoint
    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save(
        {"step": global_step, "model_state_dict": student.state_dict(), "config": student.config},
        final_path,
    )

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
    print(f"    {DIM}1.{RESET} python distill/export_bitnet.py --checkpoint {final_path} --output models/distilled-qwen-bitnet")
    print(f"    {DIM}2.{RESET} python utils/convert-hf-to-gguf-bitnet.py models/distilled-qwen-bitnet/ --outtype f32")
    print(f"    {DIM}3.{RESET} ./build/bin/llama-quantize models/distilled-qwen-bitnet/ggml-model-f32.gguf models/distilled-qwen-bitnet/ggml-model-i2_s.gguf I2_S 1")
    print(f"    {DIM}4.{RESET} ./chat.sh models/distilled-qwen-bitnet")
    print()

    student_cap.remove()
    teacher_cap.remove()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BitDistill: Qwen2.5-0.5B -> 1.58-bit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""{DIM}Examples:
  python distill/distill.py --max_steps 50 --device cpu    # smoke test
  python distill/distill.py --max_steps 500 --device mps   # short run
  python distill/distill.py --epochs 2 --device mps        # full run{RESET}""",
    )
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=None, help="Override epochs, stop after N optimizer steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=5.0, help="Distillation temperature")
    parser.add_argument("--lambda_ld", type=float, default=10.0, help="Logits distillation weight")
    parser.add_argument("--gamma_ad", type=float, default=1e-5, help="Attention distillation weight")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--output_dir", default="distill")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

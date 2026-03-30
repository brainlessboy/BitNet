"""
BitDistill: Distill a teacher model -> 1.58-bit BitNet student.

Based on "BitNet Distillation" (arxiv 2510.13998).
Implements Stage 1 (arch modification) and Stage 3 (distillation).
Stage 2 (continued pre-training) is skipped per ablation results.

Usage:
    # Smoke test
    python distill/distill.py --max_steps 50 --device cpu

    # Local MPS run
    python distill/distill.py --epochs 2 --device mps

    # Single GPU (cloud)
    python distill/distill.py --device cuda --teacher_model Qwen/Qwen2.5-3B \
        --dataset slimorca --batch_size 4 --accumulation_steps 4 --epochs 2

    # Multi-GPU distributed (8x A100)
    torchrun --nproc_per_node=8 distill/distill.py --device cuda \
        --teacher_model Qwen/Qwen2.5-3B --student_model Qwen/Qwen2.5-3B \
        --dataset slimorca --batch_size 4 --accumulation_steps 4 --epochs 2

    # Resume from checkpoint
    python distill/distill.py --device cuda --resume_from distill/checkpoints/step_1000.pt
"""

import argparse
import functools
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

# Conditional FSDP imports (not available on MPS/CPU-only machines)
try:
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.utils.data.distributed import DistributedSampler
    HAS_FSDP = True
except ImportError:
    HAS_FSDP = False


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
# Training log: append-only JSONL for dashboard viewer
# ---------------------------------------------------------------------------
class TrainingLog:
    """Appends one JSON line per optimizer step to a log file."""

    def __init__(self, path, total_steps):
        self.path = path
        self.total_steps = total_steps
        # Write header line with metadata
        import json
        with open(self.path, "w") as f:
            f.write(json.dumps({"type": "header", "total_steps": total_steps}) + "\n")

    def record(self, step, loss, ce, ld, ad, lr, elapsed, eta):
        import json
        line = json.dumps({
            "type": "step", "step": step, "loss": loss,
            "ce": ce, "ld": ld, "ad": ad,
            "lr": lr, "elapsed": elapsed, "eta": eta,
        })
        with open(self.path, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def setup_distributed():
    """Initialize distributed training if launched via torchrun."""
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return 0, 0, 1  # single process

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    """Get the underlying model from FSDP wrapper."""
    return model.module if hasattr(model, "module") else model


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
    target_device = student_cap.q.device
    loss = torch.tensor(0.0, device=target_device)

    for key in ["q", "k", "v"]:
        s = getattr(student_cap, key)
        t = getattr(teacher_cap, key).to(target_device)

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

    # Mask padding positions (labels == -100) from KL divergence
    mask = (labels != -100).unsqueeze(-1)  # (batch, seq_len, 1)
    student_soft = F.log_softmax(student_logits.float() / tau, dim=-1)
    teacher_soft = F.softmax(teacher_logits.float() / tau, dim=-1)
    kl_per_token = F.kl_div(student_soft, teacher_soft, reduction="none").sum(dim=-1)  # (batch, seq_len)
    n_tokens = mask.sum().clamp(min=1)
    loss_ld = (kl_per_token * mask.squeeze(-1)).sum() / n_tokens * (tau ** 2)

    loss_ad = compute_attention_distillation_loss(student_cap, teacher_cap)

    total = loss_ce + lambda_ld * loss_ld + gamma_ad * loss_ad
    return total, loss_ce.item(), loss_ld.item(), loss_ad.item()


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def empty_device_cache(device_str: str):
    if device_str == "cuda":
        torch.cuda.empty_cache()
    elif device_str == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _format_alpaca(example):
    if example["input"]:
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )


def _format_sharegpt(example):
    parts = []
    for turn in example["conversations"]:
        role = turn["from"]
        if role == "system":
            parts.append(f"### System:\n{turn['value']}")
        elif role == "human":
            parts.append(f"### User:\n{turn['value']}")
        elif role == "gpt":
            parts.append(f"### Assistant:\n{turn['value']}")
    return "\n\n".join(parts)


DATASET_CONFIGS = {
    "alpaca": {
        "path": "yahma/alpaca-cleaned",
        "split": "train",
        "formatter": _format_alpaca,
    },
    "slimorca": {
        "path": "Open-Orca/SlimOrca",
        "split": "train",
        "formatter": _format_sharegpt,
    },
    "openhermes": {
        "path": "teknium/OpenHermes-2.5",
        "split": "train",
        "formatter": _format_sharegpt,
    },
}


def build_dataloader(tokenizer, batch_size: int, max_length: int,
                     dataset_name: str = "alpaca", rank: int = 0, world_size: int = 1):
    cfg = DATASET_CONFIGS[dataset_name]
    dataset = load_dataset(cfg["path"], split=cfg["split"])
    formatter = cfg["formatter"]

    def format_and_tokenize(example):
        text = formatter(example)
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

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True, num_workers=0,
    )
    return dataloader, sampler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    # --- Distributed setup ---
    rank, local_rank, world_size = setup_distributed()
    is_distributed = world_size > 1
    is_main = rank == 0

    teacher_name = args.teacher_model.split("/")[-1]
    student_name = args.student_model.split("/")[-1]

    if is_main:
        banner(f"BitDistill: {teacher_name} -> 1.58-bit BitNet ({student_name})")
        section("Configuration")
        info("Device", args.device)
        if is_distributed:
            info("Distributed", f"{world_size} GPUs (FSDP)")
        info("Teacher", args.teacher_model)
        info("Student", args.student_model)
        info("Dataset", args.dataset)
        effective = args.batch_size * args.accumulation_steps * world_size
        info("Effective batch", f"{args.batch_size} x {args.accumulation_steps} x {world_size} = {effective}")
        info("Sequence length", args.max_length)
        info("Learning rate", f"{args.lr} (cosine decay)")
        info("Temperature", args.tau)
        info("Loss weights", f"lambda_ld={args.lambda_ld}, gamma_ad={args.gamma_ad}")
        info("Max steps", args.max_steps if args.max_steps else "unlimited")

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    # Teacher can live on a separate GPU to free VRAM for the student
    teacher_device = torch.device(args.teacher_device) if args.teacher_device else device

    # Enable TF32 on CUDA for ~3x speedup on A100/H100
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if is_main:
            info("TF32", "enabled")
            if args.teacher_device:
                info("Teacher device", str(teacher_device))
                info("Student device", str(device))

    # Load tokenizer (from teacher model)
    if is_main:
        section("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_main:
        info("Vocab size", f"{tokenizer.vocab_size:,}")

    # Load teacher (frozen FP16) — replicated on each GPU
    if is_main:
        section(f"Loading teacher model ({teacher_name} FP16)")
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.to(teacher_device)
    if is_main:
        teacher_params = sum(p.numel() for p in teacher.parameters())
        info("Parameters", f"{teacher_params:,}")
        info("Loaded in", f"{time.time() - t0:.1f}s")

    # Load student and modify
    if is_main:
        section("Loading student model + BitLinear + SubLN surgery")
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
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

    if is_main:
        total_params = sum(p.numel() for p in student.parameters())
        trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
        info("Total parameters", f"{total_params:,}")
        info("Trainable", f"{trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        info("Frozen", f"{total_params - trainable_params:,} (embeddings)")
        info("Modified in", f"{time.time() - t0:.1f}s")

    # Register QKV hooks BEFORE FSDP wrapping
    n_student_layers = len(student.model.layers)
    n_teacher_layers = len(teacher.model.layers)
    last_student_idx = n_student_layers - 1
    last_teacher_idx = n_teacher_layers - 1

    student_cap = QKVCapture()
    teacher_cap = QKVCapture()
    student_cap.register(student.model.layers[last_student_idx].self_attn)
    teacher_cap.register(teacher.model.layers[last_teacher_idx].self_attn)
    if is_main:
        info("Distill layer", f"student layer {last_student_idx}, teacher layer {last_teacher_idx}")

    # FSDP wrapping for distributed training
    if is_distributed:
        if is_main:
            section("Wrapping student with FSDP")

        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        auto_wrap = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2DecoderLayer},
        )
        mp_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        student = FSDP(
            student,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap,
            mixed_precision=mp_policy,
            device_id=local_rank,
            use_orig_params=True,
        )
        if is_main:
            info("Sharding", "FULL_SHARD across all GPUs")

    # Resume from checkpoint (before optimizer creation, after FSDP for state dict compat)
    global_step = 0
    start_epoch = 0
    resume_optim_state = None
    if args.resume_from:
        if is_main:
            section(f"Resuming from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if is_distributed:
            # Load into the unwrapped model before FSDP has sharded
            # Since FSDP is already applied, use FSDP's state dict loading
            with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT):
                student.load_state_dict(ckpt["model_state_dict"])
        else:
            student.load_state_dict(ckpt["model_state_dict"])
        resume_optim_state = ckpt.get("optimizer_state_dict")
        global_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        if is_main:
            info("Resumed step", global_step)
            info("Resumed epoch", start_epoch)
        del ckpt

    # Build dataloader
    if is_main:
        section(f"Loading dataset ({args.dataset})")
    t0 = time.time()
    dataloader, sampler = build_dataloader(
        tokenizer, args.batch_size, args.max_length, args.dataset,
        rank=rank, world_size=world_size,
    )
    if is_main:
        info("Samples", f"{len(dataloader) * args.batch_size * world_size:,}")
        info("Batches/epoch/GPU", f"{len(dataloader):,}")
        info("Loaded in", f"{time.time() - t0:.1f}s")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    if resume_optim_state is not None:
        if is_distributed:
            optim_state = FSDP.optim_state_dict_to_load(
                student, optimizer, resume_optim_state,
            )
            optimizer.load_state_dict(optim_state)
        else:
            optimizer.load_state_dict(resume_optim_state)
    del resume_optim_state

    total_steps = args.max_steps or (len(dataloader) * args.epochs // args.accumulation_steps)
    warmup_steps = min(200, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)

    # Training log for dashboard viewer: python distill/dashboard.py distill/training_log.jsonl
    training_log = None
    if is_main:
        log_path = os.path.join(args.output_dir, "training_log.jsonl")
        training_log = TrainingLog(log_path, total_steps)

    # Sync all ranks before training
    if is_distributed:
        dist.barrier()

    # --- Training ---
    if is_main:
        banner(f"Training  ({total_steps} steps)")
        info("Warmup", f"{warmup_steps} steps")
        info("Save every", f"{args.save_every} steps")
        info("Checkpoints", ckpt_dir)
        info("Training log", log_path)
        info("Dashboard", f"python distill/dashboard.py {log_path}")
        print()

    accum_loss = 0.0
    accum_ce = 0.0
    accum_ld = 0.0
    accum_ad = 0.0
    best_loss = float("inf")
    optimizer.zero_grad()
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if is_main:
            print(f"  {CYAN}{BOLD}Epoch {epoch + 1}/{args.epochs}{RESET}")
            print()

        for batch_idx, batch in enumerate(dataloader):
            # Skip batches if resuming mid-epoch
            batches_per_step = args.accumulation_steps
            expected_batch = (global_step * batches_per_step) % len(dataloader)
            if epoch == start_epoch and global_step > 0 and batch_idx < expected_batch:
                continue

            micro_step = batch_idx + 1
            accum_pos = ((batch_idx) % args.accumulation_steps) + 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Teacher forward (may be on a different GPU)
            with torch.no_grad():
                t_input_ids = input_ids.to(teacher_device)
                t_attention_mask = attention_mask.to(teacher_device)
                teacher_out = teacher(input_ids=t_input_ids, attention_mask=t_attention_mask)
                teacher_logits = teacher_out.logits.float().detach().to(device)
                del teacher_out, t_input_ids, t_attention_mask
            empty_device_cache(args.device)

            # Student forward (fp32 for stable gradients through BitLinear)
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

            # Progress bar for accumulation (main rank only)
            if is_main:
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

                # Average loss across ranks for accurate logging
                if is_distributed:
                    loss_tensor = torch.tensor(
                        [avg_loss, avg_ce, avg_ld, avg_ad], device=device
                    )
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    avg_loss, avg_ce, avg_ld, avg_ad = loss_tensor.tolist()

                if avg_loss < best_loss:
                    best_loss = avg_loss

                if is_main:
                    elapsed = time.time() - t_start
                    sps = global_step / elapsed if elapsed > 0 else 0
                    eta = (total_steps - global_step) / sps if sps > 0 else 0
                    lr_now = scheduler.get_last_lr()[0]

                    print()  # newline after progress bar
                    step_summary(global_step, total_steps, avg_loss, avg_ce, avg_ld, avg_ad,
                                 lr_now, elapsed, eta)

                    if training_log:
                        training_log.record(global_step, avg_loss, avg_ce, avg_ld, avg_ad,
                                            lr_now, elapsed, eta)

                accum_loss = 0.0
                accum_ce = 0.0
                accum_ld = 0.0
                accum_ad = 0.0

                # Save checkpoint (rank 0 only)
                if global_step % args.save_every == 0 and is_main:
                    save_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                    print(f"    {DIM}Saving checkpoint...{RESET}", end="", flush=True)

                    if is_distributed:
                        save_policy = FullStateDictConfig(
                            offload_to_cpu=True, rank0_only=True,
                        )
                        with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT, save_policy):
                            cpu_state_dict = student.state_dict()
                        torch.save(
                            {
                                "step": global_step,
                                "epoch": epoch,
                                "model_state_dict": cpu_state_dict,
                                "scheduler_state_dict": scheduler.state_dict(),
                                "config": unwrap_model(student).config,
                            },
                            save_path,
                        )
                    else:
                        torch.save(
                            {
                                "step": global_step,
                                "epoch": epoch,
                                "model_state_dict": student.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "config": student.config,
                            },
                            save_path,
                        )

                    size_mb = os.path.getsize(save_path) / (1024 * 1024)
                    print(f"\r    {GREEN}Saved{RESET} {save_path} {DIM}({size_mb:.0f} MB){RESET}")

                    # Auto-cleanup: keep only last 2 periodic checkpoints
                    existing = sorted(
                        [f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pt")],
                        key=lambda f: int(f.split("_")[1].split(".")[0]),
                    )
                    for old in existing[:-2]:
                        os.remove(os.path.join(ckpt_dir, old))

                if is_distributed:
                    dist.barrier()

                if args.max_steps and global_step >= args.max_steps:
                    break

            if batch_idx % 50 == 0:
                empty_device_cache(args.device)

        if args.max_steps and global_step >= args.max_steps:
            break

        if is_main:
            print(f"\n  {GREEN}Epoch {epoch + 1} complete{RESET}\n")

    # Save final checkpoint
    final_path = os.path.join(ckpt_dir, "final.pt")
    if is_main:
        if is_distributed:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = student.state_dict()
            torch.save(
                {"step": global_step, "epoch": epoch,
                 "model_state_dict": cpu_state_dict,
                 "config": unwrap_model(student).config},
                final_path,
            )
        else:
            torch.save(
                {"step": global_step, "epoch": epoch,
                 "model_state_dict": student.state_dict(),
                 "config": student.config},
                final_path,
            )

    if is_distributed:
        dist.barrier()

    # Summary
    if is_main:
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

    student_cap.remove()
    teacher_cap.remove()
    cleanup_distributed()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BitDistill: teacher -> 1.58-bit BitNet student",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""{DIM}Examples:
  python distill/distill.py --max_steps 50 --device cpu                    # smoke test
  python distill/distill.py --epochs 2 --device mps                        # local MPS
  python distill/distill.py --device cuda --teacher_model Qwen/Qwen2.5-3B  # single GPU
  torchrun --nproc_per_node=8 distill/distill.py --device cuda \\
    --teacher_model Qwen/Qwen2.5-3B --student_model Qwen/Qwen2.5-3B       # 8x GPU
  python distill/distill.py --device cuda --resume_from distill/checkpoints/step_1000.pt{RESET}""",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--teacher_model", default="Qwen/Qwen2.5-0.5B", help="HuggingFace teacher model")
    parser.add_argument("--student_model", default="Qwen/Qwen2.5-0.5B", help="HuggingFace student base model")
    parser.add_argument("--dataset", default="alpaca", choices=["alpaca", "slimorca", "openhermes"])
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
    parser.add_argument("--resume_from", default=None, help="Resume from checkpoint path")
    parser.add_argument("--teacher_device", default=None, help="Put teacher on separate GPU (e.g. cuda:0) while student uses --device")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

"""
Probing strategies for dataset-free distillation.

Generates training inputs directly — no external dataset needed.
Two modes:
  1. Random probing: uniform random token sequences (explore)
  2. Adversarial probing: optimize inputs to find maximum teacher/student disagreement (exploit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


def random_probe(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate random token sequences for probing."""
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


class EmbeddingSpaceProbe:
    """
    Probes the teacher's embedding space directly.

    Computes the mean and covariance of the teacher's embedding matrix,
    then samples continuous vectors from that distribution. These probe
    the teacher's function in the "inhabited" region of embedding space.

    Returns inputs_embeds (not input_ids) — continuous vectors that
    represent meaningful combinations of concepts.
    """

    def __init__(self, teacher: PreTrainedModel):
        embed_weight = teacher.model.embed_tokens.weight.float().detach()

        # Compute statistics of the embedding space
        self.mean = embed_weight.mean(dim=0)  # (embed_dim,)
        self.std = embed_weight.std(dim=0)    # (embed_dim,)

        # For richer sampling: compute top principal components
        centered = embed_weight - self.mean.unsqueeze(0)
        # Use SVD on a subset for efficiency (full vocab is huge)
        n_sample = min(embed_weight.shape[0], 10000)
        indices = torch.randperm(embed_weight.shape[0])[:n_sample]
        subset = centered[indices]
        U, S, Vh = torch.linalg.svd(subset, full_matrices=False)
        # Keep top components that explain 95% of variance
        energy = S.pow(2).cumsum(0) / S.pow(2).sum()
        n_components = max(1, (energy < 0.95).sum().item() + 1)
        n_components = min(n_components, 256)  # cap for efficiency

        self.components = Vh[:n_components]  # (n_components, embed_dim)
        self.component_scales = S[:n_components]  # (n_components,)
        self.embed_dim = embed_weight.shape[1]

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample continuous embedding vectors from the teacher's distribution.

        Returns:
            inputs_embeds: (batch_size, seq_len, embed_dim) — continuous vectors
        """
        # Sample in principal component space, then project to embedding space
        # This gives vectors that lie in the "shape" of the embedding distribution
        n_comp = self.components.shape[0]
        z = torch.randn(batch_size, seq_len, n_comp, device=device)
        z = z * self.component_scales.to(device).unsqueeze(0).unsqueeze(0)

        # Project to embedding space
        embeds = z @ self.components.to(device)  # (batch, seq, embed_dim)

        # Add mean
        embeds = embeds + self.mean.to(device).unsqueeze(0).unsqueeze(0)

        return embeds


@torch.no_grad()
def smart_random_probe(
    teacher: PreTrainedModel,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate semi-structured random sequences.

    Start with random tokens, then let the teacher "continue" them.
    This produces inputs that are more in-distribution for the teacher,
    hitting regions where it has meaningful opinions.

    Half the sequence is random, half is teacher-generated.
    """
    # Start with random prefix
    prefix_len = seq_len // 4
    prefix = torch.randint(0, vocab_size, (batch_size, prefix_len), device=device)

    # Let teacher continue
    generated = prefix
    for _ in range(seq_len - prefix_len):
        logits = teacher(generated).logits[:, -1, :]
        # Sample from teacher distribution (not greedy — want diversity)
        probs = torch.softmax(logits / 0.8, dim=-1)
        next_token = torch.multinomial(probs, 1)
        generated = torch.cat([generated, next_token], dim=-1)

    return generated


def adversarial_probe(
    teacher: PreTrainedModel,
    student: PreTrainedModel,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    n_steps: int = 10,
    lr: float = 0.1,
) -> torch.Tensor:
    """
    Find inputs where teacher and student disagree most.

    Optimizes continuous embeddings via gradient ascent to maximize
    KL divergence between teacher and student. Then snaps to nearest
    discrete tokens for actual training.

    Args:
        teacher: Teacher model (frozen)
        student: Student model (frozen during probing)
        vocab_size: Vocabulary size
        batch_size: Batch size
        seq_len: Sequence length
        device: Device
        n_steps: Optimization steps for the probe
        lr: Learning rate for probe optimization

    Returns:
        Token IDs that maximize teacher/student disagreement
    """
    # Get embedding layer
    embed = teacher.model.embed_tokens

    # Start from random embeddings
    embed_dim = embed.weight.shape[1]
    probe_embeds = torch.randn(batch_size, seq_len, embed_dim, device=device) * 0.1
    probe_embeds.requires_grad_(True)

    optimizer = torch.optim.Adam([probe_embeds], lr=lr)

    student_was_training = student.training
    student.eval()

    for step in range(n_steps):
        optimizer.zero_grad()

        # Forward both models with continuous embeddings
        with torch.no_grad():
            teacher_logits = teacher(inputs_embeds=probe_embeds.detach().float().to(
                next(teacher.parameters()).dtype)).logits.float()

        student_logits = student(inputs_embeds=probe_embeds).logits.float()

        # Maximize disagreement (negative KL = gradient ascent on disagreement)
        teacher_probs = F.softmax(teacher_logits, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        agreement = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        # We want to MAXIMIZE disagreement, so minimize negative agreement
        (-agreement).backward()
        optimizer.step()

    if student_was_training:
        student.train()

    # Snap continuous embeddings to nearest discrete tokens
    with torch.no_grad():
        # Find nearest embedding for each position
        probe_flat = probe_embeds.detach().reshape(-1, embed_dim)
        embed_weights = embed.weight.float()

        # Compute distances to all embeddings
        # Use chunked computation to avoid OOM
        chunk_size = 1024
        token_ids = []
        for i in range(0, probe_flat.shape[0], chunk_size):
            chunk = probe_flat[i:i + chunk_size]
            dists = torch.cdist(chunk, embed_weights)
            token_ids.append(dists.argmin(dim=-1))

        token_ids = torch.cat(token_ids).reshape(batch_size, seq_len)

    return token_ids


class ProbeScheduler:
    """
    Schedules the mix of random vs adversarial probing during training.

    Early training: mostly random (broad exploration)
    Late training: mostly adversarial (focused refinement)
    """

    def __init__(self, total_steps: int, adversarial_warmup: int = 100):
        self.total_steps = total_steps
        self.adversarial_warmup = adversarial_warmup

    def get_adversarial_ratio(self, step: int) -> float:
        """Returns fraction of batch that should be adversarial probes."""
        if step < self.adversarial_warmup:
            return 0.0
        progress = (step - self.adversarial_warmup) / max(self.total_steps - self.adversarial_warmup, 1)
        # Linear ramp from 0 to 0.5
        return min(0.5, progress * 0.5)

"""
Loss functions for dataset-free distillation.

Since we have no "correct" labels, all signal comes from matching
the teacher's behavior. Three complementary losses:

1. Logit matching: student logits match teacher logits (KL divergence)
2. Hidden state matching: student hidden states match teacher's (MSE)
3. Attention matching: student attention patterns match teacher's (KL on self-relations)
"""

import torch
import torch.nn.functional as F


def logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tau: float = 2.0,
    teacher_is_probs: bool = False,
) -> torch.Tensor:
    """
    KL divergence between student and teacher logit distributions.

    Args:
        student_logits: Raw student logits
        teacher_logits: Teacher logits OR pre-computed probabilities
        tau: Temperature
        teacher_is_probs: If True, teacher_logits are already probabilities
                         (from offline/disk mode). Skip softmax on teacher.
    """
    student_logits = student_logits.float().clamp(-100, 100)
    student_soft = F.log_softmax(student_logits / tau, dim=-1)

    if teacher_is_probs:
        # Teacher targets are already probabilities — use directly
        teacher_soft = teacher_logits.float().clamp(min=1e-8)
        # Renormalize in case top-K doesn't sum to exactly 1
        teacher_soft = teacher_soft / teacher_soft.sum(dim=-1, keepdim=True)
    else:
        teacher_logits = teacher_logits.float().clamp(-100, 100)
        teacher_soft = F.softmax(teacher_logits / tau, dim=-1)

    loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (tau ** 2)

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=loss.device, requires_grad=True)

    return loss


def hidden_state_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between student and teacher hidden states.

    Normalized per-dimension to handle different scales.
    """
    # Normalize both to unit variance
    s = student_hidden.float()
    t = teacher_hidden.float()

    s = s / s.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    t = t / t.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)

    return F.mse_loss(s, t)


def attention_relation_loss(
    student_q: torch.Tensor,
    teacher_q: torch.Tensor,
    student_k: torch.Tensor,
    teacher_k: torch.Tensor,
) -> torch.Tensor:
    """
    MiniLM-style self-relation distillation.

    Matches the student's attention self-relation matrix to the teacher's.
    This transfers HOW the model relates tokens to each other.
    """
    loss = torch.tensor(0.0, device=student_q.device)

    for s, t in [(student_q, teacher_q), (student_k, teacher_k)]:
        s = F.normalize(s.float(), dim=-1)
        t = F.normalize(t.float(), dim=-1)

        s_rel = torch.bmm(s, s.transpose(-1, -2))
        t_rel = torch.bmm(t, t.transpose(-1, -2))

        s_log_prob = F.log_softmax(s_rel, dim=-1)
        t_prob = F.softmax(t_rel, dim=-1)
        loss = loss + F.kl_div(s_log_prob, t_prob, reduction="batchmean")

    return loss / 2.0


class QKCapture:
    """Capture Q and K tensors from attention layers for relation distillation."""

    def __init__(self):
        self.q = None
        self.k = None
        self._hooks = []

    def register(self, attn_module):
        def q_hook(module, input, output):
            self.q = output.detach()
        def k_hook(module, input, output):
            self.k = output.detach()

        self._hooks.append(attn_module.q_proj.register_forward_hook(q_hook))
        self._hooks.append(attn_module.k_proj.register_forward_hook(k_hook))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def combined_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_cap: QKCapture | None = None,
    teacher_cap: QKCapture | None = None,
    tau: float = 2.0,
    lambda_ld: float = 1.0,
    gamma_ad: float = 0.01,
    teacher_is_probs: bool = False,
) -> tuple[torch.Tensor, float, float]:
    """
    Combined distillation loss.

    Since there are no labels, CE loss is not used.
    All signal comes from matching teacher distributions.

    Returns:
        (total_loss, ld_value, ad_value)
    """
    loss_ld = logit_distillation_loss(student_logits, teacher_logits, tau=tau,
                                       teacher_is_probs=teacher_is_probs)

    loss_ad = torch.tensor(0.0, device=student_logits.device)
    if student_cap is not None and teacher_cap is not None:
        if student_cap.q is not None and teacher_cap.q is not None:
            loss_ad = attention_relation_loss(
                student_cap.q, teacher_cap.q,
                student_cap.k, teacher_cap.k,
            )

    total = lambda_ld * loss_ld + gamma_ad * loss_ad
    return total, loss_ld.item(), loss_ad.item()

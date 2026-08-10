"""Loss helpers used by SFT / KD / DPO training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean CE over positions where mask==1. logits: (B,T,V), targets/mask: (B,T)."""
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    mask = mask.to(loss.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(teacher || student) * T^2, averaged over masked positions.

    teacher_logits should be detached by caller.
    """
    t = float(temperature)
    s = student_logits / t
    tea = teacher_logits / t
    log_p_s = F.log_softmax(s, dim=-1)
    p_t = F.softmax(tea, dim=-1)
    # (B, T, V) -> (B, T)
    kl = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1) * (t * t)
    if mask is None:
        return kl.mean()
    mask = mask.to(kl.dtype)
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Standard DPO loss (token-sum log-probs already reduced per sequence)."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


def sequence_logprobs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Sum log-prob of target tokens over masked positions. Returns (B,)."""
    log_probs = F.log_softmax(logits, dim=-1)
    # gather target logprob
    tgt = targets.unsqueeze(-1)
    token_logp = log_probs.gather(-1, tgt).squeeze(-1)
    mask = mask.to(token_logp.dtype)
    return (token_logp * mask).sum(dim=-1)

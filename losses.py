"""Loss helpers used by SFT / KD / DPO / GRPO training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Loss aggregation modes shared by the GRPO policy loss and the KL penalty.
#   seq_mean   : (1/N) * sum_i  (1/|o_i|) * sum_t  -- original GRPO, length-normalized per sequence
#   token_mean : (1/sum_i |o_i|) * sum_i sum_t     -- DAPO token-level loss
#   dr_grpo    : (1/(N * L_max)) * sum_i sum_t     -- Dr. GRPO, constant normalizer (no length bias)
AGGREGATIONS = ("seq_mean", "token_mean", "dr_grpo")


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


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Log-prob of each target token. logits: (B,T,V), targets: (B,T) -> (B,T)."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def sequence_logprobs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Sum log-prob of target tokens over masked positions. Returns (B,)."""
    token_logp = token_logprobs(logits, targets)
    mask = mask.to(token_logp.dtype)
    return (token_logp * mask).sum(dim=-1)


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-position entropy of the next-token distribution. (B,T,V) -> (B,T)."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def grpo_advantages(
    rewards: torch.Tensor,
    normalize_std: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Group-relative advantages. ``rewards``: (n_groups, group_size) -> same shape.

    This is the piece that replaces PPO's learned critic: the baseline is the
    mean reward of the other samples drawn for the *same* prompt.

    ``normalize_std=True`` reproduces GRPO (z-score). ``False`` reproduces
    Dr. GRPO, which drops the std divisor because it up-weights groups whose
    rewards happen to have low variance (i.e. very hard or very easy prompts).
    """
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be (n_groups, group_size), got {tuple(rewards.shape)}")
    rewards = rewards.float()
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    if not normalize_std:
        return centered
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return centered / (std + eps)


def zero_variance_groups(rewards: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
    """(n_groups,) bool mask of groups where every rollout got the same reward.

    Those groups have all-zero advantages and contribute no gradient — DAPO's
    dynamic sampling drops them so the compute is not wasted.
    """
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be (n_groups, group_size), got {tuple(rewards.shape)}")
    spread = rewards.float().max(dim=-1).values - rewards.float().min(dim=-1).values
    return spread.abs() <= tol


def _masked_aggregate(
    per_token: torch.Tensor,
    mask: torch.Tensor,
    aggregation: str,
    normalizer: Optional[float] = None,
    max_completion_len: Optional[int] = None,
) -> torch.Tensor:
    """Reduce a (N,T) per-token quantity to a scalar under the chosen convention.

    ``normalizer`` overrides the denominator so a rollout batch can be split
    into micro-batches: pass the whole batch's denominator to every chunk and
    the summed chunk losses equal the single-pass loss.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    mask = mask.to(per_token.dtype)
    if aggregation == "seq_mean":
        per_seq = (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        denom = float(per_token.size(0)) if normalizer is None else float(normalizer)
        return per_seq.sum() / max(denom, 1.0)

    total = (per_token * mask).sum()
    if normalizer is not None:
        denom = float(normalizer)
    elif aggregation == "token_mean":
        denom = float(mask.sum().item())
    else:
        if max_completion_len is None:
            raise ValueError("aggregation='dr_grpo' requires max_completion_len")
        denom = float(per_token.size(0) * max_completion_len)
    return total / max(denom, 1.0)


def grpo_policy_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.2,
    aggregation: str = "seq_mean",
    normalizer: Optional[float] = None,
    max_completion_len: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    """Clipped group-relative policy-gradient loss.

    ``logprobs`` / ``old_logprobs`` / ``mask``: (N, T) over completion tokens.
    ``advantages``: (N,), one scalar per rollout (constant along the sequence).

    ``clip_eps_low != clip_eps_high`` gives DAPO's "clip-higher", which leaves
    more room for low-probability tokens to grow before the update is clipped.
    """
    if advantages.dim() != 1 or advantages.size(0) != logprobs.size(0):
        raise ValueError(
            f"advantages must be (N,) matching logprobs batch {logprobs.size(0)}, "
            f"got {tuple(advantages.shape)}"
        )
    ratio = torch.exp(logprobs - old_logprobs)
    adv = advantages.to(ratio.dtype).unsqueeze(-1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high) * adv
    per_token = -torch.min(unclipped, clipped)

    loss = _masked_aggregate(per_token, mask, aggregation, normalizer, max_completion_len)

    with torch.no_grad():
        mask_f = mask.to(ratio.dtype)
        n_tokens = mask_f.sum().clamp_min(1.0)
        metrics = {
            "policy_loss": float(loss.detach()),
            "ratio_mean": float((ratio * mask_f).sum() / n_tokens),
            "clip_frac": float(((clipped < unclipped).to(ratio.dtype) * mask_f).sum() / n_tokens),
        }
    return loss, metrics


def approx_kl(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    aggregation: str = "seq_mean",
    normalizer: Optional[float] = None,
    max_completion_len: Optional[int] = None,
) -> torch.Tensor:
    """Schulman's low-variance k3 estimator of KL(policy || reference), masked.

    Per token: ``exp(r) - r - 1`` with ``r = log pi_ref - log pi_theta``. It is
    non-negative by construction, unlike the plain ``-r`` estimator.
    """
    log_ratio = ref_logprobs - logprobs
    per_token = torch.exp(log_ratio) - log_ratio - 1.0
    return _masked_aggregate(per_token, mask, aggregation, normalizer, max_completion_len)

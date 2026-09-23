"""Held-out evaluation used inside the training loops.

Training loss alone cannot tell you whether a run is any good, and across a
data-mixture ablation it cannot even be compared: two mixtures tokenize to
different distributions, so their training losses are on different scales.
Every stage therefore reports a metric on a held-out split.

Token-weighted, not batch-averaged. Averaging per-batch means would weight a
batch of short documents the same as a batch of long ones, which quietly
changes the number as the batch composition changes.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Iterable, Optional

import torch

from losses import sequence_logprobs, token_logprobs


@torch.no_grad()
def evaluate_lm(
    model,
    loader: Iterable,
    device: str,
    ctx: Any = None,
    max_batches: Optional[int] = None,
) -> dict:
    """Cross-entropy and perplexity over a held-out loader, token-weighted.

    Works for any loader yielding ``(inputs, targets, loss_mask)``, so pretrain
    and SFT share it — SFT's mask simply restricts the loss to assistant spans.
    """
    ctx = ctx if ctx is not None else nullcontext()
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    batches = 0
    try:
        for batch in loader:
            if max_batches is not None and batches >= max_batches:
                break
            X, Y, mask = (t.to(device) for t in batch[:3])
            with ctx:
                logits = model(X).logits
            token_loss = -token_logprobs(logits, Y)
            mask = mask.to(token_loss.dtype)
            total_loss += float((token_loss * mask).sum())
            total_tokens += int(mask.sum())
            batches += 1
    finally:
        if was_training:
            model.train()

    if total_tokens == 0:
        return {"loss": float("nan"), "ppl": float("nan"), "tokens": 0, "batches": 0}
    mean = total_loss / total_tokens
    return {
        "loss": mean,
        # Guard the exponential: a diverged run would otherwise raise instead of
        # recording the very number that shows it diverged.
        "ppl": math.exp(mean) if mean < 60 else float("inf"),
        "tokens": total_tokens,
        "batches": batches,
    }


@torch.no_grad()
def evaluate_preference(
    policy,
    ref,
    loader: Iterable,
    device: str,
    beta: float,
    ctx: Any = None,
    max_batches: Optional[int] = None,
) -> dict:
    """DPO metrics on held-out pairs.

    ``accuracy`` is the share of pairs the policy ranks correctly and is the
    metric to watch: DPO loss keeps falling while the model merely sharpens an
    ordering it already had, so loss alone overstates progress.

    ``margin`` is the mean implicit-reward gap, and ``chosen``/``rejected``
    reward shifts show *how* the gap moved — a gap that widens purely by
    pushing the rejected branch down is a different behaviour from one that
    lifts the chosen branch.
    """
    ctx = ctx if ctx is not None else nullcontext()
    was_training = policy.training
    policy.eval()
    correct = pairs = 0
    margin_sum = chosen_sum = rejected_sum = 0.0
    batches = 0
    try:
        for batch in loader:
            if max_batches is not None and batches >= max_batches:
                break
            cX, cY, cM, rX, rY, rM = (t.to(device) for t in batch)
            with ctx:
                policy_chosen = sequence_logprobs(policy(cX).logits, cY, cM)
                policy_rejected = sequence_logprobs(policy(rX).logits, rY, rM)
                ref_chosen = sequence_logprobs(ref(cX).logits, cY, cM)
                ref_rejected = sequence_logprobs(ref(rX).logits, rY, rM)

            chosen_reward = beta * (policy_chosen - ref_chosen)
            rejected_reward = beta * (policy_rejected - ref_rejected)
            margin = chosen_reward - rejected_reward

            correct += int((margin > 0).sum())
            pairs += margin.numel()
            margin_sum += float(margin.sum())
            chosen_sum += float(chosen_reward.sum())
            rejected_sum += float(rejected_reward.sum())
            batches += 1
    finally:
        if was_training:
            policy.train()

    if pairs == 0:
        return {"accuracy": float("nan"), "margin": float("nan"), "pairs": 0, "batches": 0}
    return {
        "accuracy": correct / pairs,
        "margin": margin_sum / pairs,
        "chosen_reward": chosen_sum / pairs,
        "rejected_reward": rejected_sum / pairs,
        "pairs": pairs,
        "batches": batches,
    }

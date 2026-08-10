"""Shared training helpers: seed, AMP, LR schedule, checkpoint I/O."""

from __future__ import annotations

import math
import os
import random
from contextlib import nullcontext
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "y", "1"):
        return True
    if s in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(f"Cannot parse boolean from {v!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_autocast_scaler(device: str, dtype: str):
    """Return (autocast_context, GradScaler|None).

    GradScaler is only used for fp16 on CUDA. bf16 does not need loss scaling.
    """
    dtype = dtype.lower()
    use_cuda = "cuda" in device and torch.cuda.is_available()
    if use_cuda and dtype in ("float16", "fp16", "bfloat16", "bf16"):
        amp_dtype = torch.float16 if dtype in ("float16", "fp16") else torch.bfloat16
        ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        ctx = nullcontext()

    use_scaler = use_cuda and dtype in ("float16", "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None
    return ctx, scaler


def get_lr(
    step: int,
    total_steps: int,
    lr: float,
    warmup_ratio: float = 0.1,
) -> float:
    """Cosine decay with linear warmup. ``step`` is 1-based optimizer-update index."""
    if total_steps <= 0:
        return lr
    step = max(1, min(step, total_steps))
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step <= warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    min_lr = 0.1 * lr
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def optimizer_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Any,
    grad_clip: float,
) -> None:
    """Unscale (if scaler), clip grad norm, step, update, zero_grad."""
    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def flush_pending_grads(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Any,
    grad_clip: float,
    pending: bool,
) -> bool:
    """If ``pending`` (leftover grads from a partial accumulation window at epoch end),
    run ``optimizer_step`` and return ``False``. Otherwise return ``pending`` unchanged.
    """
    if pending:
        optimizer_step(model, optimizer, scaler, grad_clip)
        return False
    return pending


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scaler: Any,
    epoch: int,
    step: int,
    global_step: int,
    loss: float,
    config: Any,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "loss": loss,
        "config": getattr(config, "__dict__", config),
    }
    torch.save(payload, path)


def _is_wrapped_checkpoint(obj: Any) -> bool:
    return isinstance(obj, dict) and "model_state_dict" in obj


def load_weights(
    path: str,
    model: nn.Module,
    device: str,
    strict: bool = False,
) -> dict:
    """Load weights from a raw state_dict or a training checkpoint dict."""
    obj = torch.load(path, map_location=device, weights_only=False)
    if _is_wrapped_checkpoint(obj):
        state = obj["model_state_dict"]
    elif isinstance(obj, dict):
        # Heuristic: tensor values => state_dict
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            state = obj
        elif "state_dict" in obj:
            state = obj["state_dict"]
        else:
            # Might be a wrapped dict without our key — try filtering tensor entries
            state = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
            if not state:
                raise ValueError(f"Unrecognized checkpoint format: {path}")
    else:
        raise ValueError(f"Unrecognized checkpoint type: {type(obj)}")

    state = {k: v for k, v in state.items() if "mask" not in k}
    model.load_state_dict(state, strict=strict)
    return obj if isinstance(obj, dict) else {"model_state_dict": state}


def load_train_state(
    checkpoint: dict,
    optimizer: Optional[optim.Optimizer],
    scaler: Any,
) -> Tuple[int, int, int, float]:
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return (
        int(checkpoint.get("epoch", 0)),
        int(checkpoint.get("step", 0)),
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("loss", float("inf"))),
    )

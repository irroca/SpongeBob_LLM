"""Shared training helpers: seed, AMP, LR schedule, checkpoint I/O, CLI/wandb."""

from __future__ import annotations

import argparse
import math
import os
import random
from contextlib import nullcontext
from typing import Any, Optional, Sequence, Tuple

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
    run ``optimizer_step`` and return ``True`` (flushed). Otherwise return ``False``.
    """
    if pending:
        optimizer_step(model, optimizer, scaler, grad_clip)
        return True
    return False


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


def add_common_train_args(
    parser: argparse.ArgumentParser,
    *,
    save_dir: str = "results",
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    dtype: str = "float32",
    num_workers: int = 0,
    accumulation_steps: int = 1,
    grad_clip: float = 1.0,
    log_step: int = 10,
    save_step: int = 1000,
    max_seq_len: int = 512,
    data_path: str = "datasets/pretrain.jsonl",
    resume_from: Optional[str] = None,
    seed: int = 1337,
    wandb_project: str = "SpongeBob",
    skip: Sequence[str] = (),
) -> argparse.ArgumentParser:
    """Add the CLI flags shared by every training entry point (pretrain/SFT/distill/dpo/grpo).

    Each script passes its own defaults via keyword args (e.g. ``learning_rate``,
    ``wandb_project``, ``data_path``) and then adds any stage-specific extras
    (``--pretrained_path``, ``--teacher_path``, ``--beta``, ...) after calling this.
    ``--device`` always defaults to ``"cuda"`` if a GPU is available, else ``"cpu"``.

    ``skip`` drops flags that make no sense for a stage rather than letting it
    define its own copy: GRPO is driven by ``--rl_steps`` over env-sampled
    prompts, so it has no epochs and no DataLoader-style gradient accumulation.
    """
    common = {
        "save_dir": dict(type=str, default=save_dir),
        "epochs": dict(type=int, default=epochs),
        "batch_size": dict(type=int, default=batch_size),
        "learning_rate": dict(type=float, default=learning_rate),
        "device": dict(type=str, default="cuda" if torch.cuda.is_available() else "cpu"),
        "use_wandb": dict(type=str2bool, default=False),
        "wandb_project": dict(type=str, default=wandb_project),
        "dtype": dict(type=str, default=dtype),
        "num_workers": dict(type=int, default=num_workers),
        "accumulation_steps": dict(type=int, default=accumulation_steps),
        "grad_clip": dict(type=float, default=grad_clip),
        "log_step": dict(type=int, default=log_step),
        "save_step": dict(type=int, default=save_step),
        "max_seq_len": dict(type=int, default=max_seq_len),
        "data_path": dict(type=str, default=data_path),
        "resume_from": dict(type=str, default=resume_from),
        "seed": dict(type=int, default=seed),
    }
    unknown = [name for name in skip if name not in common]
    if unknown:
        raise ValueError(f"skip contains unknown common args: {unknown}")
    for name, kwargs in common.items():
        if name not in skip:
            parser.add_argument(f"--{name}", **kwargs)
    return parser


def init_wandb_if_needed(args: Any, run_name: Optional[str] = None) -> Any:
    """Initialize wandb (via the ``swanlab`` shim) when ``args.use_wandb`` is set.

    Returns the wandb-like module on success, or ``None`` when logging is disabled.
    The import is lazy: ``swanlab`` is an optional dependency (see requirements.txt)
    and is only required if a caller actually passes ``--use_wandb True``.
    """
    if not getattr(args, "use_wandb", False):
        return None
    import swanlab as wandb  # noqa: F811

    name = run_name or f"run-bs{getattr(args, 'batch_size', '?')}"
    wandb.init(project=args.wandb_project, name=name, config=vars(args))
    return wandb

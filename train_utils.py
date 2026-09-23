"""Shared training helpers: seed, AMP, LR schedule, checkpoint I/O, CLI/wandb."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import warnings
from contextlib import nullcontext
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim

from config import LLMConfig


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


_AMP_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def resolve_device(preferred: Optional[str] = None) -> str:
    """Pick an accelerator: explicit request > cuda > mps > cpu.

    Apple Silicon is worth auto-selecting: measured on an M5 Pro, a ~100M model
    trains at 5.7k token/s on ``mps`` against 1.4k on ``cpu``, and 9.1k with
    bf16 autocast. Falling back to ``cpu`` there silently costs ~6x.
    """
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def accelerator_type(device: str) -> Optional[str]:
    """``"cuda"`` / ``"mps"`` for a usable accelerator, else ``None``."""
    if "cuda" in device and torch.cuda.is_available():
        return "cuda"
    if device.startswith("mps") and mps_available():
        return "mps"
    return None


def build_autocast_scaler(device: str, dtype: str):
    """Return (autocast_context, GradScaler|None).

    GradScaler exists to keep fp16 gradients from underflowing, so it is only
    created for fp16. bf16 has fp32's exponent range and needs no scaling —
    that is why ``--dtype bfloat16`` correctly produces no scaler.

    **Prefer bf16 over fp16 on MPS.** Measured on an M5 Pro with a 100M model,
    fp16 autocast moved the loss from -0.3278 to +0.0018 while bf16 held at
    -0.3276; fp16's narrow exponent range does not survive this model's
    attention path on that backend.
    """
    dtype = dtype.lower()
    backend = accelerator_type(device)
    amp_dtype = _AMP_DTYPES.get(dtype)

    ctx = (
        torch.amp.autocast(backend, dtype=amp_dtype)
        if backend and amp_dtype is not None
        else nullcontext()
    )
    use_scaler = backend is not None and amp_dtype is torch.float16
    scaler = torch.amp.GradScaler(backend, enabled=True) if use_scaler else None
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
) -> float:
    """Unscale (if scaler), clip grad norm, step, update, zero_grad.

    Returns the pre-clip gradient norm. GRPO needs it: its on-policy loss value
    is uninformative (see README), so the gradient norm is what shows whether
    an update carried any signal.
    """
    if scaler is not None:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm)


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
    result = model.load_state_dict(state, strict=strict)
    missing = [k for k in getattr(result, "missing_keys", []) if "mask" not in k]
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        # strict=False raises on shape mismatch but silently tolerates absent keys, so a
        # checkpoint from a different depth would leave whole layers randomly initialized.
        warnings.warn(
            f"{path}: architecture does not match the checkpoint. "
            f"{len(missing)} missing key(s) stayed randomly initialized "
            f"(e.g. {missing[:3]}), {len(unexpected)} unused key(s) (e.g. {unexpected[:3]}). "
            "Pass the checkpoint path so the architecture can be resolved from it, or set "
            "the --dim/--n_layers/... flags to match.",
            RuntimeWarning,
            stacklevel=2,
        )
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


MODEL_ARCH_FIELDS = (
    "dim",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "hidden_dim",
    "multiple_of",
    "norm_eps",
    "rope_theta",
    "dropout",
)


def model_arch_defaults() -> dict:
    """Architecture defaults read straight off ``LLMConfig.__init__``.

    Keeps the CLI from drifting away from the library default when the config changes.
    """
    params = inspect.signature(LLMConfig.__init__).parameters
    return {
        name: params[name].default
        for name in MODEL_ARCH_FIELDS
        if name in params and params[name].default is not inspect.Parameter.empty
    }


DEFAULT_TOKENIZER_PATH = "./tokenizer/zh_6400"


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the flags every entry point that *builds a model* needs: tokenizer + architecture.

    Every architecture flag defaults to ``None`` meaning "not specified", which is
    what lets :func:`resolve_model_config` tell an explicit request apart from a
    fallback and apply the precedence CLI > checkpoint > library default.
    """
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=DEFAULT_TOKENIZER_PATH,
        help=f"default: {DEFAULT_TOKENIZER_PATH}; vocab_size is read from it",
    )
    defaults = model_arch_defaults()
    types = {
        "dim": int,
        "n_layers": int,
        "n_heads": int,
        "n_kv_heads": int,
        "hidden_dim": int,
        "multiple_of": int,
        "norm_eps": float,
        "rope_theta": float,
        "dropout": float,
    }
    group = parser.add_argument_group("model architecture")
    for name in MODEL_ARCH_FIELDS:
        group.add_argument(
            f"--{name}",
            type=types[name],
            default=None,
            help=f"default: {defaults.get(name)} (or the value stored in the loaded checkpoint)",
        )
    return parser


def _state_dict_of(obj: Any) -> Optional[dict]:
    if _is_wrapped_checkpoint(obj):
        return obj["model_state_dict"]
    if isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    return None


def infer_arch_from_state_dict(state: dict) -> dict:
    """Recover what the tensor shapes can prove about a checkpoint's architecture.

    ``n_heads`` is **not** recoverable: ``wq`` is always ``(dim, dim)`` because
    ``head_dim = dim // n_heads``. Only the kv/q head *ratio* is visible, so this
    returns ``kv_q_ratio`` and leaves ``n_heads``/``n_kv_heads`` to the caller.
    """
    arch: dict = {}
    emb = state.get("tok_embeddings.weight")
    if emb is not None:
        arch["vocab_size"], arch["dim"] = int(emb.shape[0]), int(emb.shape[1])
    layer_ids = [
        int(k.split(".")[1]) for k in state if k.startswith("layers.") and k.split(".")[1].isdigit()
    ]
    if layer_ids:
        arch["n_layers"] = max(layer_ids) + 1
    w1 = state.get("layers.0.feed_forward.w1.weight")
    if w1 is not None:
        arch["hidden_dim"] = int(w1.shape[0])
    wq, wk = state.get("layers.0.attention.wq.weight"), state.get("layers.0.attention.wk.weight")
    if wq is not None and wk is not None and wq.shape[0]:
        arch["kv_q_ratio"] = wk.shape[0] / wq.shape[0]
    return arch


def config_sidecar_path(weights_path: str) -> str:
    """Where the architecture of a raw ``*_final.pth`` is recorded."""
    return f"{os.path.splitext(weights_path)[0]}.config.json"


def save_final_weights(path: str, model: nn.Module, config: LLMConfig) -> None:
    """Save a stage's final weights as a plain ``state_dict`` plus a config sidecar.

    The weights file stays a bare ``state_dict`` so anything that already reads it
    keeps working, but ``n_heads`` cannot be recovered from tensor shapes alone
    (``wq`` is always ``dim x dim``), so the architecture goes next to it in JSON.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    arch = {name: getattr(config, name) for name in (*MODEL_ARCH_FIELDS, "vocab_size")}
    arch["max_seq_len"] = config.max_seq_len
    with open(config_sidecar_path(path), "w", encoding="utf-8") as fh:
        json.dump(arch, fh, ensure_ascii=False, indent=2)


def read_checkpoint_arch(path: str) -> dict:
    """Architecture recorded in (or implied by) a checkpoint.

    Three sources, in order of authority: the config dict inside a training
    checkpoint, the ``*.config.json`` sidecar next to final weights, and finally
    the tensor shapes. Shapes alone cannot pin down ``n_heads``, so when only
    they are available the caller is warned rather than silently given a guess.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    arch: dict = {}
    if isinstance(obj, dict) and isinstance(obj.get("config"), dict):
        saved = obj["config"]
        arch = {k: saved[k] for k in (*MODEL_ARCH_FIELDS, "vocab_size") if k in saved}

    sidecar = config_sidecar_path(path)
    if not arch and os.path.exists(sidecar):
        with open(sidecar, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        arch = {k: saved[k] for k in (*MODEL_ARCH_FIELDS, "vocab_size") if k in saved}

    state = _state_dict_of(obj)
    if state is not None:
        inferred = infer_arch_from_state_dict(state)
        # Shapes beat a recorded config: the tensors are what will actually be loaded.
        arch.update({k: v for k, v in inferred.items() if k != "kv_q_ratio"})
        if "kv_q_ratio" in inferred and "n_kv_heads" not in arch:
            arch["kv_q_ratio"] = inferred["kv_q_ratio"]
    return arch


def resolve_model_config(
    args: Any,
    vocab_size: int,
    checkpoint_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
) -> LLMConfig:
    """Build an ``LLMConfig`` with precedence: explicit CLI > checkpoint > library default.

    Passing ``checkpoint_path`` is what makes "train a 6-layer model, then SFT it"
    work without restating every architecture flag — and what turns a tokenizer
    or depth mismatch into an error instead of a half-loaded model.
    """
    settings = model_arch_defaults()
    ckpt_arch: dict = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt_arch = read_checkpoint_arch(checkpoint_path)
        ckpt_vocab = ckpt_arch.pop("vocab_size", None)
        if ckpt_vocab is not None and int(ckpt_vocab) != int(vocab_size):
            raise ValueError(
                f"{checkpoint_path} was trained with vocab_size={ckpt_vocab} but the tokenizer "
                f"has {vocab_size}. Retraining the tokenizer invalidates old checkpoints; "
                "use the matching tokenizer or start from scratch."
            )
        ratio = ckpt_arch.pop("kv_q_ratio", None)
        settings.update({k: v for k, v in ckpt_arch.items() if k in MODEL_ARCH_FIELDS})
        if ratio is not None and getattr(args, "n_kv_heads", None) is None:
            n_heads = getattr(args, "n_heads", None) or settings["n_heads"]
            settings["n_kv_heads"] = max(1, round(n_heads * ratio))
            warnings.warn(
                f"{checkpoint_path} records no architecture, so n_heads={n_heads} was assumed "
                f"and n_kv_heads={settings['n_kv_heads']} derived from the kv/q shape ratio. "
                "The tensor shapes will load either way, but head grouping changes the model: "
                "pass --n_heads explicitly if the checkpoint used a different value.",
                RuntimeWarning,
                stacklevel=2,
            )

    explicit = {
        name: getattr(args, name)
        for name in MODEL_ARCH_FIELDS
        if getattr(args, name, None) is not None
    }
    settings.update(explicit)

    if max_seq_len is None:
        max_seq_len = getattr(args, "max_seq_len", None)
    return LLMConfig(vocab_size=vocab_size, max_seq_len=max_seq_len, **settings)


def describe_model(model: nn.Module, config: LLMConfig, label: str = "model") -> str:
    """One-line architecture + parameter summary, deduplicating tied weights."""
    seen, total = set(), 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return (
        f"{label}: dim={config.dim} layers={config.n_layers} heads={config.n_heads} "
        f"kv_heads={config.n_kv_heads} hidden={config.hidden_dim} vocab={config.vocab_size} "
        f"seq={config.max_seq_len} -> {total / 1e6:.3f}M params"
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
    val_data_path: str = "",
    val_every: int = 200,
    val_batches: int = 20,
    resume_from: Optional[str] = None,
    seed: int = 1337,
    wandb_project: str = "Whetstone",
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
        "device": dict(type=str, default=resolve_device()),
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
        "val_data_path": dict(
            type=str,
            default=val_data_path,
            help="Held-out JSONL. Training loss is not comparable across data "
                 "mixtures; this is what an ablation should be judged on",
        ),
        "val_every": dict(type=int, default=val_every, help="Optimizer steps between evaluations"),
        "val_batches": dict(type=int, default=val_batches, help="Cap batches per evaluation; 0 = all"),
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


def build_val_loader(dataset_cls, args, tokenizer, **dataset_kwargs):
    """DataLoader over ``--val_data_path``, or ``None`` when none was given.

    Never shuffled: the evaluation set must be identical between runs and
    between evaluations within a run, or the curve measures the sampler.
    """
    from torch.utils.data import DataLoader

    path = getattr(args, "val_data_path", "")
    if not path or not os.path.exists(path):
        return None
    dataset = dataset_cls(path, tokenizer, max_length=args.max_seq_len, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 0),
    )


def should_evaluate(global_step: int, args: Any) -> bool:
    val_every = getattr(args, "val_every", 0)
    return bool(val_every) and global_step > 0 and global_step % val_every == 0


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

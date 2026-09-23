"""Per-run training records: everything needed to review a run months later.

A training run leaves three files in ``{save_dir}/runs/{run_id}/``:

``meta.json``
    Written once at startup. The full CLI args, the resolved model
    architecture and parameter count, the environment (python/torch/device/
    dtype/host), the git commit and whether the tree was dirty, and a
    fingerprint of every input file. This is what makes a run *reproducible* —
    a metric curve with no record of which data and which commit produced it
    cannot be reviewed, only admired.

``metrics.jsonl``
    One JSON object per logged point, appended as training proceeds. Append-only
    so a crashed run keeps everything up to the crash. Each row carries
    ``split`` (``train``/``val``), the step, wall-clock elapsed, cumulative
    tokens and throughput, plus whatever the stage logs.

``summary.json``
    Written on exit, including on failure: status, duration, totals, the last
    and best values of each metric, and the traceback if it crashed. A run that
    died at 3am should say so rather than just stopping.

Deliberately dependency-free — ``swanlab``/``wandb`` remain optional and
orthogonal. Read the results with ``analyze_runs.py``.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

FINGERPRINT_BYTES = 1 << 20  # hash a 1 MiB head, not a 30GB corpus
RUNS_DIRNAME = "runs"


def _git_info() -> dict:
    """Commit, branch and dirty flag; empty when git is unavailable."""

    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {}
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def file_fingerprint(path: str) -> Optional[dict]:
    """Identify a data file without reading all of it.

    Full hashing a multi-GB corpus on every run is not worth the minutes, so
    this is size + mtime + a hash of the first megabyte. Enough to notice that
    the data changed between two ablation runs, which is the actual question.
    """
    import hashlib

    if not path or not os.path.isfile(path):
        return None
    stat = os.stat(path)
    digest = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        digest.update(fh.read(FINGERPRINT_BYTES))
    return {
        "path": path,
        "bytes": stat.st_size,
        "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
        "head_sha": digest.hexdigest(),
    }


def _dataset_manifest(paths: Sequence[str]) -> Optional[dict]:
    """The ``datatools.prepare`` manifest beside the data, if there is one.

    Links a run to the exact mixture that produced its corpus, which is the
    difference between "ablation run 3" and "ablation run 3, 30% Chinese".
    """
    for path in paths:
        if not path:
            continue
        candidate = os.path.join(os.path.dirname(path) or ".", "manifest.json")
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            return {
                "path": candidate,
                "mixture": manifest.get("mixture"),
                "tokens": manifest.get("total_tokens_collected"),
                "sources": [s.get("name") for s in manifest.get("sources", [])],
            }
    return None


def _environment(device: str) -> dict:
    import torch

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "device": device,
    }
    if "cuda" in device and torch.cuda.is_available():
        info["accelerator"] = torch.cuda.get_device_name(0)
    elif device.startswith("mps"):
        info["accelerator"] = platform.processor() or "Apple Silicon"
    return info


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass
class RunRecorder:
    """Records one training run. Use via :meth:`start` as a context manager."""

    run_dir: str
    stage: str
    run_id: str
    started: float = field(default_factory=time.perf_counter)
    tokens: int = 0
    rows: int = 0
    _last: dict = field(default_factory=dict)
    _best: dict = field(default_factory=dict)
    _handle: Any = None
    _finished: bool = False

    # Bookkeeping fields, not objectives: "best lr" is just the warmup peak and
    # "best epoch" is just the last one, so tracking extrema for them is noise.
    NOT_OBJECTIVES = frozenset({
        "epoch", "lr", "step", "tokens", "batches", "pairs", "sec",
        "grad_norm", "ratio_mean", "groups_used", "elapsed_s",
    })
    # Everything else is a maximum unless it reads like a loss.
    LOWER_IS_BETTER = frozenset({"loss", "ppl", "ce", "kd", "dpo_loss", "kl", "hack_rate"})

    @classmethod
    def start(
        cls,
        stage: str,
        args: Any,
        config: Any = None,
        model: Any = None,
        data_paths: Sequence[str] = (),
        extra: Optional[dict] = None,
    ) -> "RunRecorder":
        save_dir = getattr(args, "save_dir", "results")
        run_id = f"{stage}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_dir = os.path.join(save_dir, RUNS_DIRNAME, run_id)
        os.makedirs(run_dir, exist_ok=True)

        meta = {
            "run_id": run_id,
            "stage": stage,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "command": " ".join(sys.argv),
            "args": {
                k: _jsonable(v)
                for k, v in vars(args).items()
                if k not in ("lm_config",)
            },
            "env": _environment(getattr(args, "device", "cpu")),
            "git": _git_info(),
            "data": [
                f
                for f in (file_fingerprint(p) for p in dict.fromkeys(p for p in data_paths if p))
                if f
            ],
            # Namespaced rather than merged: a stage passing extra={"env": ...}
            # would otherwise silently overwrite the recorded environment.
            "extra": dict(extra or {}),
        }
        if config is not None:
            meta["model"] = {
                k: _jsonable(getattr(config, k, None))
                for k in (
                    "dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim",
                    "vocab_size", "max_seq_len", "dropout", "rope_theta",
                )
            }
        if model is not None:
            unique = {p.data_ptr(): p.numel() for p in model.parameters()}
            meta.setdefault("model", {})["params"] = sum(unique.values())
        manifest = _dataset_manifest(data_paths)
        if manifest:
            meta["dataset_manifest"] = manifest

        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        recorder = cls(run_dir=run_dir, stage=stage, run_id=run_id)
        recorder._handle = open(os.path.join(run_dir, "metrics.jsonl"), "a", encoding="utf-8")
        return recorder

    # -- recording ---------------------------------------------------------

    def add_tokens(self, count: int) -> None:
        self.tokens += int(count)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def log(self, step: int, split: str = "train", **metrics: Any) -> dict:
        """Append one point. Throughput and elapsed time are added automatically."""
        elapsed = self.elapsed
        row = {
            "step": int(step),
            "split": split,
            "elapsed_s": round(elapsed, 3),
            "tokens": self.tokens,
            "tokens_per_s": round(self.tokens / elapsed, 1) if elapsed > 0 else 0.0,
            **{k: _jsonable(v) for k, v in metrics.items()},
        }
        if self._handle is not None:
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._handle.flush()  # a killed run keeps every point up to the kill
        self.rows += 1
        self._remember(split, metrics)
        return row

    def log_eval(self, step: int, **metrics: Any) -> dict:
        return self.log(step, split="val", **metrics)

    def _remember(self, split: str, metrics: dict) -> None:
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            name = key if split == "train" else f"{split}_{key}"
            self._last[name] = value
            if key in self.NOT_OBJECTIVES:
                continue
            lower = key in self.LOWER_IS_BETTER or key.endswith("loss") or key.endswith("ppl")
            current = self._best.get(name)
            if current is None or (value < current if lower else value > current):
                self._best[name] = value

    # -- lifecycle ---------------------------------------------------------

    def finish(self, status: str = "completed", error: Optional[str] = None, **extra: Any) -> dict:
        if self._finished:
            return {}
        self._finished = True
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        summary = {
            "run_id": self.run_id,
            "stage": self.stage,
            "status": status,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_s": round(self.elapsed, 2),
            "logged_points": self.rows,
            "tokens": self.tokens,
            "tokens_per_s": round(self.tokens / self.elapsed, 1) if self.elapsed > 0 else 0.0,
            "last": self._last,
            "best": self._best,
            **{k: _jsonable(v) for k, v in extra.items()},
        }
        if error:
            summary["error"] = error
        with open(os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        return summary

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is KeyboardInterrupt:
            self.finish(status="interrupted")
        elif exc_type is not None:
            self.finish(
                status="failed",
                error="".join(traceback.format_exception(exc_type, exc, tb))[-4000:],
            )
        else:
            self.finish(status="completed")
        return False  # never swallow the exception


# -- reading -------------------------------------------------------------


def read_run(run_dir: str) -> dict:
    """Load one run's meta, metrics and summary. Missing pieces come back empty."""

    def load_json(name: str) -> dict:
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}

    metrics = []
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a run killed mid-write leaves a partial last line
    return {
        "run_dir": run_dir,
        "meta": load_json("meta.json"),
        "summary": load_json("summary.json"),
        "metrics": metrics,
    }


def discover_runs(root: str) -> list[str]:
    """Run directories under ``root``, oldest first.

    ``root`` may be a save_dir, its ``runs/`` subdirectory, or a run directory.
    """
    if os.path.isfile(os.path.join(root, "meta.json")):
        return [root]
    base = os.path.join(root, RUNS_DIRNAME)
    if not os.path.isdir(base):
        base = root
    if not os.path.isdir(base):
        return []
    found = [
        os.path.join(base, name)
        for name in sorted(os.listdir(base))
        if os.path.isfile(os.path.join(base, name, "meta.json"))
    ]
    return found

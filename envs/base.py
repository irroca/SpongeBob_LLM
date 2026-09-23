"""Task / reward interfaces shared by every RLVR environment.

An environment owns two things and nothing else:

1. **Prompts** — how a task is sampled and rendered into a chat message.
2. **Rewards** — how a raw model completion is turned into a scalar, with the
   individual reward components kept separate so training can log them.

Everything here is pure Python (no torch), so reward rules stay unit-testable
without a model.
"""

from __future__ import annotations

import abc
import json
import random
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Task:
    """One verifiable problem instance."""

    question: str
    answer: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Reward:
    """Reward breakdown for a single completion.

    ``total`` is what GRPO optimizes; the components exist so training can log
    accuracy and format separately and detect format-only reward hacking.
    """

    total: float
    accuracy: float
    format: float
    format_ok: bool
    correct: bool
    parsed_answer: str | None = None

    @property
    def hacked_format(self) -> bool:
        """Well-formed output with a wrong answer: the format reward alone was collected."""
        return self.format_ok and not self.correct


class TaskEnv(abc.ABC):
    """Base class for verifiable-reward environments."""

    name: str = "task"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    @property
    def system_prompt(self) -> str | None:
        """Optional system message; ``None`` falls back to the tokenizer template default."""
        return None

    @abc.abstractmethod
    def sample_task(self) -> Task:
        """Draw a single task from this environment's own RNG."""

    @abc.abstractmethod
    def render(self, task: Task) -> str:
        """Render a task as the user-turn content shown to the policy."""

    @abc.abstractmethod
    def score(self, completion: str, task: Task) -> Reward:
        """Score one raw completion string against the task's ground truth."""

    def sample(self, n: int) -> list[Task]:
        return [self.sample_task() for _ in range(n)]

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)


_REGISTRY: dict[str, type[TaskEnv]] = {}


def register_env(cls: type[TaskEnv]) -> type[TaskEnv]:
    _REGISTRY[cls.name] = cls
    return cls


def make_env(name: str, **kwargs: Any) -> TaskEnv:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown env {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_envs() -> list[str]:
    return sorted(_REGISTRY)


def load_tasks(path: str) -> list[Task]:
    """Read a fixed task set from JSONL (``{"question": ..., "answer": ...}`` per line).

    Used for held-out evaluation so accuracy curves are measured on the same
    problems across runs instead of a freshly sampled set every time.
    """
    tasks = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = {k: v for k, v in row.items() if k not in ("question", "answer")}
            tasks.append(Task(question=row["question"], answer=str(row["answer"]), meta=meta))
    return tasks

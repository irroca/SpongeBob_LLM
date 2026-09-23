"""Verifiable arithmetic environment (RLVR).

The policy must answer in a fixed shape::

    <think>任意推理过程</think><answer>42</answer>

Two rule-based rewards, no reward model and no human labels:

* **accuracy** — the integer inside ``<answer>`` equals the ground truth.
* **format** — both tag pairs are present, closed, and ``<think>`` precedes
  ``<answer>``.

Accuracy is gated on a parseable ``<answer>`` tag by default (``strict=True``),
so the format reward is a prerequisite rather than a free bonus. Setting
``strict=False`` falls back to "last integer anywhere in the completion", which
is the knob for measuring how much of a low score is a formatting failure
versus an arithmetic failure.

``envs.generate_data`` turns the same env into SFT / DPO / eval datasets so all
four post-training stages can be compared on one task.
"""

from __future__ import annotations

import random
import re
from typing import Sequence

from .base import Reward, Task, TaskEnv, register_env

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９－＋", "0123456789-+")
_INT_RE = re.compile(r"-?\d+")


def _tag_span(text: str, open_tag: str, close_tag: str) -> tuple[int, int] | None:
    """Return (content_start, content_end) of the first closed tag pair, else None."""
    start = text.find(open_tag)
    if start < 0:
        return None
    content_start = start + len(open_tag)
    end = text.find(close_tag, content_start)
    if end < 0:
        return None
    return content_start, end


def extract_tag(text: str, tag: str) -> str | None:
    """Content of the first well-formed ``<tag>...</tag>`` pair, or None."""
    span = _tag_span(text, f"<{tag}>", f"</{tag}>")
    return None if span is None else text[span[0] : span[1]]


def has_valid_format(text: str) -> bool:
    """True when think/answer are both closed and the whole think block precedes <answer>."""
    think = _tag_span(text, THINK_OPEN, THINK_CLOSE)
    answer = _tag_span(text, ANSWER_OPEN, ANSWER_CLOSE)
    if think is None or answer is None:
        return False
    think_end = think[1] + len(THINK_CLOSE)
    answer_start = answer[0] - len(ANSWER_OPEN)
    return think_end <= answer_start


def normalize_number(raw: str | None) -> str | None:
    """Normalize a model-written number: fullwidth digits, separators, stray punctuation."""
    if raw is None:
        return None
    cleaned = raw.translate(_FULLWIDTH_DIGITS).strip()
    cleaned = cleaned.replace(",", "").replace("，", "").replace(" ", "")
    cleaned = cleaned.strip("。.、;；:：")
    if not cleaned:
        return None
    match = _INT_RE.fullmatch(cleaned)
    if match is None:
        return None
    return str(int(cleaned))


def parse_answer(completion: str, strict: bool = True) -> str | None:
    """Extract the predicted integer.

    ``strict``: only read the ``<answer>`` tag. Otherwise fall back to the last
    integer anywhere in the completion.
    """
    tagged = normalize_number(extract_tag(completion, "answer"))
    if tagged is not None or strict:
        return tagged
    numbers = _INT_RE.findall(completion.translate(_FULLWIDTH_DIGITS))
    return str(int(numbers[-1])) if numbers else None


def _digit_parts(value: int) -> list[int]:
    """35 -> [30, 5]; 100 -> [100]; 0 -> [0]."""
    digits = str(abs(value))
    parts = [
        int(ch) * 10 ** (len(digits) - 1 - i)
        for i, ch in enumerate(digits)
        if ch != "0"
    ]
    return parts or [0]


def gold_completion(a: int, op: str, b: int) -> str:
    """Reference chain-of-thought + answer, used for SFT cold start and DPO pairs."""
    parts = _digit_parts(b)
    cur = a
    steps = []
    for part in parts:
        nxt = cur + part if op == "+" else cur - part
        steps.append(f"{cur} {op} {part} = {nxt}")
        cur = nxt
    if len(parts) > 1:
        head = f"把 {b} 拆成 " + " 和 ".join(str(p) for p in parts) + "："
    else:
        head = ""
    reasoning = head + "；".join(steps) + "。"
    return f"{THINK_OPEN}{reasoning}{THINK_CLOSE}{ANSWER_OPEN}{cur}{ANSWER_CLOSE}"


@register_env
class ArithmeticEnv(TaskEnv):
    """Multi-digit addition / subtraction with rule-based rewards."""

    name = "arithmetic"

    def __init__(
        self,
        seed: int = 0,
        ops: Sequence[str] = ("+", "-"),
        max_digits: int = 2,
        min_digits: int = 1,
        allow_negative: bool = False,
        accuracy_weight: float = 1.0,
        format_weight: float = 0.2,
        strict: bool = True,
    ):
        super().__init__(seed)
        bad_ops = [op for op in ops if op not in ("+", "-")]
        if bad_ops:
            raise ValueError(f"Unsupported ops: {bad_ops}")
        if not 1 <= min_digits <= max_digits:
            raise ValueError("require 1 <= min_digits <= max_digits")
        self.ops = tuple(ops)
        self.max_digits = max_digits
        self.min_digits = min_digits
        self.allow_negative = allow_negative
        self.accuracy_weight = accuracy_weight
        self.format_weight = format_weight
        self.strict = strict

    @property
    def system_prompt(self) -> str:
        return "你是一个只会做算术的助手，必须严格按照给定格式回答。"

    def _operand(self) -> int:
        digits = self._rng.randint(self.min_digits, self.max_digits)
        low = 10 ** (digits - 1) if digits > 1 else 0
        return self._rng.randint(low, 10**digits - 1)

    def sample_task(self) -> Task:
        op = self._rng.choice(self.ops)
        a, b = self._operand(), self._operand()
        if op == "-" and not self.allow_negative and b > a:
            a, b = b, a
        answer = a + b if op == "+" else a - b
        return Task(
            question=f"{a} {op} {b}",
            answer=str(answer),
            meta={"a": a, "b": b, "op": op, "n_digits": max(len(str(a)), len(str(b)))},
        )

    def render(self, task: Task) -> str:
        return (
            f"计算 {task.question} 的结果。"
            f"先在 {THINK_OPEN}{THINK_CLOSE} 里写推理过程，"
            f"再在 {ANSWER_OPEN}{ANSWER_CLOSE} 里只写最终答案数字。"
        )

    def score(self, completion: str, task: Task) -> Reward:
        format_ok = has_valid_format(completion)
        predicted = parse_answer(completion, strict=self.strict)
        correct = predicted is not None and predicted == normalize_number(task.answer)
        accuracy = self.accuracy_weight if correct else 0.0
        fmt = self.format_weight if format_ok else 0.0
        return Reward(
            total=accuracy + fmt,
            accuracy=accuracy,
            format=fmt,
            format_ok=format_ok,
            correct=correct,
            parsed_answer=predicted,
        )

    def gold_completion(self, task: Task) -> str:
        return gold_completion(task.meta["a"], task.meta["op"], task.meta["b"])

    def corrupt_completion(self, task: Task, rng: random.Random) -> str:
        """A deliberately worse completion, used as the DPO ``rejected`` branch."""
        gold = self.gold_completion(task)
        answer = int(task.answer)
        mode = rng.choice(["off_by_one", "digit_slip", "no_tags", "format_only"])
        if mode == "off_by_one":
            wrong = answer + rng.choice([-1, 1])
            return re.sub(
                rf"{ANSWER_OPEN}.*?{ANSWER_CLOSE}",
                f"{ANSWER_OPEN}{wrong}{ANSWER_CLOSE}",
                gold,
            )
        if mode == "digit_slip":
            wrong = answer + rng.choice([-10, 10])
            return f"{THINK_OPEN}大概算一下。{THINK_CLOSE}{ANSWER_OPEN}{wrong}{ANSWER_CLOSE}"
        if mode == "no_tags":
            return f"答案是 {answer}。"
        return f"{THINK_OPEN}{THINK_CLOSE}{ANSWER_OPEN}{answer + rng.choice([-2, 2])}{ANSWER_CLOSE}"

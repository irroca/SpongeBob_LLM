"""Schema detection, validation and JSONL IO for every dataset the repo consumes.

Four record shapes exist across the training stages, and every data tool needs
to handle all of them without being told which is which:

====================  ================================================
``pretrain``          ``{"text": ...}``
``sft``               ``{"conversations": [{"role": ..., "content": ...}]}``
``preference``        ``{"prompt": ..., "chosen": ..., "rejected": ...}``
``task``              ``{"question": ..., "answer": ...}``  (RLVR eval sets)
====================  ================================================

Reading is streaming and fault-tolerant: a corrupt line in a 10GB dump should
be reported and skipped, not abort the job.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

PRETRAIN, SFT, PREFERENCE, TASK, UNKNOWN = "pretrain", "sft", "preference", "task", "unknown"
SCHEMAS = (PRETRAIN, SFT, PREFERENCE, TASK)

_WHITESPACE = re.compile(r"\s+")


@dataclass
class Record:
    """One parsed JSONL line plus where it came from."""

    data: dict
    line_no: int
    source: str = ""

    @property
    def schema(self) -> str:
        return detect_schema(self.data)

    @property
    def text(self) -> str:
        return record_text(self.data)


@dataclass
class ReadStats:
    """What happened while reading a file, so callers can report instead of guess."""

    total_lines: int = 0
    parsed: int = 0
    blank: int = 0
    malformed: int = 0
    malformed_examples: list[tuple[int, str]] = field(default_factory=list)


def detect_schema(record: dict) -> str:
    """Classify a record by the fields it carries."""
    if not isinstance(record, dict):
        return UNKNOWN
    if "conversations" in record:
        return SFT
    if "chosen" in record and "rejected" in record:
        return PREFERENCE
    if "text" in record:
        return PRETRAIN
    if "question" in record and "answer" in record:
        return TASK
    return UNKNOWN


def record_text(record: dict) -> str:
    """The canonical text of a record, used for length stats and deduplication.

    Conversations and preference pairs are flattened so that two records saying
    the same thing in the same shape produce the same string.
    """
    schema = detect_schema(record)
    if schema == PRETRAIN:
        return str(record.get("text", ""))
    if schema == SFT:
        turns = record.get("conversations") or []
        return "\n".join(
            f"{t.get('role', '')}: {t.get('content', '')}" for t in turns if isinstance(t, dict)
        )
    if schema == PREFERENCE:
        return "\n".join(
            str(record.get(key, "")) for key in ("prompt", "chosen", "rejected")
        )
    if schema == TASK:
        return f"{record.get('question', '')} => {record.get('answer', '')}"
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def prompt_text(record: dict) -> str:
    """The input side of a record, for contamination checks against eval sets.

    Deduplicating a preference pair or an eval question by its *full* text misses
    the case that matters: the same prompt appearing in both train and eval.
    """
    schema = detect_schema(record)
    if schema == SFT:
        turns = record.get("conversations") or []
        first_user = next(
            (t for t in turns if isinstance(t, dict) and t.get("role") == "user"), None
        )
        if first_user is None and turns:
            first_user = turns[0]
        return str((first_user or {}).get("content", ""))
    if schema == PREFERENCE:
        return str(record.get("prompt", ""))
    if schema == TASK:
        return str(record.get("question", ""))
    return record_text(record)


def normalize_text(text: str, casefold: bool = False) -> str:
    """NFKC + whitespace collapse, so trivial formatting differences are not new documents."""
    text = unicodedata.normalize("NFKC", text).strip()
    text = _WHITESPACE.sub(" ", text)
    return text.casefold() if casefold else text


def validate(record: dict) -> list[str]:
    """Problems that make a record unusable for training. Empty list means fine."""
    problems = []
    schema = detect_schema(record)
    if schema == UNKNOWN:
        return ["unknown schema: expected one of text / conversations / chosen+rejected / question+answer"]

    if schema == PRETRAIN:
        if not str(record.get("text", "")).strip():
            problems.append("empty text")
    elif schema == SFT:
        turns = record.get("conversations")
        if not isinstance(turns, list) or not turns:
            problems.append("conversations must be a non-empty list")
        else:
            roles = []
            for i, turn in enumerate(turns):
                if not isinstance(turn, dict) or "content" not in turn:
                    problems.append(f"turn {i} is not an object with a content field")
                    continue
                if not str(turn.get("content", "")).strip():
                    problems.append(f"turn {i} has empty content")
                roles.append(turn.get("role"))
            if roles and roles[0] not in (None, "user", "system"):
                problems.append(f"first turn role is {roles[0]!r}, expected user or system")
            if not any(r == "assistant" for r in roles):
                problems.append("no assistant turn, nothing would be trained on")
    elif schema == PREFERENCE:
        for key in ("prompt", "chosen", "rejected"):
            if not str(record.get(key, "")).strip():
                problems.append(f"empty {key}")
        if str(record.get("chosen", "")).strip() == str(record.get("rejected", "")).strip():
            problems.append("chosen and rejected are identical, the pair carries no preference")
    elif schema == TASK:
        for key in ("question", "answer"):
            if not str(record.get(key, "")).strip():
                problems.append(f"empty {key}")
    return problems


def read_jsonl(
    path: str,
    stats: Optional[ReadStats] = None,
    max_records: Optional[int] = None,
) -> Iterator[Record]:
    """Stream a JSONL file, skipping blank and malformed lines.

    A single bad line in a large dump should be reported, not fatal, so parse
    errors are counted in ``stats`` (with the first few examples) and skipped.
    """
    stats = stats if stats is not None else ReadStats()
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            stats.total_lines += 1
            line = line.strip()
            if not line:
                stats.blank += 1
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                stats.malformed += 1
                if len(stats.malformed_examples) < 5:
                    stats.malformed_examples.append((line_no, str(exc)))
                continue
            if not isinstance(data, dict):
                stats.malformed += 1
                if len(stats.malformed_examples) < 5:
                    stats.malformed_examples.append((line_no, f"expected object, got {type(data).__name__}"))
                continue
            stats.parsed += 1
            yield Record(data=data, line_no=line_no, source=path)
            if max_records is not None and stats.parsed >= max_records:
                return


def write_jsonl(path: str, records: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count

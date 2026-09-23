"""Quality filters for pretraining text, with per-rule rejection accounting.

Every rule returns the *name* of the reason a document was dropped rather than
a bare boolean, because the useful output of a filtering pass is not the kept
set — it is knowing which rule removed how much. A threshold that silently
deletes 80% of a corpus is a bug, and you only see it if rejections are
attributed.

Thresholds are deliberately not tuned here. Run ``datatools.stats`` on the real
corpus first, read the percentiles, then set them in the mixture spec. The
defaults below are loose enough to only remove obvious junk.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

_WORD_RE = re.compile(r"[A-Za-z]+")
_CJK_RANGE = ("\u4e00", "\u9fff")


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if _CJK_RANGE[0] <= ch <= _CJK_RANGE[1]) / len(text)


def latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    hits = sum(
        1
        for ch in text
        if ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN")
    )
    return hits / len(text)


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ch.isdigit()) / len(text)


def symbol_ratio(text: str) -> float:
    """Share of characters that are neither alphanumeric nor whitespace.

    Very high values mean tables, ASCII art, base64 blobs or minified assets.
    Note that code is legitimately symbol-heavy, so code sources need a much
    looser bound than prose.
    """
    if not text:
        return 0.0
    return sum(1 for ch in text if not (ch.isalnum() or ch.isspace())) / len(text)


def ngram_repetition(text: str, ngram: int = 10) -> float:
    """Share of character n-grams that are repeats within one document."""
    if len(text) <= ngram:
        return 0.0
    grams = [text[i : i + ngram] for i in range(len(text) - ngram + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def duplicate_line_ratio(text: str) -> float:
    """Share of non-blank lines that are exact duplicates of an earlier line.

    Catches navigation menus and boilerplate blocks that survive per-document
    dedup because each page's *combination* of junk is unique.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def mean_word_length(text: str) -> float:
    """Mean length of latin word runs; extreme values indicate tokenizer junk."""
    words = _WORD_RE.findall(text)
    return sum(len(w) for w in words) / len(words) if words else 0.0


@dataclass
class FilterConfig:
    """Thresholds for one source. ``None`` disables a rule."""

    min_chars: int = 1
    max_chars: Optional[int] = None
    min_cjk_ratio: Optional[float] = None
    min_latin_ratio: Optional[float] = None
    max_digit_ratio: Optional[float] = 0.5
    max_symbol_ratio: Optional[float] = None
    max_repetition: Optional[float] = 0.5
    max_duplicate_lines: Optional[float] = 0.5
    min_mean_word_length: Optional[float] = None
    max_mean_word_length: Optional[float] = None
    blocklist: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, values: Optional[dict]) -> "FilterConfig":
        values = dict(values or {})
        unknown = set(values) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"Unknown filter keys: {sorted(unknown)}")
        if "blocklist" in values:
            values["blocklist"] = tuple(values["blocklist"])
        return cls(**values)


def reject_reason(text: str, config: FilterConfig) -> Optional[str]:
    """Name of the first rule that rejects ``text``, or ``None`` if it passes.

    Rules are ordered cheapest-first so a huge junk document is discarded on
    its length before any per-character scan runs.
    """
    length = len(text)
    if length < config.min_chars:
        return "too_short"
    if config.max_chars is not None and length > config.max_chars:
        return "too_long"

    if config.blocklist:
        lowered = text.lower()
        if any(term.lower() in lowered for term in config.blocklist):
            return "blocklist"

    if config.min_cjk_ratio is not None and cjk_ratio(text) < config.min_cjk_ratio:
        return "low_cjk_ratio"
    if config.min_latin_ratio is not None and latin_ratio(text) < config.min_latin_ratio:
        return "low_latin_ratio"
    if config.max_digit_ratio is not None and digit_ratio(text) > config.max_digit_ratio:
        return "high_digit_ratio"
    if config.max_symbol_ratio is not None and symbol_ratio(text) > config.max_symbol_ratio:
        return "high_symbol_ratio"

    if config.max_duplicate_lines is not None and duplicate_line_ratio(text) > config.max_duplicate_lines:
        return "duplicate_lines"
    if config.max_repetition is not None and ngram_repetition(text) > config.max_repetition:
        return "repetitive"

    if config.min_mean_word_length is not None or config.max_mean_word_length is not None:
        mean_len = mean_word_length(text)
        if mean_len:  # only meaningful when the text has latin words at all
            if config.min_mean_word_length is not None and mean_len < config.min_mean_word_length:
                return "short_words"
            if config.max_mean_word_length is not None and mean_len > config.max_mean_word_length:
                return "long_words"
    return None


def filter_texts(
    texts: Iterable[str],
    config: FilterConfig,
) -> tuple[list[int], Counter]:
    """Partition by the filters. Returns (kept indices, rejection counts by reason)."""
    kept: list[int] = []
    rejected: Counter = Counter()
    for index, text in enumerate(texts):
        reason = reject_reason(text, config)
        if reason is None:
            kept.append(index)
        else:
            rejected[reason] += 1
    return kept, rejected


def describe_rejections(total: int, rejected: Mapping[str, int]) -> str:
    kept = total - sum(rejected.values())
    lines = [f"  kept {kept}/{total} ({kept / max(total, 1):.1%})"]
    for reason, count in Counter(rejected).most_common():
        lines.append(f"    -{count:>8} {reason} ({count / max(total, 1):.1%})")
    return "\n".join(lines)

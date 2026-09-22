"""Corpus statistics: the first thing to run on any new dataset.

Reports what determines whether a corpus is trainable at all — schema validity,
length distribution, language mix, duplication and repetitiveness — plus two
stage-specific checks that matter for this repo:

* **SFT**: turn counts and role structure.
* **Preference**: the chosen/rejected length gap. If ``chosen`` is
  systematically longer, DPO will partly learn "longer is better" rather than
  the intended preference, and that shows up here before training starts.

::

    python3 -m datatools.stats datasets/pretrain.jsonl
    python3 -m datatools.stats datasets/*.jsonl --tokenizer ./tokenizer/zh_6400 --json report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from typing import Optional, Sequence

from .records import (
    PREFERENCE,
    PRETRAIN,
    SFT,
    ReadStats,
    detect_schema,
    normalize_text,
    read_jsonl,
    record_text,
    validate,
)

PERCENTILES = (1, 10, 25, 50, 75, 90, 99)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile; avoids a numpy dependency in the reporting path."""
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round(q / 100.0 * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


def describe(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        **{f"p{q}": percentile(ordered, q) for q in PERCENTILES},
    }


def char_profile(text: str) -> dict:
    """Script mix of a document, used to spot language contamination and junk."""
    if not text:
        return {"cjk": 0.0, "latin": 0.0, "digit": 0.0, "other": 0.0}
    cjk = latin = digit = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            cjk += 1
        elif ch.isdigit():
            digit += 1
        elif ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN"):
            latin += 1
    total = len(text)
    return {
        "cjk": cjk / total,
        "latin": latin / total,
        "digit": digit / total,
        "other": (total - cjk - latin - digit) / total,
    }


def repetition_ratio(text: str, ngram: int = 10) -> float:
    """Share of character n-grams that are repeats within one document.

    High values flag boilerplate loops and degenerate scrapes — the documents
    that teach a model to repeat itself.
    """
    if len(text) <= ngram:
        return 0.0
    grams = [text[i : i + ngram] for i in range(len(text) - ngram + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _conversation_stats(records: Sequence[dict]) -> dict:
    turns = [len(r.get("conversations") or []) for r in records]
    role_sequences = Counter(
        tuple((t.get("role") or "?") for t in (r.get("conversations") or []) if isinstance(t, dict))
        for r in records
    )
    return {
        "turns": describe(turns),
        "most_common_role_sequences": [
            {"roles": list(seq), "count": n} for seq, n in role_sequences.most_common(5)
        ],
    }


def _preference_stats(records: Sequence[dict]) -> dict:
    chosen = [len(str(r.get("chosen", ""))) for r in records]
    rejected = [len(str(r.get("rejected", ""))) for r in records]
    longer = sum(1 for c, r in zip(chosen, rejected) if c > r)
    mean_chosen = sum(chosen) / len(chosen) if chosen else 0.0
    mean_rejected = sum(rejected) / len(rejected) if rejected else 0.0
    return {
        "chosen_chars": describe(chosen),
        "rejected_chars": describe(rejected),
        "chosen_longer_frac": longer / len(chosen) if chosen else 0.0,
        "mean_length_gap": mean_chosen - mean_rejected,
    }


def analyze(
    path: str,
    tokenizer=None,
    max_records: Optional[int] = None,
    casefold: bool = False,
) -> dict:
    read_stats = ReadStats()
    schemas: Counter = Counter()
    problems: Counter = Counter()
    seen_hashes: set[str] = set()
    records: list[dict] = []
    char_lengths: list[int] = []
    token_lengths: list[int] = []
    repetitions: list[float] = []
    profile_totals = {"cjk": 0.0, "latin": 0.0, "digit": 0.0, "other": 0.0}
    exact_duplicates = 0
    empty = 0

    for record in read_jsonl(path, read_stats, max_records=max_records):
        data = record.data
        records.append(data)
        schemas[detect_schema(data)] += 1
        for problem in validate(data):
            problems[problem.split(":")[0]] += 1

        text = record_text(data)
        if not text.strip():
            empty += 1
            continue

        char_lengths.append(len(text))
        repetitions.append(repetition_ratio(text))
        for key, value in char_profile(text).items():
            profile_totals[key] += value

        digest = hashlib.blake2b(
            normalize_text(text, casefold).encode("utf-8"), digest_size=16
        ).hexdigest()
        if digest in seen_hashes:
            exact_duplicates += 1
        else:
            seen_hashes.add(digest)

        if tokenizer is not None:
            token_lengths.append(len(tokenizer(text, add_special_tokens=False).input_ids))

    scored = max(len(char_lengths), 1)
    report = {
        "path": path,
        "lines": {
            "total": read_stats.total_lines,
            "parsed": read_stats.parsed,
            "blank": read_stats.blank,
            "malformed": read_stats.malformed,
            "malformed_examples": read_stats.malformed_examples,
        },
        "schemas": dict(schemas),
        "validation_problems": dict(problems),
        "empty_text": empty,
        "exact_duplicates": exact_duplicates,
        "exact_duplicate_frac": exact_duplicates / max(read_stats.parsed, 1),
        "chars": describe(char_lengths),
        "repetition_ratio": describe(repetitions),
        "script_mix": {k: v / scored for k, v in profile_totals.items()},
    }
    if tokenizer is not None:
        report["tokens"] = describe(token_lengths)

    dominant = schemas.most_common(1)[0][0] if schemas else None
    if dominant == SFT:
        report["sft"] = _conversation_stats(records)
    elif dominant == PREFERENCE:
        report["preference"] = _preference_stats(records)
    return report


def _fmt(stats: dict) -> str:
    if not stats.get("count"):
        return "n/a"
    return (
        f"mean={stats['mean']:.1f} p10={stats['p10']:.0f} p50={stats['p50']:.0f} "
        f"p90={stats['p90']:.0f} p99={stats['p99']:.0f} max={stats['max']:.0f}"
    )


def render(report: dict) -> str:
    lines = [f"=== {report['path']}"]
    counts = report["lines"]
    lines.append(
        f"  lines: {counts['total']} total, {counts['parsed']} parsed, "
        f"{counts['blank']} blank, {counts['malformed']} malformed"
    )
    for line_no, reason in counts["malformed_examples"]:
        lines.append(f"    line {line_no}: {reason}")
    lines.append(f"  schemas: {report['schemas']}")
    if report["validation_problems"]:
        lines.append(f"  validation problems: {report['validation_problems']}")
    lines.append(
        f"  duplicates: {report['exact_duplicates']} exact "
        f"({report['exact_duplicate_frac']:.2%}), {report['empty_text']} empty"
    )
    lines.append(f"  chars: {_fmt(report['chars'])}")
    if "tokens" in report:
        lines.append(f"  tokens: {_fmt(report['tokens'])}")
    rep = report["repetition_ratio"]
    if rep.get("count"):
        lines.append(f"  repetition: mean={rep['mean']:.3f} p90={rep['p90']:.3f} max={rep['max']:.3f}")
    mix = report["script_mix"]
    lines.append(
        f"  script mix: cjk={mix['cjk']:.2f} latin={mix['latin']:.2f} "
        f"digit={mix['digit']:.2f} other={mix['other']:.2f}"
    )
    if "sft" in report:
        lines.append(f"  turns: {_fmt(report['sft']['turns'])}")
        for entry in report["sft"]["most_common_role_sequences"]:
            lines.append(f"    {entry['count']:>7} x {entry['roles']}")
    if "preference" in report:
        pref = report["preference"]
        lines.append(
            f"  chosen vs rejected: chosen longer in {pref['chosen_longer_frac']:.1%} of pairs, "
            f"mean gap {pref['mean_length_gap']:+.1f} chars"
        )
        if pref["chosen_longer_frac"] > 0.7:
            lines.append(
                "    warning: chosen is longer in most pairs; DPO will partly learn "
                "'longer is better' rather than the intended preference"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report statistics for JSONL datasets")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to a tokenizer for token-length stats")
    parser.add_argument("--max_records", type=int, default=None, help="Sample only the first N records")
    parser.add_argument("--casefold", action="store_true", help="Case-insensitive exact-duplicate matching")
    parser.add_argument("--json", type=str, default=None, help="Write the full report to this path")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    reports = []
    for path in args.paths:
        report = analyze(path, tokenizer, args.max_records, args.casefold)
        reports.append(report)
        print(render(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(reports if len(reports) > 1 else reports[0], fh, ensure_ascii=False, indent=2)
        print(f"\nreport -> {args.json}")


if __name__ == "__main__":
    main()

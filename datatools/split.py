"""Deterministic train / validation / holdout splits.

Splitting by a hash of the document's own content rather than by position or by
a shuffled index buys two properties that matter for ablations:

* **Reproducible.** Re-running the pipeline puts every document in the same
  split, so two ablation runs differ only in the variable under test.
* **Stable under corpus growth.** Adding documents does not reshuffle the
  existing ones, so a validation curve stays comparable across data versions.

The holdout split exists to be *never* trained on at any stage — it is what the
final numbers get reported against, and what decontamination protects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .records import ReadStats, normalize_text, read_jsonl, record_text, write_jsonl

TRAIN, VAL, HOLDOUT = "train", "val", "holdout"


def bucket(text: str, salt: str = "") -> float:
    """Map a document to a stable value in [0, 1) from its content."""
    digest = hashlib.blake2b(
        f"{salt}\x00{normalize_text(text)}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


@dataclass
class SplitReport:
    counts: dict[str, int] = field(default_factory=dict)
    val_fraction: float = 0.0
    holdout_fraction: float = 0.0
    salt: str = ""

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> dict:
        return {
            "counts": self.counts,
            "total": self.total,
            "val_fraction": self.val_fraction,
            "holdout_fraction": self.holdout_fraction,
            "salt": self.salt,
        }


def assign_split(
    text: str,
    val_fraction: float,
    holdout_fraction: float,
    salt: str = "",
) -> str:
    value = bucket(text, salt)
    if value < holdout_fraction:
        return HOLDOUT
    if value < holdout_fraction + val_fraction:
        return VAL
    return TRAIN


def split_records(
    records: Sequence[dict],
    val_fraction: float = 0.005,
    holdout_fraction: float = 0.005,
    salt: str = "",
) -> tuple[dict[str, list[dict]], SplitReport]:
    if not 0 <= val_fraction < 1 or not 0 <= holdout_fraction < 1:
        raise ValueError("fractions must be in [0, 1)")
    if val_fraction + holdout_fraction >= 1:
        raise ValueError("val_fraction + holdout_fraction must be < 1")

    splits: dict[str, list[dict]] = {TRAIN: [], VAL: [], HOLDOUT: []}
    for record in records:
        name = assign_split(record_text(record), val_fraction, holdout_fraction, salt)
        splits[name].append(record)

    report = SplitReport(
        counts={name: len(rows) for name, rows in splits.items()},
        val_fraction=val_fraction,
        holdout_fraction=holdout_fraction,
        salt=salt,
    )
    return splits, report


def render(report: SplitReport) -> str:
    lines = ["=== split"]
    for name in (TRAIN, VAL, HOLDOUT):
        count = report.counts.get(name, 0)
        lines.append(f"  {name:<8} {count:>9} ({count / max(report.total, 1):.2%})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a JSONL dataset deterministically")
    parser.add_argument("path")
    parser.add_argument("--out_prefix", type=str, required=True, help="Writes <prefix>_{train,val,holdout}.jsonl")
    parser.add_argument("--val_fraction", type=float, default=0.005)
    parser.add_argument("--holdout_fraction", type=float, default=0.005)
    parser.add_argument("--salt", type=str, default="", help="Change to re-draw the split")
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    read_stats = ReadStats()
    records = [r.data for r in read_jsonl(args.path, read_stats)]
    splits, report = split_records(
        records, args.val_fraction, args.holdout_fraction, args.salt
    )
    for name, rows in splits.items():
        path = f"{args.out_prefix}_{name}.jsonl"
        write_jsonl(path, rows)
        print(f"  wrote {len(rows):>9} -> {path}")
    print(render(report))

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

"""Deduplicate a JSONL corpus: exact first, then MinHash/LSH near-duplicates.

Exact duplicates are removed by hashing normalized text, which is cheap and
catches copy-paste. Near-duplicates need MinHash: web text is full of documents
that differ only in a header or a date, and they inflate a corpus without
adding information — worse, they get memorized.

Optionally also drops training records whose *prompt* is exactly equal (after
normalization) to one in an evaluation set (``--against``). That is the cheap
check; for real contamination use ``datatools.decontaminate``, which matches
13-grams and therefore also catches a benchmark question embedded in a longer
web page. Use this flag only when both sides are known to be per-prompt
records, e.g. two generated task sets.

**Memory bound:** MinHash keeps a signature per surviving document (~1KB at 128
permutations), so this holds roughly 1–2M documents. For a multi-billion-token
corpus, run it per source file rather than over the whole mixture — which is
why ``datatools.prepare`` does only streaming exact dedup inline.

::

    python3 -m datatools.dedup datasets/raw.jsonl --out datasets/clean.jsonl
    python3 -m datatools.dedup datasets/sft.jsonl --out datasets/sft_clean.jsonl \\
        --threshold 0.85 --against datasets/arith_eval.jsonl --report dedup.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .minhash import MinHasher, choose_bands, cluster_near_duplicates, lsh_threshold
from .records import ReadStats, normalize_text, prompt_text, read_jsonl, record_text, write_jsonl


@dataclass
class DedupReport:
    input_records: int = 0
    exact_removed: int = 0
    near_removed: int = 0
    contaminated_removed: int = 0
    kept: int = 0
    near_clusters: int = 0
    largest_cluster: int = 0
    threshold: float = 0.0
    bands: int = 0
    rows: int = 0
    effective_threshold: float = 0.0
    examples: list[dict] = field(default_factory=list)

    @property
    def retention(self) -> float:
        return self.kept / max(self.input_records, 1)

    def to_dict(self) -> dict:
        return {**self.__dict__, "retention": self.retention}


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def exact_duplicate_indices(texts: Sequence[str], casefold: bool = False) -> set[int]:
    """Indices to drop, keeping the first occurrence of each normalized text."""
    seen: set[str] = set()
    drop: set[int] = set()
    for index, text in enumerate(texts):
        digest = _digest(normalize_text(text, casefold))
        if digest in seen:
            drop.add(index)
        else:
            seen.add(digest)
    return drop


def contaminated_indices(
    records: Sequence[dict],
    holdout_prompts: set[str],
    casefold: bool = False,
) -> set[int]:
    """Training indices whose prompt also appears in a held-out set."""
    return {
        index
        for index, record in enumerate(records)
        if normalize_text(prompt_text(record), casefold) in holdout_prompts
    }


def load_prompt_set(paths: Sequence[str], casefold: bool = False) -> set[str]:
    prompts: set[str] = set()
    for path in paths:
        for record in read_jsonl(path):
            text = normalize_text(prompt_text(record.data), casefold)
            if text:
                prompts.add(text)
    return prompts


def dedup_records(
    records: Sequence[dict],
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram: int = 5,
    seed: int = 0,
    casefold: bool = False,
    holdout_prompts: Optional[set[str]] = None,
    near: bool = True,
    bands: Optional[int] = None,
    rows: Optional[int] = None,
) -> tuple[list[dict], DedupReport]:
    """Drop exact duplicates, then near-duplicates, then contaminated records.

    Order matters: removing exact duplicates first shrinks the set MinHash has
    to signature, and contamination is checked last so the report attributes
    each removal to exactly one reason.
    """
    report = DedupReport(input_records=len(records), threshold=threshold)
    texts = [record_text(r) for r in records]

    drop = exact_duplicate_indices(texts, casefold)
    report.exact_removed = len(drop)
    survivors = [i for i in range(len(records)) if i not in drop]

    if near and len(survivors) > 1:
        hasher = MinHasher(num_perm=num_perm, ngram=ngram, seed=seed)
        signatures = hasher.signatures([texts[i] for i in survivors])
        if bands is None or rows is None:
            bands, rows = choose_bands(num_perm, threshold)
        report.bands, report.rows = bands, rows
        report.effective_threshold = lsh_threshold(bands, rows)
        clusters = cluster_near_duplicates(signatures, threshold, bands, rows)
        report.near_clusters = len(clusters)
        report.largest_cluster = max((len(c) for c in clusters), default=0)
        for cluster in clusters:
            # Keep the earliest member so output order matches input order.
            keeper = survivors[cluster[0]]
            for member in cluster[1:]:
                drop.add(survivors[member])
                if len(report.examples) < 5:
                    report.examples.append(
                        {
                            "kept_index": keeper,
                            "dropped_index": survivors[member],
                            "kept": texts[keeper][:120],
                            "dropped": texts[survivors[member]][:120],
                        }
                    )
        report.near_removed = len(drop) - report.exact_removed

    if holdout_prompts:
        remaining = [i for i in range(len(records)) if i not in drop]
        hits = contaminated_indices([records[i] for i in remaining], holdout_prompts, casefold)
        contaminated = {remaining[i] for i in hits}
        report.contaminated_removed = len(contaminated)
        drop |= contaminated

    kept = [records[i] for i in range(len(records)) if i not in drop]
    report.kept = len(kept)
    return kept, report


def render(report: DedupReport, read_stats: Optional[ReadStats] = None) -> str:
    lines = ["=== dedup"]
    if read_stats is not None and read_stats.malformed:
        lines.append(f"  skipped {read_stats.malformed} malformed line(s)")
    lines.append(f"  input: {report.input_records} records")
    lines.append(f"  exact duplicates removed: {report.exact_removed}")
    lines.append(
        f"  near duplicates removed: {report.near_removed} "
        f"({report.near_clusters} clusters, largest {report.largest_cluster})"
    )
    if report.bands:
        lines.append(
            f"  LSH: {report.bands} bands x {report.rows} rows, "
            f"target jaccard {report.threshold:.2f}, S-curve midpoint {report.effective_threshold:.2f}"
        )
    if report.contaminated_removed:
        lines.append(f"  contaminated with holdout prompts: {report.contaminated_removed}")
    lines.append(f"  kept: {report.kept} ({report.retention:.1%} retention)")
    for example in report.examples[:3]:
        lines.append(f"    kept    [{example['kept_index']}]: {example['kept']!r}")
        lines.append(f"    dropped [{example['dropped_index']}]: {example['dropped']!r}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate a JSONL dataset")
    parser.add_argument("path")
    parser.add_argument("--out", type=str, required=True, help="Where to write the kept records")
    parser.add_argument("--threshold", type=float, default=0.8, help="Jaccard threshold for near-duplicates")
    parser.add_argument("--num_perm", type=int, default=128, help="MinHash signature length")
    parser.add_argument("--ngram", type=int, default=5, help="Character shingle size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--casefold", action="store_true", help="Case-insensitive matching")
    parser.add_argument("--exact_only", action="store_true", help="Skip the MinHash pass")
    parser.add_argument(
        "--bands",
        type=int,
        default=None,
        help="Override the LSH banding (bands*rows must equal --num_perm). More bands = "
             "higher recall and more candidate pairs to verify",
    )
    parser.add_argument("--rows", type=int, default=None, help="Rows per LSH band")
    parser.add_argument(
        "--against",
        nargs="*",
        default=[],
        help="Eval/holdout JSONL files; training records sharing a prompt are dropped",
    )
    parser.add_argument("--report", type=str, default=None, help="Write the report as JSON")
    args = parser.parse_args()

    read_stats = ReadStats()
    records = [r.data for r in read_jsonl(args.path, read_stats)]
    holdout = load_prompt_set(args.against, args.casefold) if args.against else None

    kept, report = dedup_records(
        records,
        threshold=args.threshold,
        num_perm=args.num_perm,
        ngram=args.ngram,
        seed=args.seed,
        casefold=args.casefold,
        holdout_prompts=holdout,
        near=not args.exact_only,
        bands=args.bands,
        rows=args.rows,
    )
    write_jsonl(args.out, kept)
    print(render(report, read_stats))
    print(f"  wrote -> {args.out}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
        print(f"  report -> {args.report}")


if __name__ == "__main__":
    main()

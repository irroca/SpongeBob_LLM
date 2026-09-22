"""Build a training corpus from a mixture spec.

Mixture weights are in **tokens**, but corpora are published in documents and
bytes, so the budget can only be enforced by tokenizing as we go. The pipeline
is therefore streaming end to end: each source is pulled lazily, filtered,
token-counted, and written out the moment its share of the budget is met. At
10B tokens the corpus is ~30GB of text and nothing can be held in memory.

Stages, in this order:

1. **Pull** — HuggingFace ``streaming=True``, or a local JSONL (used by tests).
2. **Filter** — per-source thresholds from ``datatools.filters``; every
   rejection is attributed to the rule that caused it.
3. **Exact dedup** — a running digest set, which is the only dedup that scales
   in a single streaming pass (see the note on near-dedup below).
4. **Decontaminate** — 13-gram overlap against the eval sets.
5. **Split** — deterministic train/val/holdout by content hash.
6. **Manifest** — actual tokens and documents per source, rejection counts per
   rule, and the seeds, so two ablation runs can be told apart.

**Near-duplicate dedup is not part of this pass.** MinHash needs a signature
per surviving document (~1KB at 128 permutations), so it is bounded to roughly
1–2M documents in memory. Run ``datatools.dedup`` on a single source file when
it fits. In practice the yield is low here: FineWeb-Edu and FineWeb2-HQ are
already MinHash-deduplicated upstream, so the remaining near-duplicates come
from cross-source overlap and from our own synthetic data.

::

    python3 -m datatools.prepare configs/mixture_v1.json --dry_run
    python3 -m datatools.prepare configs/mixture_v1.json --out_dir datasets/prepared
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

from .decontaminate import DEFAULT_N, EvalIndex, load_eval_index
from .filters import FilterConfig, reject_reason
from .records import detect_schema, normalize_text, read_jsonl, record_text, write_jsonl
from .split import HOLDOUT, TRAIN, VAL, assign_split

TOKENIZE_BATCH = 256


@dataclass
class SourceSpec:
    """One component of the mixture."""

    name: str
    weight: float
    text_field: str = "text"
    jsonl: Optional[str] = None
    hf: Optional[dict] = None
    filters: FilterConfig = field(default_factory=FilterConfig)
    max_records: Optional[int] = None
    note: str = ""

    @classmethod
    def from_dict(cls, values: dict) -> "SourceSpec":
        values = dict(values)
        if "filters" in values:
            values["filters"] = FilterConfig.from_dict(values["filters"])
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown source keys: {sorted(unknown)}")
        source = cls(**values)
        if (source.jsonl is None) == (source.hf is None):
            raise ValueError(f"source {source.name!r} needs exactly one of 'jsonl' or 'hf'")
        if source.weight <= 0:
            raise ValueError(f"source {source.name!r} must have weight > 0")
        return source


@dataclass
class MixtureSpec:
    name: str
    total_tokens: int
    sources: list[SourceSpec]
    tokenizer: str = "./tokenizer/zh_6400"
    seed: int = 0
    val_fraction: float = 0.005
    holdout_fraction: float = 0.005
    decontaminate: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict) -> "MixtureSpec":
        values = dict(values)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown mixture keys: {sorted(unknown)}")
        values["sources"] = [SourceSpec.from_dict(s) for s in values.get("sources", [])]
        spec = cls(**values)
        if not spec.sources:
            raise ValueError("mixture needs at least one source")
        total_weight = sum(s.weight for s in spec.sources)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"source weights must sum to 1.0, got {total_weight:.6f}")
        return spec

    @classmethod
    def load(cls, path: str) -> "MixtureSpec":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def token_budget(self, source: SourceSpec) -> int:
        return int(round(self.total_tokens * source.weight))


@dataclass
class SourceReport:
    name: str
    target_tokens: int = 0
    tokens: int = 0
    documents: int = 0
    seen: int = 0
    rejected: Counter = field(default_factory=Counter)
    exact_duplicates: int = 0
    exhausted: bool = False

    @property
    def fill(self) -> float:
        return self.tokens / max(self.target_tokens, 1)

    @property
    def keep_rate(self) -> float:
        """Share of records read that survived filtering and exact dedup."""
        return self.documents / max(self.seen, 1)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target_tokens": self.target_tokens,
            "tokens": self.tokens,
            "documents": self.documents,
            "records_read": self.seen,
            "fill": self.fill,
            "keep_rate": self.keep_rate,
            "tokens_per_doc": self.tokens / max(self.documents, 1),
            "rejected": dict(self.rejected),
            "exact_duplicates": self.exact_duplicates,
            "exhausted": self.exhausted,
        }


def iter_raw(source: SourceSpec) -> Iterator[dict]:
    """Yield raw records from a source, lazily."""
    if source.jsonl is not None:
        for record in read_jsonl(source.jsonl):
            yield record.data
        return

    from datasets import load_dataset  # imported lazily: tests run offline

    options = dict(source.hf or {})
    options.setdefault("streaming", True)
    options.setdefault("split", "train")
    path = options.pop("path")
    dataset = load_dataset(path, **options)
    for row in dataset:
        yield dict(row)


def to_record(raw: dict, source: SourceSpec) -> Optional[dict]:
    """Normalize a raw row into one of the repo's four schemas.

    Rows that already match a known schema pass through unchanged, so a
    preference or conversation dataset can be mixed in without special-casing.
    """
    if detect_schema(raw) != "unknown":
        return raw
    value = raw.get(source.text_field)
    if value is None:
        return None
    return {"text": str(value)}


def _batched(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def count_tokens(tokenizer, texts: Sequence[str]) -> list[int]:
    """Token counts for a batch. Batched because this is the pipeline's hot path."""
    if not texts:
        return []
    encoded = tokenizer(list(texts), add_special_tokens=False)
    return [len(ids) for ids in encoded["input_ids"]]


def prepare_source(
    spec: MixtureSpec,
    source: SourceSpec,
    tokenizer,
    out_path: Optional[str],
    progress_every: int = 0,
) -> SourceReport:
    """Stream one source until its token budget is met, writing filtered records."""
    report = SourceReport(name=source.name, target_tokens=spec.token_budget(source))
    seen_digests: set[bytes] = set()
    buffer: list[tuple[dict, str]] = []
    handle = None
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        handle = open(out_path, "w", encoding="utf-8")

    def flush() -> bool:
        """Tokenize and emit the buffer. Returns True when the budget is met.

        Records pulled into the buffer but never emitted (because the budget
        filled mid-batch) are subtracted back out of ``seen``, so the reported
        keep rate stays a true filter pass rate rather than being diluted by
        the tokenization batch size.
        """
        nonlocal buffer
        if not buffer:
            return False
        counts = count_tokens(tokenizer, [text for _, text in buffer])
        for position, ((record, _), n_tokens) in enumerate(zip(buffer, counts)):
            if handle is not None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            report.tokens += n_tokens
            report.documents += 1
            if report.tokens >= report.target_tokens:
                report.seen -= len(buffer) - position - 1
                buffer = []
                return True
        buffer = []
        return False

    try:
        for raw in iter_raw(source):
            report.seen += 1
            if source.max_records is not None and report.seen > source.max_records:
                break

            record = to_record(raw, source)
            if record is None:
                report.rejected["missing_text_field"] += 1
                continue

            text = record_text(record)
            reason = reject_reason(text, source.filters)
            if reason is not None:
                report.rejected[reason] += 1
                continue

            digest = hashlib.blake2b(
                normalize_text(text).encode("utf-8"), digest_size=16
            ).digest()
            if digest in seen_digests:
                report.exact_duplicates += 1
                continue
            seen_digests.add(digest)

            buffer.append((record, text))
            if len(buffer) >= TOKENIZE_BATCH and flush():
                return report
            if progress_every and report.documents and report.documents % progress_every == 0:
                print(
                    f"    {source.name}: {report.documents} docs, "
                    f"{report.tokens / 1e6:.1f}M/{report.target_tokens / 1e6:.1f}M tokens",
                    flush=True,
                )
        else:
            report.exhausted = True
        flush()
    finally:
        if handle is not None:
            handle.close()
    return report


def finalize(
    spec: MixtureSpec,
    source_paths: Sequence[str],
    out_dir: str,
    eval_index: Optional[EvalIndex],
    lcs_threshold: Optional[float] = None,
) -> dict:
    """Single streaming pass: decontaminate, assign splits, write the three files."""
    from .decontaminate import lcs_ratio, match_record

    handles = {
        name: open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8")
        for name in (TRAIN, VAL, HOLDOUT)
    }
    counts = Counter()
    contaminated = 0
    rescued = 0
    try:
        for path in source_paths:
            for record in read_jsonl(path):
                if eval_index is not None:
                    match = match_record(record.data, eval_index)
                    if match is not None:
                        part, hit = match
                        if lcs_threshold is not None and lcs_ratio(part, eval_index.items[hit]) < lcs_threshold:
                            rescued += 1
                        else:
                            contaminated += 1
                            continue
                name = assign_split(
                    record_text(record.data),
                    spec.val_fraction,
                    spec.holdout_fraction,
                    salt=str(spec.seed),
                )
                handles[name].write(json.dumps(record.data, ensure_ascii=False) + "\n")
                counts[name] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "counts": dict(counts),
        "contaminated_removed": contaminated,
        "lcs_rescued": rescued,
        "eval_items": len(eval_index) if eval_index is not None else 0,
    }


def render_sources(reports: Sequence[SourceReport]) -> str:
    header = (
        f"{'source':<16} {'target':>12} {'tokens':>12} {'fill':>6} "
        f"{'docs':>9} {'read':>9} {'kept%':>6} {'tok/doc':>8}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        lines.append(
            f"{report.name:<16} {report.target_tokens:>12} {report.tokens:>12} "
            f"{report.fill:>5.0%} {report.documents:>9} {report.seen:>9} "
            f"{report.keep_rate:>5.0%} {report.tokens / max(report.documents, 1):>8.0f}"
        )
        if report.exhausted and report.fill < 0.99:
            lines.append(
                f"  ! {report.name} ran out of data at {report.fill:.0%} of its budget"
            )
        for reason, count in report.rejected.most_common(4):
            lines.append(f"    -{count:>8} {reason}")
        if report.exact_duplicates:
            lines.append(f"    -{report.exact_duplicates:>8} exact_duplicate")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a corpus from a mixture spec")
    parser.add_argument("spec", help="Mixture spec JSON")
    parser.add_argument("--out_dir", type=str, default="datasets/prepared")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report what each source would contribute without writing the corpus",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply total_tokens, e.g. 0.01 for a quick pipeline check",
    )
    parser.add_argument("--progress_every", type=int, default=0, help="Docs between progress lines")
    args = parser.parse_args()

    spec = MixtureSpec.load(args.spec)
    if args.scale != 1.0:
        spec.total_tokens = int(spec.total_tokens * args.scale)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer)
    budget = (
        f"{spec.total_tokens / 1e9:.3f}B" if spec.total_tokens >= 1e8
        else f"{spec.total_tokens:,}"
    )
    print(f"mixture {spec.name!r}: {budget} tokens, tokenizer {spec.tokenizer}")

    os.makedirs(args.out_dir, exist_ok=True)
    source_dir = os.path.join(args.out_dir, "sources")
    reports, paths = [], []
    for source in spec.sources:
        out_path = None if args.dry_run else os.path.join(source_dir, f"{source.name}.jsonl")
        reports.append(prepare_source(spec, source, tokenizer, out_path, args.progress_every))
        if out_path:
            paths.append(out_path)

    print(render_sources(reports))

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    decon = spec.decontaminate or {}
    eval_index = None
    if decon.get("against"):
        eval_index = load_eval_index(
            decon["against"], decon.get("n", DEFAULT_N), decon.get("field", "auto")
        )
        print(f"\nindexed {len(eval_index)} eval items for decontamination")
    split_info = finalize(
        spec, paths, args.out_dir, eval_index, decon.get("lcs_threshold")
    )
    print(f"splits: {split_info['counts']}, contaminated removed: {split_info['contaminated_removed']}")

    manifest = {
        "mixture": spec.name,
        "tokenizer": spec.tokenizer,
        "total_tokens_requested": spec.total_tokens,
        "total_tokens_collected": sum(r.tokens for r in reports),
        "seed": spec.seed,
        "val_fraction": spec.val_fraction,
        "holdout_fraction": spec.holdout_fraction,
        "sources": [r.to_dict() for r in reports],
        "split": split_info,
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()

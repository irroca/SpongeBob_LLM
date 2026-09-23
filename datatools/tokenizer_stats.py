"""Measure how well a tokenizer compresses a corpus, and how many tokens it yields.

Two questions this answers, both needed before a mixture can be built:

1. **How many tokens is this pile of text?** Mixture ratios are expressed in
   tokens, but corpora are published in documents, words or bytes. The count
   depends on the tokenizer, so it has to be measured, not assumed.
2. **Is this tokenizer adequate for this data?** A vocabulary trained on Chinese
   shreds source code: measured on this repo's 6400-token vocabulary,
   ``grpo_advantages`` becomes ``gr|p|o|_|ad|v|ant|ages``. ``chars_per_token``
   and ``single_char_frac`` quantify that, which is what makes a vocab-size
   ablation decidable instead of a guess.

::

    python3 -m datatools.tokenizer_stats --probe
    python3 -m datatools.tokenizer_stats datasets/zh.jsonl datasets/code.jsonl
    python3 -m datatools.tokenizer_stats datasets/zh.jsonl --tokenizer ./tok_16k ./tok_32k
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Optional, Sequence

from .records import read_jsonl, record_text

# Short probes covering the domains this project mixes. Useful for a quick read
# on a candidate tokenizer before any corpus has been downloaded.
PROBES: dict[str, str] = {
    "zh": "深度学习模型的训练过程中，数据质量往往比模型结构更为重要，这一点在小模型上尤其明显。",
    "en": "The quality of the pretraining corpus matters more than the architecture, "
          "especially for small models trained on a limited token budget.",
    "code": (
        "def grpo_advantages(rewards, normalize_std=True, eps=1e-6):\n"
        "    centered = rewards - rewards.mean(dim=-1, keepdim=True)\n"
        "    if not normalize_std:\n"
        "        return centered\n"
        "    return centered / (rewards.std(dim=-1, unbiased=False) + eps)\n"
    ),
    "math": "Let f(x) = 3x^2 + 5x - 2. Then f'(x) = 6x + 5, so f'(2) = 17.",
    "cot": "<think>把 25 拆成 20 和 5：37 + 20 = 57；57 + 5 = 62。</think><answer>62</answer>",
}


@dataclass
class Fertility:
    """Compression statistics for one tokenizer over one body of text."""

    documents: int
    chars: int
    tokens: int
    single_char_tokens: int
    unk_tokens: int

    @property
    def chars_per_token(self) -> float:
        """Higher is better. Roughly 4+ for English, 1.5+ for Chinese, 3+ for code."""
        return self.chars / max(self.tokens, 1)

    @property
    def tokens_per_doc(self) -> float:
        return self.tokens / max(self.documents, 1)

    @property
    def single_char_frac(self) -> float:
        """Share of tokens that decode to a single character.

        Only comparable *within* a script. Chinese characters are meaningful
        units, so a Chinese vocabulary legitimately sits above 50% here; reading
        that as fragmentation is a mistake. To judge fragmentation, compare
        ``chars_per_token`` between two same-script domains — English prose vs
        source code under one tokenizer is the useful pair.
        """
        return self.single_char_tokens / max(self.tokens, 1)

    @property
    def unk_frac(self) -> float:
        return self.unk_tokens / max(self.tokens, 1)

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "chars": self.chars,
            "tokens": self.tokens,
            "chars_per_token": self.chars_per_token,
            "tokens_per_doc": self.tokens_per_doc,
            "single_char_frac": self.single_char_frac,
            "unk_frac": self.unk_frac,
        }


def measure(tokenizer, texts: Sequence[str]) -> Fertility:
    unk_id = getattr(tokenizer, "unk_token_id", None)
    documents = chars = tokens = single = unk = 0
    for text in texts:
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        documents += 1
        chars += len(text)
        tokens += len(ids)
        if unk_id is not None:
            unk += sum(1 for i in ids if i == unk_id)
        single += sum(1 for i in ids if len(tokenizer.decode([i])) == 1)
    return Fertility(documents, chars, tokens, single, unk)


def measure_corpus(
    tokenizer,
    path: str,
    max_records: Optional[int] = None,
) -> Fertility:
    texts = [record_text(r.data) for r in read_jsonl(path, max_records=max_records)]
    return measure(tokenizer, texts)


def project_tokens(sample: Fertility, total_documents: int) -> float:
    """Extrapolate a full corpus's token count from a sampled subset.

    Mixture weights are in tokens, so a corpus published as "188M documents"
    has to be converted before it can be budgeted.
    """
    return sample.tokens_per_doc * total_documents


def render(rows: Sequence[tuple[str, str, Fertility]]) -> str:
    header = (
        f"{'tokenizer':<20} {'source':<22} {'docs':>7} {'tokens':>10} "
        f"{'chars/tok':>10} {'1-char':>7} {'unk':>6}"
    )
    lines = [header, "-" * len(header)]
    for tokenizer_name, source, stats in rows:
        lines.append(
            f"{tokenizer_name[-20:]:<20} {source[-22:]:<22} {stats.documents:>7} "
            f"{stats.tokens:>10} {stats.chars_per_token:>10.2f} "
            f"{stats.single_char_frac:>6.1%} {stats.unk_frac:>5.1%}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure tokenizer compression on a corpus")
    parser.add_argument("paths", nargs="*", help="JSONL files to measure")
    parser.add_argument(
        "--tokenizer",
        nargs="+",
        default=["./tokenizer/zh_6400"],
        help="One or more tokenizers to compare",
    )
    parser.add_argument("--probe", action="store_true", help="Also measure the built-in domain probes")
    parser.add_argument("--max_records", type=int, default=2000, help="Documents sampled per file")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    if not args.paths and not args.probe:
        parser.error("pass at least one JSONL path, or --probe")

    from transformers import AutoTokenizer

    rows: list[tuple[str, str, Fertility]] = []
    for name in args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(name)
        if args.probe:
            for domain, text in PROBES.items():
                rows.append((name, f"probe:{domain}", measure(tokenizer, [text])))
        for path in args.paths:
            rows.append((name, path, measure_corpus(tokenizer, path, args.max_records)))

    print(render(rows))

    if args.json:
        payload = [
            {"tokenizer": name, "source": source, **stats.to_dict()}
            for name, source, stats in rows
        ]
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nreport -> {args.json}")


if __name__ == "__main__":
    main()

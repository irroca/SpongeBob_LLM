"""Fetch benchmark sets from HuggingFace and convert them to the repo's JSONL.

Two uses, and they pull in opposite directions:

* **Decontamination.** ``datatools.prepare`` needs these files in its spec's
  ``decontaminate.against`` list, otherwise the check runs against nothing and
  every eval number afterwards is unverifiable. For this purpose more surface
  is better, so worked solutions are kept in a ``solution`` field —
  ``records.record_parts`` indexes it, because a page quoting the solution but
  not the question is still contamination.
* **Evaluation and RL prompts.** ``envs.base.load_tasks`` reads the same
  ``{"question", "answer"}`` schema, so a fetched set can be handed straight to
  ``grpo.py --eval_path``. Here the answer has to be a single verifiable
  string, so multiple-choice letters are resolved to their option text.

Each converter is written against field names verified on the HuggingFace API,
and is a pure function so the tests can exercise it offline on a recorded row.

::

    python3 -m datatools.fetch_evals --all --out_dir datasets/eval
    python3 -m datatools.fetch_evals --sets gsm8k math500 --update_spec configs/mixture_v1.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .records import write_jsonl

_GSM8K_FINAL = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


def _as_list(value: Any) -> list:
    """Some HF rows carry list fields as their Python ``repr``; accept both."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    return []


def convert_gsm8k(row: dict) -> Optional[dict]:
    """``answer`` is a chain of thought ending in ``#### <final>``."""
    answer = str(row.get("answer", ""))
    match = _GSM8K_FINAL.search(answer)
    if match is None:
        return None
    return {
        "question": str(row.get("question", "")),
        "answer": match.group(1).replace(",", "").strip(),
        "solution": answer.split("####")[0].strip(),
        "source": "gsm8k",
    }


def convert_math500(row: dict) -> Optional[dict]:
    problem, answer = str(row.get("problem", "")), str(row.get("answer", ""))
    if not problem or not answer:
        return None
    return {
        "question": problem,
        "answer": answer,
        "solution": str(row.get("solution", "")),
        "subject": row.get("subject"),
        "level": row.get("level"),
        "source": "math500",
    }


def _tal_option_content(row: dict) -> Optional[str]:
    """Resolve the answer letter to its option text.

    ``answer_option_list`` is a list of single-element lists of
    ``{"aoVal": "A", "content": "..."}``. The bare letter is useless as a
    verifiable target, so the content is what gets stored.
    """
    letter = str(row.get("answer_value", "")).strip()
    if not letter:
        return None
    for group in _as_list(row.get("answer_option_list")):
        for option in _as_list(group) or ([group] if isinstance(group, dict) else []):
            if isinstance(option, dict) and str(option.get("aoVal", "")).strip() == letter:
                return str(option.get("content", "")).strip()
    return None


def convert_tal_scq5k(row: dict) -> Optional[dict]:
    problem = str(row.get("problem", "")).strip()
    content = _tal_option_content(row)
    if not problem or not content:
        return None
    analysis = _as_list(row.get("answer_analysis"))
    return {
        "question": problem,
        "answer": content,
        "answer_letter": str(row.get("answer_value", "")).strip(),
        "solution": str(analysis[0]) if analysis else "",
        "difficulty": row.get("difficulty"),
        "source": "tal_scq5k",
    }


def convert_mmlu(row: dict) -> Optional[dict]:
    """``answer`` is an index into ``choices``."""
    choices = _as_list(row.get("choices"))
    index = row.get("answer")
    if not choices or not isinstance(index, int) or not 0 <= index < len(choices):
        return None
    return {
        "question": str(row.get("question", "")),
        "answer": str(choices[index]),
        "choices": [str(c) for c in choices],
        "subject": row.get("subject"),
        "source": "mmlu",
    }


def convert_big_math(row: dict) -> Optional[dict]:
    """Keeps ``llama8b_solve_rate``: a per-problem pass rate over 64 rollouts.

    That field is what makes a difficulty curriculum possible. Zero reward is an
    absorbing state for GRPO — a group where every rollout fails has no reward
    variance and therefore no gradient — so prompts can be selected in a
    mid-difficulty band instead of hoping for variance.
    """
    problem, answer = str(row.get("problem", "")), str(row.get("answer", ""))
    if not problem or not answer:
        return None
    return {
        "question": problem,
        "answer": answer,
        "solve_rate": row.get("llama8b_solve_rate"),
        "domain": row.get("domain"),
        "source": "big_math",
    }


@dataclass
class EvalSource:
    name: str
    hf: dict
    convert: Callable[[dict], Optional[dict]]
    note: str = ""
    gated: bool = False
    default_max: Optional[int] = None


EVAL_SOURCES: dict[str, EvalSource] = {
    "gsm8k": EvalSource(
        name="gsm8k",
        hf={"path": "openai/gsm8k", "name": "main", "split": "test"},
        convert=convert_gsm8k,
        note="1319 grade-school word problems; the decontamination target SmolLM2 uses",
    ),
    "math500": EvalSource(
        name="math500",
        hf={"path": "HuggingFaceH4/MATH-500", "split": "test"},
        convert=convert_math500,
        note="The standard 500-problem MATH evaluation subset",
    ),
    "tal_scq5k_cn": EvalSource(
        name="tal_scq5k_cn",
        hf={"path": "math-eval/TAL-SCQ5K", "data_files": "TAL-SCQ5K-CN/test.jsonl", "split": "train"},
        convert=convert_tal_scq5k,
        note="MIT-licensed Chinese competition math; the only clean Chinese verifiable source",
    ),
    "tal_scq5k_en": EvalSource(
        name="tal_scq5k_en",
        hf={"path": "math-eval/TAL-SCQ5K", "data_files": "TAL-SCQ5K-EN/test.jsonl", "split": "train"},
        convert=convert_tal_scq5k,
        note="English half of TAL-SCQ5K",
    ),
    "mmlu": EvalSource(
        name="mmlu",
        hf={"path": "cais/mmlu", "name": "all", "split": "test"},
        convert=convert_mmlu,
        note="14042 items; decontaminate against it even though we will not score well on it",
    ),
    "big_math": EvalSource(
        name="big_math",
        hf={"path": "SynthLabsAI/Big-Math-RL-Verified", "split": "train"},
        convert=convert_big_math,
        note="251k RL prompts with per-problem solve rates; gated, needs an HF token",
        gated=True,
        default_max=20000,
    ),
}

# Sets worth indexing for decontamination. big_math is an RL prompt pool, not an
# eval set, so training on it is intended rather than leakage.
DECONTAMINATION_SETS = ("gsm8k", "math500", "tal_scq5k_cn", "tal_scq5k_en", "mmlu")


@dataclass
class FetchReport:
    name: str
    written: int = 0
    skipped: int = 0
    path: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)


def fetch(source: EvalSource, out_dir: str, max_records: Optional[int] = None) -> FetchReport:
    from datasets import load_dataset

    report = FetchReport(name=source.name)
    limit = max_records if max_records is not None else source.default_max
    try:
        options = dict(source.hf)
        dataset = load_dataset(options.pop("path"), **options)
    except Exception as exc:  # network, gating, or a renamed dataset
        report.error = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
        if source.gated:
            report.notes.append("dataset is gated; accept its terms and set HF_TOKEN")
        return report

    rows = []
    for row in dataset:
        converted = source.convert(dict(row))
        if converted is None:
            report.skipped += 1
            continue
        rows.append(converted)
        if limit is not None and len(rows) >= limit:
            break

    report.path = os.path.join(out_dir, f"{source.name}.jsonl")
    report.written = write_jsonl(report.path, rows)
    return report


def update_spec_decontamination(spec_path: str, paths: Sequence[str]) -> list[str]:
    """Point a mixture spec's ``decontaminate.against`` at the fetched files.

    Without this the check is a no-op, which is the failure mode that makes an
    eval number quietly meaningless.
    """
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    decon = spec.setdefault("decontaminate", {})
    merged = sorted(set(decon.get("against") or []) | set(paths))
    decon["against"] = merged
    decon.setdefault("n", 13)
    decon.setdefault("lcs_threshold", 0.6)
    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return merged


def render(reports: Sequence[FetchReport]) -> str:
    header = f"{'set':<16} {'written':>9} {'skipped':>9}  path"
    lines = [header, "-" * (len(header) + 20)]
    for report in reports:
        if report.error:
            lines.append(f"{report.name:<16} {'FAILED':>9} {'':>9}  {report.error}")
            for note in report.notes:
                lines.append(f"{'':<16} {'':>9} {'':>9}  ! {note}")
        else:
            lines.append(
                f"{report.name:<16} {report.written:>9} {report.skipped:>9}  {report.path}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch benchmark sets as repo JSONL")
    parser.add_argument("--sets", nargs="*", choices=sorted(EVAL_SOURCES), default=None)
    parser.add_argument("--all", action="store_true", help="Fetch every registered set")
    parser.add_argument(
        "--decontamination_only",
        action="store_true",
        help=f"Fetch just the sets worth indexing for leakage: {', '.join(DECONTAMINATION_SETS)}",
    )
    parser.add_argument("--out_dir", type=str, default="datasets/eval")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument(
        "--update_spec",
        type=str,
        default=None,
        help="Mixture spec to point at the fetched files (e.g. configs/mixture_v1.json)",
    )
    args = parser.parse_args()

    if args.all:
        names = list(EVAL_SOURCES)
    elif args.decontamination_only:
        names = list(DECONTAMINATION_SETS)
    elif args.sets:
        names = args.sets
    else:
        parser.error("pass --sets, --all, or --decontamination_only")

    os.makedirs(args.out_dir, exist_ok=True)
    reports = []
    for name in names:
        source = EVAL_SOURCES[name]
        print(f"fetching {name}: {source.note}", flush=True)
        reports.append(fetch(source, args.out_dir, args.max_records))
    print()
    print(render(reports))

    if args.update_spec:
        usable = [
            r.path for r in reports if r.written and r.name in DECONTAMINATION_SETS
        ]
        if usable:
            merged = update_spec_decontamination(args.update_spec, usable)
            print(f"\n{args.update_spec} decontaminate.against -> {merged}")
        else:
            print("\nno decontamination sets fetched; spec left unchanged")


if __name__ == "__main__":
    main()

"""Summarize one or more GRPO metric logs.

``grpo.py`` writes one JSON object per step to ``{save_dir}/grpo_metrics.jsonl``.
Per-step RL numbers are far too noisy to read directly, so this averages them
over fixed windows and lines runs up side by side for ablations::

    python3 analyze_grpo.py results/grpo_metrics.jsonl
    python3 analyze_grpo.py results_a/grpo_metrics.jsonl results_b/grpo_metrics.jsonl --window 25
"""

from __future__ import annotations

import argparse
import json
from statistics import mean
from typing import Iterable, Sequence

TRAIN_COLUMNS = (
    "reward_mean",
    "accuracy",
    "format_rate",
    "hack_rate",
    "silent_group_frac",
    "entropy",
    "kl",
    "grad_norm",
    "completion_len",
)
EVAL_COLUMNS = ("eval_accuracy", "eval_format_rate", "eval_hack_rate")


def read_metrics(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def window_means(
    rows: Sequence[dict],
    window: int,
    columns: Iterable[str] = TRAIN_COLUMNS,
) -> list[dict]:
    """Average each column over ``[k*window, (k+1)*window)`` step buckets.

    Rows missing a column (e.g. eval-only records) are skipped for that column
    rather than counted as zero.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    columns = list(columns)
    buckets: dict[int, list[dict]] = {}
    for row in rows:
        if "step" not in row:
            continue
        buckets.setdefault(int(row["step"]) // window, []).append(row)

    summary = []
    for index in sorted(buckets):
        bucket = buckets[index]
        entry = {
            "start": index * window,
            "end": index * window + window - 1,
            "n": len(bucket),
        }
        for column in columns:
            values = [r[column] for r in bucket if column in r]
            if values:
                entry[column] = mean(values)
        summary.append(entry)
    return summary


def _render(summary: Sequence[dict], columns: Sequence[str]) -> str:
    present = [c for c in columns if any(c in row for row in summary)]
    header = f"{'steps':>13} {'n':>4} " + " ".join(f"{c[:9]:>9}" for c in present)
    lines = [header, "-" * len(header)]
    for row in summary:
        cells = " ".join(
            f"{row[c]:>9.3f}" if c in row else f"{'-':>9}" for c in present
        )
        lines.append(f"{row['start']:>5}-{row['end']:<7} {row['n']:>4} {cells}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GRPO metric logs")
    parser.add_argument("paths", nargs="+", help="One or more grpo_metrics.jsonl files")
    parser.add_argument("--window", type=int, default=25, help="Steps averaged per row")
    args = parser.parse_args()

    for path in args.paths:
        rows = read_metrics(path)
        print(f"\n=== {path}  ({len(rows)} records)")
        print(_render(window_means(rows, args.window), TRAIN_COLUMNS))

        evals = [r for r in rows if any(c in r for c in EVAL_COLUMNS)]
        if evals:
            print("\neval checkpoints:")
            print(_render(window_means(evals, 1, EVAL_COLUMNS), EVAL_COLUMNS))


if __name__ == "__main__":
    main()

"""Turn an RLVR environment into JSONL datasets for the other training stages.

One task definition feeds every stage, which is what makes SFT / DPO / GRPO
comparable on the same problems::

    python3 -m envs.generate_data --split sft        --n 2000 --out datasets/arith_sft.jsonl
    python3 -m envs.generate_data --split preference --n 1000 --out datasets/arith_pref.jsonl
    python3 -m envs.generate_data --split eval       --n 200  --out datasets/arith_eval.jsonl

* ``sft``        — gold chain-of-thought, the cold start GRPO needs to get any reward at all
* ``preference`` — gold vs. a deliberately corrupted answer, for the offline DPO baseline
* ``eval``       — questions plus ground truth, a fixed held-out set for accuracy curves
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Iterable

from .base import TaskEnv, available_envs, make_env

SPLITS = ("sft", "preference", "eval")


def build_rows(env: TaskEnv, split: str, n: int, rng: random.Random) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    for attr in ("gold_completion", "corrupt_completion"):
        if split != "eval" and not hasattr(env, attr):
            raise TypeError(f"env {env.name!r} cannot generate {split} data: missing {attr}()")

    rows = []
    for task in env.sample(n):
        if split == "sft":
            rows.append(
                {
                    "conversations": [
                        {"role": "user", "content": env.render(task)},
                        {"role": "assistant", "content": env.gold_completion(task)},
                    ]
                }
            )
        elif split == "preference":
            rows.append(
                {
                    "prompt": env.render(task),
                    "chosen": env.gold_completion(task),
                    "rejected": env.corrupt_completion(task, rng),
                }
            )
        else:
            rows.append({"question": task.question, "answer": task.answer, **task.meta})
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RLVR datasets from an environment")
    parser.add_argument("--split", choices=list(SPLITS), required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="arithmetic", choices=available_envs())
    parser.add_argument("--ops", type=str, default="+,-")
    parser.add_argument("--min_digits", type=int, default=1)
    parser.add_argument("--max_digits", type=int, default=2)
    args = parser.parse_args()

    kwargs = {"seed": args.seed}
    if args.env == "arithmetic":
        kwargs.update(
            ops=tuple(op.strip() for op in args.ops.split(",") if op.strip()),
            min_digits=args.min_digits,
            max_digits=args.max_digits,
        )
    env = make_env(args.env, **kwargs)
    written = write_jsonl(args.out, build_rows(env, args.split, args.n, random.Random(args.seed)))
    print(f"wrote {written} {args.split} rows -> {args.out}")


if __name__ == "__main__":
    main()

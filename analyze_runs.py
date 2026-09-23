"""Review training runs: list them, inspect one, compare several, plot curves.

Reads the records that ``runlog.RunRecorder`` writes under
``{save_dir}/runs/``. Nothing here needs a tracker service or a plotting
library — the HTML report embeds hand-built SVG, so a run can be reviewed on
any machine that can open a browser, including months later.

::

    python analyze_runs.py list results
    python analyze_runs.py show results/runs/pretrain_20260923-163403_ace3d5
    python analyze_runs.py compare results_zh00 results_zh30 --metric val_loss
    python analyze_runs.py plot results --out report.html
"""

from __future__ import annotations

import argparse
import html
import os
from statistics import mean
from typing import Iterable, Optional, Sequence

from runlog import discover_runs, read_run

# Bookkeeping columns that are never interesting as curves.
NOT_CURVES = {"step", "split", "elapsed_s", "tokens", "tokens_per_s", "epoch", "batches", "pairs"}


def metric_names(run: dict, split: Optional[str] = None) -> list[str]:
    names: set[str] = set()
    for row in run["metrics"]:
        if split and row.get("split") != split:
            continue
        names |= {k for k, v in row.items() if k not in NOT_CURVES and isinstance(v, (int, float))}
    return sorted(names)


def series(run: dict, metric: str, split: str = "train") -> list[tuple[float, float]]:
    """(step, value) points for one metric, in step order."""
    points = [
        (float(row.get("step", 0)), float(row[metric]))
        for row in run["metrics"]
        if row.get("split") == split and isinstance(row.get(metric), (int, float))
    ]
    return sorted(points, key=lambda p: p[0])


def window_means(run: dict, metric: str, split: str, buckets: int = 10) -> list[tuple[int, int, float]]:
    """Average a metric over equal step ranges, so a noisy curve is readable."""
    points = series(run, metric, split)
    if not points:
        return []
    lo, hi = points[0][0], points[-1][0]
    width = max((hi - lo) / buckets, 1e-9)
    grouped: dict[int, list[float]] = {}
    for step, value in points:
        grouped.setdefault(min(int((step - lo) / width), buckets - 1), []).append(value)
    return [
        (int(lo + index * width), int(lo + (index + 1) * width), mean(values))
        for index, values in sorted(grouped.items())
    ]


def _label(run: dict) -> str:
    meta, summary = run["meta"], run["summary"]
    return meta.get("run_id") or summary.get("run_id") or os.path.basename(run["run_dir"])


def render_list(runs: Sequence[dict]) -> str:
    header = f"{'run':<34} {'stage':<9} {'status':<11} {'steps':>7} {'tokens':>12} {'tok/s':>8}  last"
    lines = [header, "-" * (len(header) + 24)]
    for run in runs:
        summary, meta = run["summary"], run["meta"]
        last = summary.get("last", {})
        headline = next(
            (f"{k}={last[k]:.4g}" for k in ("val_loss", "val_accuracy", "loss", "dpo_loss") if k in last),
            "",
        )
        lines.append(
            f"{_label(run)[:34]:<34} {meta.get('stage', '?'):<9} "
            f"{summary.get('status', 'running'):<11} {summary.get('steps', len(run['metrics'])):>7} "
            f"{summary.get('tokens', 0):>12} {summary.get('tokens_per_s', 0):>8.0f}  {headline}"
        )
    return "\n".join(lines)


def render_show(run: dict, buckets: int = 10) -> str:
    meta, summary = run["meta"], run["summary"]
    out = [f"=== {_label(run)}", f"  dir      {run['run_dir']}"]
    out.append(f"  stage    {meta.get('stage')}   started {meta.get('started_at')}")
    status = summary.get("status", "running (no summary.json)")
    out.append(f"  status   {status}   duration {summary.get('duration_s', '?')}s")

    model = meta.get("model") or {}
    if model:
        out.append(
            f"  model    dim={model.get('dim')} layers={model.get('n_layers')} "
            f"heads={model.get('n_heads')} kv={model.get('n_kv_heads')} "
            f"vocab={model.get('vocab_size')} -> {(model.get('params') or 0) / 1e6:.1f}M"
        )
    env = meta.get("env") if isinstance(meta.get("env"), dict) else {}
    git = meta.get("git") or {}
    out.append(
        f"  env      {env.get('device')} / {env.get('accelerator', 'cpu')} "
        f"torch {env.get('torch')} python {env.get('python')}"
    )
    if git:
        out.append(f"  git      {(git.get('commit') or '')[:10]} on {git.get('branch')}"
                   f"{'  (dirty tree)' if git.get('dirty') else ''}")
    for entry in meta.get("data") or []:
        out.append(f"  data     {entry['path']}  {entry['bytes']} bytes  sha {entry['head_sha'][:10]}")
    manifest = meta.get("dataset_manifest")
    if manifest:
        out.append(f"  mixture  {manifest.get('mixture')}  {manifest.get('tokens')} tokens  "
                   f"sources={manifest.get('sources')}")
    for key, value in (meta.get("extra") or {}).items():
        out.append(f"  {key:<8} {value}")
    out.append(f"  throughput {summary.get('tokens', 0)} tokens @ {summary.get('tokens_per_s', 0):.0f} tok/s")
    if summary.get("error"):
        out.append("  error    " + summary["error"].strip().splitlines()[-1])

    for split in ("train", "val"):
        names = metric_names(run, split)
        if not names:
            continue
        out.append(f"\n  --- {split} ---")
        for metric in names:
            rows = window_means(run, metric, split, buckets)
            if not rows:
                continue
            sparkline = " ".join(f"{value:.4g}" for _, _, value in rows)
            out.append(f"  {metric:<16} {sparkline}")
    return "\n".join(out)


def render_compare(runs: Sequence[dict], metric: str, split: str, buckets: int = 6) -> str:
    out = [f"=== {split}/{metric} across {len(runs)} run(s)"]
    width = max((len(_label(r)) for r in runs), default=10)
    for run in runs:
        rows = window_means(run, metric, split, buckets)
        if not rows:
            out.append(f"  {_label(run):<{width}}  (no {split}/{metric})")
            continue
        cells = "  ".join(f"{value:>9.4g}" for _, _, value in rows)
        final = run["summary"].get("last", {}).get(metric if split == "train" else f"val_{metric}")
        out.append(f"  {_label(run):<{width}}  {cells}" + (f"   final={final:.4g}" if final is not None else ""))
    return "\n".join(out)


# -- HTML report ----------------------------------------------------------

_PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d"]


def _svg_chart(title: str, curves: Sequence[tuple[str, Sequence[tuple[float, float]]]],
               width: int = 640, height: int = 260) -> str:
    """Line chart as inline SVG, so the report needs no plotting library."""
    curves = [(name, pts) for name, pts in curves if pts]
    if not curves:
        return ""
    pad_l, pad_r, pad_t, pad_b = 62, 12, 28, 34
    xs = [x for _, pts in curves for x, _ in pts]
    ys = [y for _, pts in curves for _, y in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y0, y1 = y0 - 0.5, y1 + 0.5
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    def sx(x: float) -> float:
        return pad_l + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        return pad_t + plot_h - (y - y0) / (y1 - y0) * plot_h

    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append(f'<text x="{pad_l}" y="18" font-size="13" font-weight="600">{html.escape(title)}</text>')
    for index in range(5):  # horizontal gridlines with value labels
        value = y0 + (y1 - y0) * index / 4
        y = sy(value)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 4:.1f}" font-size="10" fill="#6b7280" '
                     f'text-anchor="end">{value:.4g}</text>')
    parts.append(f'<text x="{pad_l}" y="{height - 8}" font-size="10" fill="#6b7280">{x0:.0f}</text>')
    parts.append(f'<text x="{width - pad_r}" y="{height - 8}" font-size="10" fill="#6b7280" '
                 f'text-anchor="end">step {x1:.0f}</text>')

    for index, (name, pts) in enumerate(curves):
        colour = _PALETTE[index % len(_PALETTE)]
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
        parts.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.8"/>')
        legend_y = pad_t + 12 * index
        parts.append(f'<rect x="{width - pad_r - 130}" y="{legend_y - 8}" width="9" height="9" fill="{colour}"/>')
        parts.append(f'<text x="{width - pad_r - 117}" y="{legend_y}" font-size="10" fill="#374151">'
                     f'{html.escape(name[:24])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_html(runs: Sequence[dict]) -> str:
    """Self-contained report: one chart per metric, every run overlaid."""
    by_metric: dict[tuple[str, str], list[tuple[str, list[tuple[float, float]]]]] = {}
    for run in runs:
        for split in ("train", "val"):
            for metric in metric_names(run, split):
                points = series(run, metric, split)
                if points:
                    by_metric.setdefault((split, metric), []).append((_label(run), points))

    body = [
        "<!doctype html><meta charset='utf-8'><title>Whetstone runs</title>",
        "<style>body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:32px;color:#111}"
        "table{border-collapse:collapse;margin-bottom:24px}td,th{border:1px solid #e5e7eb;padding:4px 10px;"
        "text-align:left;font-size:12px}h1{font-size:20px}h2{font-size:15px;margin-top:28px}"
        ".charts{display:flex;flex-wrap:wrap;gap:16px}</style>",
        f"<h1>Whetstone training runs ({len(runs)})</h1>",
        "<table><tr><th>run</th><th>stage</th><th>status</th><th>params</th><th>device</th>"
        "<th>tokens</th><th>tok/s</th><th>git</th><th>headline</th></tr>",
    ]
    for run in runs:
        meta, summary = run["meta"], run["summary"]
        last = summary.get("last", {})
        headline = next(
            (f"{k}={last[k]:.4g}" for k in ("val_loss", "val_accuracy", "loss", "dpo_loss") if k in last), ""
        )
        params = (meta.get("model") or {}).get("params") or 0
        git = meta.get("git") or {}
        body.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(cell))}</td>"
                for cell in (
                    _label(run), meta.get("stage", "?"), summary.get("status", "running"),
                    f"{params / 1e6:.1f}M", (meta.get("env") or {}).get("device", "?"),
                    summary.get("tokens", 0), f"{summary.get('tokens_per_s', 0):.0f}",
                    (git.get("commit") or "")[:8] + ("*" if git.get("dirty") else ""), headline,
                )
            )
            + "</tr>"
        )
    body.append("</table>")

    for split in ("val", "train"):
        charts = [
            _svg_chart(f"{split} / {metric}", curves)
            for (chart_split, metric), curves in sorted(by_metric.items())
            if chart_split == split
        ]
        charts = [c for c in charts if c]
        if charts:
            body.append(f"<h2>{split}</h2><div class='charts'>{''.join(charts)}</div>")
    return "\n".join(body)


# -- CLI ------------------------------------------------------------------


def load_runs(roots: Iterable[str]) -> list[dict]:
    runs = []
    for root in roots:
        for run_dir in discover_runs(root):
            runs.append(read_run(run_dir))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Review Whetstone training runs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="One line per run")
    p_list.add_argument("roots", nargs="*", default=["results"])

    p_show = sub.add_parser("show", help="Everything recorded about one run")
    p_show.add_argument("roots", nargs="+")
    p_show.add_argument("--buckets", type=int, default=10)

    p_cmp = sub.add_parser("compare", help="One metric across runs, for ablations")
    p_cmp.add_argument("roots", nargs="+")
    p_cmp.add_argument("--metric", default="loss")
    p_cmp.add_argument("--split", default="val", choices=["train", "val"])
    p_cmp.add_argument("--buckets", type=int, default=6)

    p_plot = sub.add_parser("plot", help="Self-contained HTML report with curves")
    p_plot.add_argument("roots", nargs="+")
    p_plot.add_argument("--out", default="runs_report.html")

    args = parser.parse_args()
    runs = load_runs(args.roots)
    if not runs:
        print(f"no runs found under {args.roots} (expected <save_dir>/runs/<run_id>/meta.json)")
        return

    if args.command == "list":
        print(render_list(runs))
    elif args.command == "show":
        for run in runs:
            print(render_show(run, args.buckets))
            print()
    elif args.command == "compare":
        print(render_compare(runs, args.metric, args.split, args.buckets))
    elif args.command == "plot":
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(build_html(runs))
        print(f"wrote {args.out} ({len(runs)} runs) — open it in a browser")


if __name__ == "__main__":
    main()

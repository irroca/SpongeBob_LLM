import json

import pytest

from analyze_grpo import read_metrics, window_means


def test_window_means_buckets_by_step_range():
    rows = [{"step": s, "reward_mean": float(s)} for s in range(6)]

    summary = window_means(rows, window=3, columns=["reward_mean"])

    assert [(r["start"], r["end"], r["n"]) for r in summary] == [(0, 2, 3), (3, 5, 3)]
    assert summary[0]["reward_mean"] == pytest.approx(1.0)
    assert summary[1]["reward_mean"] == pytest.approx(4.0)


def test_window_means_skips_missing_columns_instead_of_zero_filling():
    """Eval records only appear every N steps; averaging them as zeros would
    silently drag every curve toward the floor."""
    rows = [
        {"step": 0, "reward_mean": 1.0},
        {"step": 1, "reward_mean": 3.0, "eval_accuracy": 0.5},
    ]

    summary = window_means(rows, window=2, columns=["reward_mean", "eval_accuracy"])

    assert summary[0]["reward_mean"] == pytest.approx(2.0)
    assert summary[0]["eval_accuracy"] == pytest.approx(0.5)


def test_window_means_ignores_rows_without_a_step():
    summary = window_means([{"reward_mean": 9.0}, {"step": 0, "reward_mean": 1.0}], 2)
    assert len(summary) == 1
    assert summary[0]["reward_mean"] == pytest.approx(1.0)


def test_window_means_rejects_non_positive_window():
    with pytest.raises(ValueError):
        window_means([{"step": 0}], window=0)


def test_read_metrics_tolerates_blank_lines(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps({"step": 0}) + "\n\n" + json.dumps({"step": 1}) + "\n", "utf-8")

    assert [r["step"] for r in read_metrics(str(path))] == [0, 1]

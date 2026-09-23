import argparse

import pytest

from analyze_runs import (
    build_html,
    load_runs,
    metric_names,
    render_compare,
    render_list,
    render_show,
    series,
    window_means,
)
from runlog import RunRecorder, read_run


def _run(tmp_path, name, stage="pretrain", losses=(3.0, 2.0, 1.0), val=(2.5, 1.5)):
    save_dir = tmp_path / name
    args = argparse.Namespace(save_dir=str(save_dir), device="cpu", batch_size=2, seed=1)
    with RunRecorder.start(stage, args) as recorder:
        for step, loss in enumerate(losses, 1):
            recorder.add_tokens(10)
            recorder.log(step, loss=loss, lr=1e-4)
        for step, value in enumerate(val, 1):
            recorder.log_eval(step * 2, loss=value, ppl=2.718 ** value)
    return read_run(recorder.run_dir)


def test_series_returns_sorted_points_for_one_split(tmp_path):
    run = _run(tmp_path, "a")

    train = series(run, "loss", "train")
    val = series(run, "loss", "val")

    assert [v for _, v in train] == [3.0, 2.0, 1.0]
    assert [s for s, _ in train] == [1.0, 2.0, 3.0]
    assert [v for _, v in val] == [2.5, 1.5]


def test_metric_names_excludes_bookkeeping_columns(tmp_path):
    names = metric_names(_run(tmp_path, "a"), "train")
    assert "loss" in names and "lr" in names
    for noise in ("step", "split", "tokens", "tokens_per_s", "elapsed_s"):
        assert noise not in names


def test_window_means_buckets_a_noisy_curve(tmp_path):
    run = _run(tmp_path, "a", losses=tuple(float(i) for i in range(20)))

    rows = window_means(run, "loss", "train", buckets=4)

    assert len(rows) == 4
    assert rows[0][2] < rows[-1][2]  # increasing series stays increasing
    assert all(start <= end for start, end, _ in rows)


def test_window_means_of_a_missing_metric_is_empty(tmp_path):
    assert window_means(_run(tmp_path, "a"), "nope", "train") == []


def test_render_list_summarizes_each_run(tmp_path):
    runs = [_run(tmp_path, "a"), _run(tmp_path, "b", stage="sft")]

    text = render_list(runs)

    assert "pretrain" in text and "sft" in text
    assert "completed" in text
    assert "val_loss=1.5" in text  # headline picks the held-out number


def test_render_show_includes_provenance_not_just_numbers(tmp_path):
    """A curve with no record of which data and commit produced it is not reviewable."""
    text = render_show(_run(tmp_path, "a"))

    assert "stage    pretrain" in text
    assert "env " in text and "torch" in text
    assert "--- train ---" in text and "--- val ---" in text
    assert "throughput" in text


def test_render_show_surfaces_a_failure_reason(tmp_path):
    args = argparse.Namespace(save_dir=str(tmp_path / "boom"), device="cpu", batch_size=1, seed=1)
    with pytest.raises(ValueError):
        with RunRecorder.start("pretrain", args) as recorder:
            recorder.log(1, loss=1.0)
            raise ValueError("loss became nan")

    text = render_show(read_run(recorder.run_dir))

    assert "failed" in text
    assert "loss became nan" in text


def test_render_compare_lines_runs_up_on_one_metric(tmp_path):
    runs = [_run(tmp_path, "a", val=(2.5, 1.5)), _run(tmp_path, "b", val=(2.4, 0.4))]

    text = render_compare(runs, "loss", "val", buckets=2)

    assert text.count("\n") == 2  # header plus one line per run
    assert "final=1.5" in text and "final=0.4" in text


def test_render_compare_notes_a_run_missing_the_metric(tmp_path):
    runs = [_run(tmp_path, "a"), _run(tmp_path, "b", val=())]
    assert "(no val/loss)" in render_compare(runs, "loss", "val")


def test_html_report_embeds_svg_curves_with_no_dependencies(tmp_path):
    runs = [_run(tmp_path, "a"), _run(tmp_path, "b")]

    page = build_html(runs)

    assert page.count("<svg") >= 2          # at least train/loss and val/loss
    assert "<path" in page and "stroke" in page
    assert "http://" not in page and "https://" not in page  # fully self-contained
    assert page.count("<tr>") == len(runs) + 1


def test_html_handles_a_flat_curve_without_dividing_by_zero(tmp_path):
    run = _run(tmp_path, "flat", losses=(2.0, 2.0, 2.0), val=())
    page = build_html([run])
    assert "<svg" in page and "nan" not in page.lower()


def test_html_escapes_run_labels(tmp_path):
    run = _run(tmp_path, "a")
    run["meta"]["run_id"] = "<script>alert(1)</script>"
    page = build_html([run])
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


def test_load_runs_reads_every_run_under_a_save_dir(tmp_path):
    _run(tmp_path, "shared", stage="pretrain")
    _run(tmp_path, "shared", stage="sft")

    runs = load_runs([str(tmp_path / "shared")])

    assert len(runs) == 2
    assert {r["meta"]["stage"] for r in runs} == {"pretrain", "sft"}

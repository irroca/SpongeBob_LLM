import argparse
import json
import os

import pytest

from runlog import RunRecorder, discover_runs, file_fingerprint, read_run


def _args(tmp_path, **overrides):
    values = {"save_dir": str(tmp_path), "device": "cpu", "batch_size": 4, "seed": 7}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_start_writes_meta_with_args_env_and_git(tmp_path):
    recorder = RunRecorder.start("pretrain", _args(tmp_path, learning_rate=5e-4))
    recorder.finish()

    meta = json.loads((tmp_path / "runs" / recorder.run_id / "meta.json").read_text("utf-8"))

    assert meta["stage"] == "pretrain"
    assert meta["args"]["learning_rate"] == pytest.approx(5e-4)
    assert meta["args"]["seed"] == 7
    assert meta["env"]["device"] == "cpu"
    assert meta["env"]["torch"] and meta["env"]["python"]
    assert meta["command"]  # the exact invocation, for re-running it later


def test_meta_records_model_architecture_and_param_count(tmp_path):
    from config import LLMConfig
    from model import Whetstone

    config = LLMConfig(dim=32, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=64, max_seq_len=64)
    model = Whetstone(config)

    recorder = RunRecorder.start("sft", _args(tmp_path), config=config, model=model)
    recorder.finish()

    model_meta = read_run(recorder.run_dir)["meta"]["model"]
    assert (model_meta["dim"], model_meta["n_layers"], model_meta["n_kv_heads"]) == (32, 2, 2)
    # Tied embeddings counted once, matching describe_model.
    assert model_meta["params"] == sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())


def test_metrics_are_appended_with_throughput(tmp_path):
    with RunRecorder.start("pretrain", _args(tmp_path)) as recorder:
        recorder.add_tokens(100)
        first = recorder.log(1, loss=2.0, lr=1e-4)
        recorder.add_tokens(100)
        recorder.log(2, loss=1.5, lr=1e-4)
        recorder.log_eval(2, loss=1.8, ppl=6.05)

    run = read_run(recorder.run_dir)
    assert [r["split"] for r in run["metrics"]] == ["train", "train", "val"]
    assert first["tokens"] == 100
    assert run["metrics"][1]["tokens"] == 200
    assert run["metrics"][1]["tokens_per_s"] > 0
    assert run["metrics"][2]["loss"] == 1.8


def test_summary_tracks_last_and_best_separately_per_split(tmp_path):
    with RunRecorder.start("pretrain", _args(tmp_path)) as recorder:
        recorder.log(1, loss=3.0)
        recorder.log(2, loss=1.0)
        recorder.log(3, loss=2.0)
        recorder.log_eval(3, loss=0.5)

    summary = read_run(recorder.run_dir)["summary"]
    assert summary["status"] == "completed"
    assert summary["last"]["loss"] == 2.0
    assert summary["best"]["loss"] == 1.0  # lower is better for a loss
    assert summary["best"]["val_loss"] == 0.5


def test_bookkeeping_fields_are_not_tracked_as_objectives(tmp_path):
    """'best lr' is just the warmup peak and 'best epoch' is just the last one."""
    with RunRecorder.start("pretrain", _args(tmp_path)) as recorder:
        recorder.log(1, loss=3.0, lr=5e-4, epoch=1, grad_norm=9.0)
        recorder.log(2, loss=2.0, lr=1e-4, epoch=2, grad_norm=1.0)

    best = read_run(recorder.run_dir)["summary"]["best"]
    assert "loss" in best
    for noise in ("lr", "epoch", "grad_norm"):
        assert noise not in best
        assert noise in read_run(recorder.run_dir)["summary"]["last"]


def test_higher_is_better_for_accuracy_like_metrics(tmp_path):
    with RunRecorder.start("dpo", _args(tmp_path)) as recorder:
        recorder.log_eval(1, accuracy=0.4)
        recorder.log_eval(2, accuracy=0.9)
        recorder.log_eval(3, accuracy=0.6)

    summary = read_run(recorder.run_dir)["summary"]
    assert summary["best"]["val_accuracy"] == 0.9
    assert summary["last"]["val_accuracy"] == 0.6


def test_a_crashed_run_still_records_why(tmp_path):
    """A run that dies at 3am should say so, not just stop."""
    with pytest.raises(RuntimeError):
        with RunRecorder.start("pretrain", _args(tmp_path)) as recorder:
            recorder.log(1, loss=2.0)
            raise RuntimeError("cuda out of memory")

    summary = read_run(recorder.run_dir)["summary"]
    assert summary["status"] == "failed"
    assert "cuda out of memory" in summary["error"]
    assert summary["logged_points"] == 1  # points before the crash survive


def test_interrupt_is_distinguished_from_failure(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with RunRecorder.start("pretrain", _args(tmp_path)) as recorder:
            raise KeyboardInterrupt

    assert read_run(recorder.run_dir)["summary"]["status"] == "interrupted"


def test_metrics_survive_a_process_kill(tmp_path):
    """Rows are flushed as they are written, so a run with no summary.json is
    still readable up to the moment it died."""
    recorder = RunRecorder.start("pretrain", _args(tmp_path))
    recorder.log(1, loss=2.0)
    recorder.log(2, loss=1.9)
    # Deliberately no finish(): simulate SIGKILL.

    run = read_run(recorder.run_dir)
    assert len(run["metrics"]) == 2
    assert run["summary"] == {}


def test_read_run_tolerates_a_truncated_last_line(tmp_path):
    recorder = RunRecorder.start("pretrain", _args(tmp_path))
    recorder.log(1, loss=2.0)
    recorder.finish()
    path = os.path.join(recorder.run_dir, "metrics.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"step": 2, "loss": 1.')  # killed mid-write

    assert len(read_run(recorder.run_dir)["metrics"]) == 1


def test_file_fingerprint_changes_with_content(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("hello", "utf-8")
    b.write_text("world", "utf-8")

    fa, fb = file_fingerprint(str(a)), file_fingerprint(str(b))

    assert fa["bytes"] == 5 and fa["head_sha"] != fb["head_sha"]
    assert file_fingerprint(str(tmp_path / "missing.jsonl")) is None


def test_dataset_manifest_links_a_run_to_its_mixture(tmp_path):
    """Without this an ablation run is 'run 3', not 'run 3, 30% Chinese'."""
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text('{"text": "x"}\n', "utf-8")
    (data_dir / "manifest.json").write_text(json.dumps({
        "mixture": "v1-bilingual-verifiable",
        "total_tokens_collected": 12345,
        "sources": [{"name": "zh_web"}, {"name": "en_web"}],
    }), "utf-8")

    recorder = RunRecorder.start(
        "pretrain", _args(tmp_path), data_paths=[str(data_dir / "train.jsonl")]
    )
    recorder.finish()

    manifest = read_run(recorder.run_dir)["meta"]["dataset_manifest"]
    assert manifest["mixture"] == "v1-bilingual-verifiable"
    assert manifest["tokens"] == 12345
    assert manifest["sources"] == ["zh_web", "en_web"]


def test_duplicate_data_paths_are_recorded_once(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("x", "utf-8")
    recorder = RunRecorder.start("pretrain", _args(tmp_path), data_paths=[str(path), str(path), ""])
    recorder.finish()

    assert len(read_run(recorder.run_dir)["meta"]["data"]) == 1


def test_discover_runs_accepts_save_dir_runs_dir_or_run_dir(tmp_path):
    first = RunRecorder.start("pretrain", _args(tmp_path)); first.finish()
    second = RunRecorder.start("sft", _args(tmp_path)); second.finish()

    from_save_dir = discover_runs(str(tmp_path))
    assert len(from_save_dir) == 2
    assert discover_runs(str(tmp_path / "runs")) == from_save_dir
    assert discover_runs(first.run_dir) == [first.run_dir]
    assert discover_runs(str(tmp_path / "nope")) == []


def test_finish_is_idempotent(tmp_path):
    recorder = RunRecorder.start("pretrain", _args(tmp_path))
    assert recorder.finish(status="completed")["status"] == "completed"
    assert recorder.finish(status="failed") == {}
    assert read_run(recorder.run_dir)["summary"]["status"] == "completed"


def test_extra_is_namespaced_and_cannot_clobber_recorded_metadata(tmp_path):
    """grpo passes extra={"rl_env": ...}; merging extras into the top level once
    let a stage silently overwrite the recorded environment with a string."""
    recorder = RunRecorder.start(
        "grpo", _args(tmp_path), extra={"env": "arithmetic", "git": "nope", "grpo": {"group_size": 8}}
    )
    recorder.finish()

    meta = read_run(recorder.run_dir)["meta"]
    assert isinstance(meta["env"], dict) and meta["env"]["device"] == "cpu"
    assert isinstance(meta["git"], dict)
    assert meta["extra"]["env"] == "arithmetic"
    assert meta["extra"]["grpo"]["group_size"] == 8


def test_non_serializable_args_do_not_break_meta(tmp_path):
    recorder = RunRecorder.start("pretrain", _args(tmp_path, weird=object()))
    recorder.finish()
    assert isinstance(read_run(recorder.run_dir)["meta"]["args"]["weird"], str)

import json

import pytest

from datatools.records import write_jsonl
from datatools.stats import (
    analyze,
    char_profile,
    describe,
    percentile,
    render,
    repetition_ratio,
)


def test_describe_reports_percentiles():
    stats = describe(list(range(1, 101)))
    assert stats["count"] == 100
    assert stats["mean"] == pytest.approx(50.5)
    assert stats["min"] == 1 and stats["max"] == 100
    assert stats["p50"] == pytest.approx(50, abs=1)
    assert stats["p90"] == pytest.approx(90, abs=1)


def test_describe_of_empty_input():
    assert describe([]) == {"count": 0}


def test_percentile_clamps_to_range():
    assert percentile([1, 2, 3], 0) == 1
    assert percentile([1, 2, 3], 100) == 3
    assert percentile([], 50) == 0.0


def test_char_profile_separates_scripts():
    profile = char_profile("磨刀abc12")
    assert profile["cjk"] == pytest.approx(2 / 7)
    assert profile["latin"] == pytest.approx(3 / 7)
    assert profile["digit"] == pytest.approx(2 / 7)


def test_char_profile_of_empty_text():
    assert char_profile("")["cjk"] == 0.0


def test_repetition_ratio_flags_looping_text():
    """Boilerplate loops are what teach a model to repeat itself."""
    assert repetition_ratio("abcdefghijklmnopqrstuvwxyz", ngram=10) == pytest.approx(0.0)
    assert repetition_ratio("abcdefghij" * 20, ngram=10) > 0.9


def test_repetition_ratio_of_short_text_is_zero():
    assert repetition_ratio("short", ngram=10) == 0.0


def test_analyze_counts_lines_schemas_and_duplicates(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"text": "磨刀石让刀更锋利"}),
            json.dumps({"text": "磨刀石让刀更锋利"}),
            json.dumps({"text": ""}),
            "{broken",
            "",
            json.dumps({"text": "砺石可以校准刀口"}),
        ]),
        "utf-8",
    )

    report = analyze(str(path))

    assert report["lines"] == {
        "total": 6, "parsed": 4, "blank": 1, "malformed": 1,
        "malformed_examples": report["lines"]["malformed_examples"],
    }
    assert report["schemas"] == {"pretrain": 4}
    assert report["exact_duplicates"] == 1
    assert report["empty_text"] == 1
    assert report["validation_problems"] == {"empty text": 1}
    assert report["chars"]["count"] == 3
    assert report["script_mix"]["cjk"] > 0.8


def test_analyze_reports_conversation_structure(tmp_path):
    path = tmp_path / "sft.jsonl"
    write_jsonl(str(path), [
        {"conversations": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]},
        {"conversations": [{"role": "user", "content": "lonely"}]},
    ])

    report = analyze(str(path))

    assert report["schemas"] == {"sft": 2}
    assert report["sft"]["turns"]["max"] == 2
    assert any("no assistant turn" in k for k in report["validation_problems"])


def test_analyze_detects_preference_length_bias(tmp_path):
    """If chosen is systematically longer, DPO partly learns 'longer is better'."""
    path = tmp_path / "pref.jsonl"
    write_jsonl(str(path), [
        {"prompt": f"q{i}", "chosen": "a" * 200, "rejected": "b" * 10} for i in range(5)
    ])

    report = analyze(str(path))

    assert report["preference"]["chosen_longer_frac"] == 1.0
    assert report["preference"]["mean_length_gap"] == pytest.approx(190.0)
    assert "longer is better" in render(report)


def test_render_omits_the_length_warning_for_balanced_pairs(tmp_path):
    path = tmp_path / "pref.jsonl"
    write_jsonl(str(path), [
        {"prompt": "q1", "chosen": "a" * 50, "rejected": "b" * 60},
        {"prompt": "q2", "chosen": "a" * 60, "rejected": "b" * 50},
    ])

    assert "longer is better" not in render(analyze(str(path)))


def test_analyze_adds_token_stats_when_a_tokenizer_is_given(tmp_path):
    from transformers import AutoTokenizer

    path = tmp_path / "data.jsonl"
    write_jsonl(str(path), [{"text": "磨刀石可以把刀刃磨得更加锋利。"}])
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer/zh_6400")

    report = analyze(str(path), tokenizer=tokenizer)

    assert report["tokens"]["count"] == 1
    assert report["tokens"]["max"] > 0
    assert "tokens:" in render(report)


def test_analyze_respects_max_records(tmp_path):
    path = tmp_path / "data.jsonl"
    write_jsonl(str(path), [{"text": f"doc {i}"} for i in range(10)])

    assert analyze(str(path), max_records=4)["lines"]["parsed"] == 4


def test_render_surfaces_malformed_line_numbers(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": "ok"}\n{oops\n', "utf-8")

    text = render(analyze(str(path)))

    assert "1 malformed" in text
    assert "line 2:" in text

import json

import pytest

from datatools.records import (
    PREFERENCE,
    PRETRAIN,
    SFT,
    TASK,
    UNKNOWN,
    ReadStats,
    detect_schema,
    normalize_text,
    prompt_text,
    read_jsonl,
    record_text,
    validate,
    write_jsonl,
)


@pytest.mark.parametrize(
    "record,expected",
    [
        ({"text": "hello"}, PRETRAIN),
        ({"conversations": [{"role": "user", "content": "hi"}]}, SFT),
        ({"prompt": "p", "chosen": "c", "rejected": "r"}, PREFERENCE),
        ({"question": "1+1", "answer": "2"}, TASK),
        ({"something": "else"}, UNKNOWN),
        ("not a dict", UNKNOWN),
    ],
)
def test_detect_schema(record, expected):
    assert detect_schema(record) == expected


def test_record_text_flattens_each_schema_consistently():
    assert record_text({"text": "hello"}) == "hello"
    assert record_text({"conversations": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]}) == "user: hi\nassistant: yo"
    assert record_text({"prompt": "p", "chosen": "c", "rejected": "r"}) == "p\nc\nr"
    assert record_text({"question": "1+1", "answer": "2"}) == "1+1 => 2"


def test_prompt_text_isolates_the_input_side():
    """Contamination is about the prompt, not the full record: the same question with
    a different answer is still leakage."""
    assert prompt_text({"prompt": "p", "chosen": "c", "rejected": "r"}) == "p"
    assert prompt_text({"question": "1+1", "answer": "2"}) == "1+1"
    assert prompt_text({"conversations": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "the question"},
        {"role": "assistant", "content": "the answer"},
    ]}) == "the question"


def test_normalize_text_collapses_formatting_noise():
    assert normalize_text("  a   b\n\tc ") == "a b c"
    assert normalize_text("ＡＢ") == "AB"  # NFKC folds fullwidth forms
    assert normalize_text("AbC", casefold=True) == "abc"
    assert normalize_text("AbC") == "AbC"


def test_validate_accepts_well_formed_records():
    assert validate({"text": "hello"}) == []
    assert validate({"conversations": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]}) == []
    assert validate({"prompt": "p", "chosen": "c", "rejected": "r"}) == []


def test_validate_flags_empty_and_structural_problems():
    assert validate({"text": "   "}) == ["empty text"]
    assert validate({"nope": 1})[0].startswith("unknown schema")
    assert "conversations must be a non-empty list" in validate({"conversations": []})


def test_validate_flags_sft_without_an_assistant_turn():
    """Training on this record computes loss over zero tokens."""
    problems = validate({"conversations": [{"role": "user", "content": "hi"}]})
    assert any("no assistant turn" in p for p in problems)


def test_validate_flags_identical_preference_pair():
    problems = validate({"prompt": "p", "chosen": "same", "rejected": "same"})
    assert any("no preference" in p for p in problems)


def test_read_jsonl_skips_malformed_lines_without_aborting(tmp_path):
    """One bad line in a huge dump must be reported, not fatal."""
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"text": "good"}),
            "{not json",
            "",
            json.dumps(["an array, not an object"]),
            json.dumps({"text": "also good"}),
        ]),
        "utf-8",
    )
    stats = ReadStats()

    records = list(read_jsonl(str(path), stats))

    assert [r.data["text"] for r in records] == ["good", "also good"]
    assert stats.parsed == 2
    assert stats.malformed == 2
    assert stats.blank == 1
    assert stats.malformed_examples[0][0] == 2  # line numbers are 1-based


def test_read_jsonl_respects_max_records(tmp_path):
    path = tmp_path / "many.jsonl"
    write_jsonl(str(path), [{"text": str(i)} for i in range(10)])

    assert len(list(read_jsonl(str(path), max_records=3))) == 3


def test_write_jsonl_roundtrips_unicode(tmp_path):
    path = tmp_path / "out.jsonl"
    rows = [{"text": "海绵宝宝"}, {"text": "hello"}]

    assert write_jsonl(str(path), rows) == 2
    assert [r.data for r in read_jsonl(str(path))] == rows
    assert "海绵宝宝" in path.read_text("utf-8")  # not \u-escaped

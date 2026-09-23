"""Converters are tested offline against rows recorded from the HuggingFace API,
so the field names stay pinned without the suite needing network access."""

import json

import pytest

from datatools.fetch_evals import (
    DECONTAMINATION_SETS,
    EVAL_SOURCES,
    FetchReport,
    convert_big_math,
    convert_gsm8k,
    convert_math500,
    convert_mmlu,
    convert_tal_scq5k,
    render,
    update_spec_decontamination,
)
from datatools.records import detect_schema, record_parts, validate

GSM8K_ROW = {
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half "
                "as many clips in May. How many clips did Natalia sell altogether?",
    "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
              "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72",
}

MATH500_ROW = {
    "problem": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.",
    "solution": "We have that $r = \\sqrt{0^2 + 3^2} = 3.$",
    "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
    "subject": "Precalculus",
    "level": "2",
    "unique_id": "test/precalculus/807.json",
}

TAL_ROW = {
    "qtype": "single_choice",
    "problem": "奶奶告诉小明：2006年共有53个星期日。小明立刻告诉奶奶：2007年的元旦一定是．",
    "answer_option_list": [
        [{"aoVal": "A", "content": "星期一 "}],
        [{"aoVal": "B", "content": "星期二 "}],
        [{"aoVal": "C", "content": "星期三 "}],
    ],
    "answer_analysis": ["2006年有365天，而365=7×52+1，所以2007年元旦是星期一。"],
    "answer_value": "B",
    "difficulty": "2",
}

MMLU_ROW = {
    "question": "What is the capital of France?",
    "subject": "geography",
    "choices": ["Berlin", "Paris", "Madrid", "Rome"],
    "answer": 1,
}

BIG_MATH_ROW = {
    "problem": "What is the remainder when 7^100 is divided by 5?",
    "answer": "1",
    "source": "olympiads",
    "domain": "number theory",
    "llama8b_solve_rate": 0.375,
}


def test_gsm8k_extracts_the_final_answer_after_the_marker():
    record = convert_gsm8k(GSM8K_ROW)
    assert record["answer"] == "72"
    assert record["question"].startswith("Natalia sold clips")
    assert "#### " not in record["solution"]
    assert "48/2" in record["solution"]  # chain of thought retained for decontamination


def test_gsm8k_strips_thousands_separators():
    record = convert_gsm8k({"question": "q", "answer": "reasoning\n#### 1,234"})
    assert record["answer"] == "1234"


def test_gsm8k_row_without_a_marker_is_skipped():
    assert convert_gsm8k({"question": "q", "answer": "no marker here"}) is None


def test_math500_keeps_problem_answer_and_solution():
    record = convert_math500(MATH500_ROW)
    assert record["answer"] == "\\left( 3, \\frac{\\pi}{2} \\right)"
    assert record["subject"] == "Precalculus"
    assert record["solution"].startswith("We have that")


def test_math500_row_missing_an_answer_is_skipped():
    assert convert_math500({"problem": "p", "answer": ""}) is None


def test_tal_resolves_the_letter_to_the_option_text():
    """A bare 'B' is useless as a verifiable target."""
    record = convert_tal_scq5k(TAL_ROW)
    assert record["answer"] == "星期二"
    assert record["answer_letter"] == "B"
    assert record["solution"].startswith("2006年有365天")


def test_tal_accepts_list_fields_serialized_as_repr():
    """The HF viewer (and some loaders) hand these back as Python reprs."""
    row = dict(TAL_ROW)
    row["answer_option_list"] = str(TAL_ROW["answer_option_list"])
    row["answer_analysis"] = str(TAL_ROW["answer_analysis"])

    record = convert_tal_scq5k(row)

    assert record["answer"] == "星期二"
    assert record["solution"].startswith("2006年")


def test_tal_row_with_an_unmatched_letter_is_skipped():
    row = dict(TAL_ROW, answer_value="Z")
    assert convert_tal_scq5k(row) is None


def test_tal_ignores_malformed_option_lists():
    assert convert_tal_scq5k(dict(TAL_ROW, answer_option_list="[not valid python")) is None


def test_mmlu_resolves_the_answer_index():
    record = convert_mmlu(MMLU_ROW)
    assert record["answer"] == "Paris"
    assert record["choices"][0] == "Berlin"


@pytest.mark.parametrize("bad", [{"choices": [], "answer": 0}, {"choices": ["a"], "answer": 5},
                                 {"choices": ["a"], "answer": None}])
def test_mmlu_rejects_out_of_range_indices(bad):
    assert convert_mmlu({"question": "q", **bad}) is None


def test_big_math_keeps_the_solve_rate():
    """The per-problem pass rate is what makes a difficulty curriculum possible:
    a group where every rollout fails has no reward variance and no gradient."""
    record = convert_big_math(BIG_MATH_ROW)
    assert record["solve_rate"] == pytest.approx(0.375)
    assert record["answer"] == "1"


@pytest.mark.parametrize(
    "converter,row",
    [
        (convert_gsm8k, GSM8K_ROW),
        (convert_math500, MATH500_ROW),
        (convert_tal_scq5k, TAL_ROW),
        (convert_mmlu, MMLU_ROW),
        (convert_big_math, BIG_MATH_ROW),
    ],
)
def test_every_converter_emits_a_valid_task_record(converter, row):
    """Output must load through envs.base.load_tasks and datatools.records alike."""
    record = converter(row)
    assert detect_schema(record) == "task"
    assert validate(record) == []


def test_solution_is_indexed_for_decontamination():
    """A page quoting only the worked solution is still contamination."""
    record = convert_gsm8k(GSM8K_ROW)
    parts = record_parts(record)
    assert record["question"] in parts
    assert record["answer"] in parts
    assert record["solution"] in parts


def test_registry_covers_the_decontamination_sets():
    assert set(DECONTAMINATION_SETS) <= set(EVAL_SOURCES)
    assert "big_math" not in DECONTAMINATION_SETS  # an RL prompt pool, not an eval set
    assert EVAL_SOURCES["big_math"].gated
    for source in EVAL_SOURCES.values():
        assert "path" in source.hf
        assert source.note


def test_update_spec_points_decontamination_at_the_files(tmp_path):
    spec_path = tmp_path / "mix.json"
    spec_path.write_text(json.dumps({"name": "t", "sources": []}), "utf-8")

    merged = update_spec_decontamination(str(spec_path), ["a/gsm8k.jsonl", "a/math500.jsonl"])

    written = json.loads(spec_path.read_text("utf-8"))
    assert merged == ["a/gsm8k.jsonl", "a/math500.jsonl"]
    assert written["decontaminate"]["against"] == merged
    assert written["decontaminate"]["n"] == 13
    assert written["decontaminate"]["lcs_threshold"] == 0.6


def test_update_spec_merges_without_duplicating(tmp_path):
    spec_path = tmp_path / "mix.json"
    spec_path.write_text(json.dumps({"decontaminate": {"against": ["a/gsm8k.jsonl"], "n": 13}}), "utf-8")

    merged = update_spec_decontamination(str(spec_path), ["a/gsm8k.jsonl", "a/mmlu.jsonl"])

    assert merged == ["a/gsm8k.jsonl", "a/mmlu.jsonl"]


def test_render_surfaces_failures_and_hints():
    reports = [
        FetchReport(name="gsm8k", written=1319, skipped=0, path="datasets/eval/gsm8k.jsonl"),
        FetchReport(name="big_math", error="GatedRepoError: access required",
                    notes=["dataset is gated; accept its terms and set HF_TOKEN"]),
    ]
    text = render(reports)
    assert "1319" in text
    assert "FAILED" in text
    assert "HF_TOKEN" in text

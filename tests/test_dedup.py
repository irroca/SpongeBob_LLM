import json

import pytest

from datatools.dedup import (
    contaminated_indices,
    dedup_records,
    exact_duplicate_indices,
    load_prompt_set,
    render,
)
from datatools.records import write_jsonl

BASE = "海绵宝宝住在比奇堡的一个菠萝里面，他每天都去蟹堡王上班，最喜欢的事情是抓水母。"
OTHER = "深度学习模型的训练需要大量高质量的语料数据，数据质量往往比模型结构更重要。"


def test_exact_duplicates_keep_the_first_occurrence():
    assert exact_duplicate_indices(["a", "b", "a", "c", "b"]) == {2, 4}


def test_exact_duplicates_ignore_whitespace_and_width_differences():
    """NFKC + whitespace collapse, so reformatting is not a new document."""
    assert exact_duplicate_indices(["a b", "a   b", "ａ b"]) == {1, 2}


def test_exact_duplicates_are_case_sensitive_unless_asked():
    assert exact_duplicate_indices(["Hello", "hello"]) == set()
    assert exact_duplicate_indices(["Hello", "hello"], casefold=True) == {1}


def test_dedup_removes_exact_then_near_duplicates():
    records = [
        {"text": BASE},
        {"text": BASE},                               # exact
        {"text": BASE.replace("抓水母", "吹泡泡")},     # near
        {"text": OTHER},
    ]

    kept, report = dedup_records(records, threshold=0.7, num_perm=256, seed=0)

    assert [r["text"] for r in kept] == [BASE, OTHER]
    assert report.exact_removed == 1
    assert report.near_removed == 1
    assert report.kept == 2
    assert report.retention == pytest.approx(0.5)


def test_dedup_preserves_input_order_and_keeps_the_earliest_member():
    records = [{"text": OTHER}, {"text": BASE}, {"text": BASE.replace("抓水母", "吹泡泡")}]

    kept, _ = dedup_records(records, threshold=0.7, num_perm=256, seed=0)

    assert [r["text"] for r in kept] == [OTHER, BASE]


def test_exact_only_mode_leaves_near_duplicates_alone():
    records = [{"text": BASE}, {"text": BASE.replace("抓水母", "吹泡泡")}]

    kept, report = dedup_records(records, near=False)

    assert len(kept) == 2
    assert report.near_removed == 0
    assert report.bands == 0


def test_dedup_works_across_schemas_not_just_pretrain_text():
    records = [
        {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]},
        {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]},
        {"conversations": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "cya"}]},
    ]

    kept, report = dedup_records(records, threshold=0.9, num_perm=64)

    assert len(kept) == 2
    assert report.exact_removed == 1


def test_contamination_matches_on_the_prompt_not_the_answer():
    """Same question with a different answer is still leakage."""
    holdout = {"1 + 1"}
    records = [
        {"question": "1 + 1", "answer": "2"},
        {"question": "1 + 1", "answer": "3"},
        {"question": "2 + 2", "answer": "4"},
    ]

    assert contaminated_indices(records, holdout) == {0, 1}


def test_dedup_drops_records_contaminated_with_holdout_prompts():
    records = [
        {"prompt": "q1", "chosen": "a", "rejected": "b"},
        {"prompt": "q2", "chosen": "c", "rejected": "d"},
    ]

    kept, report = dedup_records(records, holdout_prompts={"q1"}, num_perm=64)

    assert [r["prompt"] for r in kept] == ["q2"]
    assert report.contaminated_removed == 1


def test_removal_reasons_do_not_double_count():
    """A record that is both a duplicate and contaminated is attributed to one reason,
    so the counts still add up to what was removed."""
    records = [
        {"question": "dup", "answer": "x"},
        {"question": "dup", "answer": "x"},
        {"question": "clean", "answer": "y"},
    ]

    kept, report = dedup_records(records, holdout_prompts={"dup"}, num_perm=64)

    assert report.exact_removed == 1
    assert report.contaminated_removed == 1
    assert report.input_records - report.exact_removed - report.near_removed \
        - report.contaminated_removed == report.kept == len(kept)


def test_load_prompt_set_reads_normalized_prompts(tmp_path):
    path = tmp_path / "eval.jsonl"
    write_jsonl(str(path), [{"question": "  1 +  1 ", "answer": "2"}, {"question": "", "answer": "x"}])

    prompts = load_prompt_set([str(path)])

    assert prompts == {"1 + 1"}  # whitespace collapsed, empty prompt skipped


def test_report_render_mentions_every_removal_reason():
    records = [{"text": BASE}, {"text": BASE}, {"text": OTHER}]
    _, report = dedup_records(records, num_perm=64)

    text = render(report)

    assert "exact duplicates removed: 1" in text
    assert "near duplicates removed" in text
    assert "retention" in text


def test_report_serializes_to_json():
    _, report = dedup_records([{"text": BASE}, {"text": BASE}], num_perm=64)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["kept"] == 1
    assert payload["retention"] == pytest.approx(0.5)


def test_explicit_banding_trades_recall_for_speed():
    """LSH banding only affects which pairs get *checked*; every candidate is still
    verified against the full signature, so more bands can only raise recall."""
    records = [{"text": BASE}, {"text": BASE.replace("抓水母", "吹泡泡")}]

    coarse, coarse_report = dedup_records(
        records, threshold=0.5, num_perm=64, bands=1, rows=64, seed=0
    )
    fine, fine_report = dedup_records(
        records, threshold=0.5, num_perm=64, bands=32, rows=2, seed=0
    )

    assert coarse_report.bands == 1 and fine_report.bands == 32
    assert len(fine) <= len(coarse)
    assert fine_report.near_removed == 1  # fine banding catches the near duplicate


def test_dedup_of_empty_input_is_a_noop():
    kept, report = dedup_records([])
    assert kept == [] and report.kept == 0 and report.retention == 0.0

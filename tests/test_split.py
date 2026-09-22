import pytest

from datatools.split import HOLDOUT, TRAIN, VAL, assign_split, bucket, render, split_records


def _records(n):
    return [{"text": f"document number {i} with some filler text"} for i in range(n)]


def test_bucket_is_in_range_and_deterministic():
    value = bucket("hello")
    assert 0.0 <= value < 1.0
    assert bucket("hello") == value


def test_bucket_ignores_formatting_differences():
    """Normalized content, so a reformatted document does not switch splits."""
    assert bucket("a  b") == bucket("a b")


def test_salt_changes_the_draw():
    assert bucket("hello", salt="v1") != bucket("hello", salt="v2")


def test_assign_split_boundaries():
    assert assign_split("x", val_fraction=0.0, holdout_fraction=1.0 - 1e-9) in (HOLDOUT, VAL, TRAIN)
    assert assign_split("x", val_fraction=0.0, holdout_fraction=0.0) == TRAIN


def test_split_proportions_are_approximately_right():
    splits, report = split_records(_records(4000), val_fraction=0.05, holdout_fraction=0.05)

    assert report.total == 4000
    assert report.counts[VAL] / 4000 == pytest.approx(0.05, abs=0.02)
    assert report.counts[HOLDOUT] / 4000 == pytest.approx(0.05, abs=0.02)
    assert len(splits[TRAIN]) + len(splits[VAL]) + len(splits[HOLDOUT]) == 4000


def test_split_is_reproducible_across_runs():
    """Two ablation runs must differ only in the variable under test."""
    first, _ = split_records(_records(500), 0.1, 0.1)
    second, _ = split_records(_records(500), 0.1, 0.1)
    assert [r["text"] for r in first[VAL]] == [r["text"] for r in second[VAL]]


def test_split_is_stable_when_the_corpus_grows():
    """Adding documents must not reshuffle existing ones, or validation curves
    stop being comparable across data versions."""
    small, _ = split_records(_records(200), 0.1, 0.1)
    large, _ = split_records(_records(400), 0.1, 0.1)

    small_val = {r["text"] for r in small[VAL]}
    large_val = {r["text"] for r in large[VAL]}
    assert small_val <= large_val


def test_identical_documents_never_straddle_splits():
    duplicates = [{"text": "exactly the same text"} for _ in range(50)]
    splits, _ = split_records(duplicates, 0.3, 0.3)
    non_empty = [name for name, rows in splits.items() if rows]
    assert len(non_empty) == 1


def test_zero_fractions_put_everything_in_train():
    splits, report = split_records(_records(100), 0.0, 0.0)
    assert report.counts[TRAIN] == 100
    assert not splits[VAL] and not splits[HOLDOUT]


@pytest.mark.parametrize("val,holdout", [(-0.1, 0.1), (0.6, 0.6), (1.0, 0.0)])
def test_invalid_fractions_rejected(val, holdout):
    with pytest.raises(ValueError):
        split_records(_records(10), val, holdout)


def test_render_lists_all_three_splits():
    _, report = split_records(_records(100), 0.1, 0.1)
    text = render(report)
    for name in (TRAIN, VAL, HOLDOUT):
        assert name in text

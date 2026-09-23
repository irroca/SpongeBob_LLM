"""The committed fixtures are also the README's CPU smoke data, so their own
quality is worth asserting — the repo's data tools flagged real problems in the
previous versions of these files."""

import pytest

from datatools.records import detect_schema, read_jsonl, validate
from datatools.stats import analyze

PRETRAIN = "tests/fixtures/pretrain_tiny.jsonl"
SFT = "tests/fixtures/sft_tiny.jsonl"
PREFERENCE = "tests/fixtures/preference_tiny.jsonl"


@pytest.mark.parametrize("path,schema", [(PRETRAIN, "pretrain"), (SFT, "sft"), (PREFERENCE, "preference")])
def test_fixtures_are_valid_and_single_schema(path, schema):
    records = [r.data for r in read_jsonl(path)]
    assert records
    assert {detect_schema(r) for r in records} == {schema}
    for record in records:
        assert validate(record) == [], record


def test_preference_fixture_carries_no_length_signal():
    """If chosen were systematically longer, DPO would partly learn 'longer is
    better' and the fixture would be teaching the wrong thing."""
    report = analyze(PREFERENCE)["preference"]
    assert report["chosen_longer_frac"] == pytest.approx(0.5)
    assert abs(report["mean_length_gap"]) < 10


@pytest.mark.parametrize("path", [PRETRAIN, SFT, PREFERENCE])
def test_fixtures_have_no_duplicates(path):
    assert analyze(path)["exact_duplicates"] == 0


def test_fixtures_are_bilingual():
    """The project targets Chinese and English, so the smoke data should too."""
    for path in (PRETRAIN, SFT):
        mix = analyze(path)["script_mix"]
        assert mix["cjk"] > 0.15, path
        assert mix["latin"] > 0.15, path

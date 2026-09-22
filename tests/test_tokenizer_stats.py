import pytest
from transformers import AutoTokenizer

from datatools.records import write_jsonl
from datatools.tokenizer_stats import (
    PROBES,
    Fertility,
    measure,
    measure_corpus,
    project_tokens,
    render,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("./spongebob_tokenizer")


def test_fertility_ratios():
    stats = Fertility(documents=2, chars=100, tokens=25, single_char_tokens=5, unk_tokens=1)
    assert stats.chars_per_token == pytest.approx(4.0)
    assert stats.tokens_per_doc == pytest.approx(12.5)
    assert stats.single_char_frac == pytest.approx(0.2)
    assert stats.unk_frac == pytest.approx(0.04)


def test_fertility_of_empty_input_does_not_divide_by_zero():
    stats = Fertility(0, 0, 0, 0, 0)
    assert stats.chars_per_token == 0.0
    assert stats.tokens_per_doc == 0.0
    assert stats.single_char_frac == 0.0


def test_measure_skips_empty_documents(tokenizer):
    stats = measure(tokenizer, ["hello", "", "world"])
    assert stats.documents == 2


def test_measure_counts_match_the_tokenizer(tokenizer):
    text = "海绵宝宝喜欢抓水母"
    stats = measure(tokenizer, [text])
    assert stats.tokens == len(tokenizer(text, add_special_tokens=False).input_ids)
    assert stats.chars == len(text)


def test_chinese_vocabulary_fragments_code_relative_to_english_prose(tokenizer):
    """The committed 6400-token vocabulary was trained on Chinese. English prose
    and source code are both ASCII, and code is the more repetitive of the two, so
    a vocabulary that covered code would compress it at least as well as prose.
    It does not — which is the measurement that says retrain before adding code.

    Note the comparison is prose vs code, not code vs Chinese: Chinese single
    characters are meaningful units, so a high single_char_frac there is expected
    and that metric is not comparable across scripts.
    """
    en = measure(tokenizer, [PROBES["en"]])
    code = measure(tokenizer, [PROBES["code"]])

    assert en.chars_per_token > 3.5
    assert code.chars_per_token < 0.7 * en.chars_per_token


def test_chinese_single_char_fraction_is_high_by_construction(tokenizer):
    """Guards the interpretation above: don't read this as fragmentation."""
    zh = measure(tokenizer, [PROBES["zh"]])
    assert zh.single_char_frac > 0.5
    assert zh.chars_per_token > 1.2


def test_measure_corpus_reads_every_schema(tmp_path, tokenizer):
    path = tmp_path / "mixed.jsonl"
    write_jsonl(str(path), [
        {"text": "海绵宝宝喜欢抓水母"},
        {"conversations": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]},
        {"question": "1 + 1", "answer": "2"},
    ])

    stats = measure_corpus(tokenizer, str(path))

    assert stats.documents == 3
    assert stats.tokens > 0


def test_measure_corpus_respects_max_records(tmp_path, tokenizer):
    path = tmp_path / "many.jsonl"
    write_jsonl(str(path), [{"text": f"文档 {i}"} for i in range(20)])
    assert measure_corpus(tokenizer, str(path), max_records=5).documents == 5


def test_project_tokens_scales_by_document_count():
    """Corpora are published in documents; mixture weights are in tokens."""
    sample = Fertility(documents=100, chars=0, tokens=5000, single_char_tokens=0, unk_tokens=0)
    assert project_tokens(sample, 1_000_000) == pytest.approx(50_000_000)


def test_render_includes_a_row_per_measurement(tokenizer):
    rows = [
        ("./spongebob_tokenizer", "probe:zh", measure(tokenizer, [PROBES["zh"]])),
        ("./spongebob_tokenizer", "probe:code", measure(tokenizer, [PROBES["code"]])),
    ]
    text = render(rows)
    assert "probe:zh" in text and "probe:code" in text
    assert len(text.splitlines()) == 4  # header + rule + 2 rows

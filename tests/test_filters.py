import pytest

from datatools.filters import (
    FilterConfig,
    cjk_ratio,
    describe_rejections,
    digit_ratio,
    duplicate_line_ratio,
    filter_texts,
    latin_ratio,
    mean_word_length,
    ngram_repetition,
    reject_reason,
    symbol_ratio,
)


def test_script_ratios():
    assert cjk_ratio("海绵abc") == pytest.approx(2 / 5)
    assert latin_ratio("海绵abc") == pytest.approx(3 / 5)
    assert digit_ratio("a1b2") == pytest.approx(0.5)
    assert symbol_ratio("ab!!") == pytest.approx(0.5)


def test_ratios_of_empty_text_are_zero():
    for fn in (cjk_ratio, latin_ratio, digit_ratio, symbol_ratio):
        assert fn("") == 0.0


def test_ngram_repetition_flags_loops():
    assert ngram_repetition("abcdefghijklmnop") == pytest.approx(0.0)
    assert ngram_repetition("abcdefghij" * 20) > 0.9


def test_duplicate_line_ratio_catches_boilerplate_blocks():
    """Each page's combination of menu items is unique, so document-level dedup
    misses these; the per-document line ratio does not."""
    text = "首页\n关于我们\n联系我们\n首页\n关于我们\n联系我们\n"
    assert duplicate_line_ratio(text) == pytest.approx(0.5)
    assert duplicate_line_ratio("one line") == 0.0


def test_mean_word_length():
    assert mean_word_length("aa bbb cccc") == pytest.approx(3.0)
    assert mean_word_length("海绵宝宝") == 0.0


def test_passing_text_has_no_reason():
    config = FilterConfig(min_chars=5)
    assert reject_reason("这是一段足够长的正常中文文本。", config) is None


@pytest.mark.parametrize(
    "text,config,expected",
    [
        ("hi", FilterConfig(min_chars=10), "too_short"),
        ("x" * 100, FilterConfig(max_chars=10), "too_long"),
        ("hello world", FilterConfig(min_cjk_ratio=0.5), "low_cjk_ratio"),
        ("海绵宝宝", FilterConfig(min_latin_ratio=0.5), "low_latin_ratio"),
        ("1234567890", FilterConfig(max_digit_ratio=0.5), "high_digit_ratio"),
        ("!!!!!!!!!!", FilterConfig(max_symbol_ratio=0.5), "high_symbol_ratio"),
        ("abcdefghij" * 20, FilterConfig(max_repetition=0.5), "repetitive"),
        ("dup\ndup\ndup\n", FilterConfig(max_duplicate_lines=0.5), "duplicate_lines"),
        ("a b c d e f", FilterConfig(min_mean_word_length=3.0), "short_words"),
        ("abcdefghijklmnop", FilterConfig(max_mean_word_length=5.0), "long_words"),
        ("buy viagra now", FilterConfig(blocklist=("VIAGRA",)), "blocklist"),
    ],
)
def test_each_rule_names_itself(text, config, expected):
    assert reject_reason(text, config) == expected


def test_length_is_checked_before_per_character_scans():
    """A huge junk document should be discarded on its length, not scanned."""
    config = FilterConfig(max_chars=10, max_symbol_ratio=0.1)
    assert reject_reason("!" * 100000, config) == "too_long"


def test_word_length_rules_skip_text_without_latin_words():
    """Chinese has no latin word runs; a word-length rule must not reject it."""
    config = FilterConfig(min_mean_word_length=3.0, max_mean_word_length=10.0)
    assert reject_reason("海绵宝宝住在比奇堡的菠萝里。", config) is None


def test_none_thresholds_disable_rules():
    permissive = FilterConfig(
        min_chars=1, max_chars=None, max_digit_ratio=None,
        max_repetition=None, max_duplicate_lines=None,
    )
    assert reject_reason("1234567890" * 50, permissive) is None


def test_filter_texts_partitions_and_attributes():
    texts = ["ok text here", "hi", "ok another one", "!!!!!!!!!!!!"]
    config = FilterConfig(min_chars=5, max_symbol_ratio=0.5)

    kept, rejected = filter_texts(texts, config)

    assert kept == [0, 2]
    assert rejected == {"too_short": 1, "high_symbol_ratio": 1}


def test_config_from_dict_rejects_typos():
    """A silently ignored threshold would delete data for no reason."""
    with pytest.raises(ValueError, match="Unknown filter keys"):
        FilterConfig.from_dict({"min_char": 100})


def test_config_from_dict_accepts_known_keys():
    config = FilterConfig.from_dict({"min_chars": 50, "blocklist": ["spam"]})
    assert config.min_chars == 50
    assert config.blocklist == ("spam",)


def test_describe_rejections_reports_shares():
    text = describe_rejections(10, {"too_short": 3, "repetitive": 2})
    assert "kept 5/10" in text
    assert "too_short" in text and "repetitive" in text

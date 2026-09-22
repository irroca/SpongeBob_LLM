import json

import pytest
from transformers import AutoTokenizer

from datatools.decontaminate import build_eval_index
from datatools.prepare import (
    MixtureSpec,
    SourceSpec,
    count_tokens,
    finalize,
    iter_raw,
    prepare_source,
    render_sources,
    to_record,
)
from datatools.records import read_jsonl, write_jsonl
from datatools.split import HOLDOUT, TRAIN, VAL


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("./tokenizer/zh_6400")


_SENTENCES = [
    "深度学习模型的训练需要高质量的语料数据，否则再好的结构也难以收敛。",
    "分词器的压缩率会直接影响有效上下文长度和训练效率。",
    "强化学习阶段的奖励设计往往比算法本身更容易出错。",
    "小模型在双语场景下会面临明显的容量竞争问题。",
    "去重和污染检查是评测数字可信的前提条件。",
]


def _corpus(tmp_path, name, n, prefix="文档"):
    """Distinct, non-repetitive documents, so the quality filters keep them."""
    path = tmp_path / f"{name}.jsonl"
    write_jsonl(str(path), [
        {"text": f"{prefix}{i}：" + _SENTENCES[i % len(_SENTENCES)] + _SENTENCES[(i + 2) % len(_SENTENCES)]}
        for i in range(n)
    ])
    return str(path)


def _spec(tmp_path, sources, **kwargs):
    payload = {
        "name": "test",
        "tokenizer": "./tokenizer/zh_6400",
        "total_tokens": 10000,
        "sources": sources,
        **kwargs,
    }
    return MixtureSpec.from_dict(payload)


def test_source_requires_exactly_one_of_jsonl_or_hf():
    with pytest.raises(ValueError, match="exactly one"):
        SourceSpec.from_dict({"name": "a", "weight": 1.0})
    with pytest.raises(ValueError, match="exactly one"):
        SourceSpec.from_dict({"name": "a", "weight": 1.0, "jsonl": "x", "hf": {"path": "y"}})


def test_source_rejects_unknown_keys_and_bad_weights():
    with pytest.raises(ValueError, match="Unknown source keys"):
        SourceSpec.from_dict({"name": "a", "weight": 1.0, "jsonl": "x", "weigth": 1})
    with pytest.raises(ValueError, match="weight > 0"):
        SourceSpec.from_dict({"name": "a", "weight": 0.0, "jsonl": "x"})


def test_mixture_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        MixtureSpec.from_dict({
            "name": "t", "total_tokens": 100,
            "sources": [
                {"name": "a", "weight": 0.3, "jsonl": "x"},
                {"name": "b", "weight": 0.3, "jsonl": "y"},
            ],
        })


def test_mixture_needs_a_source():
    with pytest.raises(ValueError, match="at least one source"):
        MixtureSpec.from_dict({"name": "t", "total_tokens": 100, "sources": []})


def test_token_budget_splits_by_weight():
    spec = MixtureSpec.from_dict({
        "name": "t", "total_tokens": 1000,
        "sources": [
            {"name": "a", "weight": 0.7, "jsonl": "x"},
            {"name": "b", "weight": 0.3, "jsonl": "y"},
        ],
    })
    assert spec.token_budget(spec.sources[0]) == 700
    assert spec.token_budget(spec.sources[1]) == 300


def test_shipped_mixture_spec_is_valid():
    """configs/mixture_v1.json must stay loadable as the plan evolves."""
    spec = MixtureSpec.load("configs/mixture_v1.json")
    assert spec.total_tokens == 10_000_000_000
    assert sum(s.weight for s in spec.sources) == pytest.approx(1.0)
    assert {s.name for s in spec.sources} == {
        "zh_web", "en_web", "code", "math", "books", "synthetic_textbook",
    }
    assert all(s.hf is not None for s in spec.sources)


def test_to_record_wraps_a_custom_text_field():
    source = SourceSpec.from_dict({"name": "c", "weight": 1.0, "jsonl": "x", "text_field": "content"})
    assert to_record({"content": "print(1)"}, source) == {"text": "print(1)"}
    assert to_record({"other": "x"}, source) is None


def test_to_record_passes_known_schemas_through():
    """A conversation or preference dataset can be mixed in unchanged."""
    source = SourceSpec.from_dict({"name": "s", "weight": 1.0, "jsonl": "x"})
    record = {"conversations": [{"role": "user", "content": "hi"}]}
    assert to_record(record, source) is record


def test_count_tokens_matches_the_tokenizer(tokenizer):
    texts = ["磨刀石", "hello world"]
    assert count_tokens(tokenizer, texts) == [
        len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts
    ]
    assert count_tokens(tokenizer, []) == []


def test_iter_raw_reads_local_jsonl(tmp_path):
    path = _corpus(tmp_path, "src", 3)
    assert len(list(iter_raw(SourceSpec.from_dict({"name": "s", "weight": 1.0, "jsonl": path})))) == 3


def test_prepare_source_stops_at_its_token_budget(tmp_path, tokenizer):
    path = _corpus(tmp_path, "big", 400)
    spec = _spec(tmp_path, [{"name": "s", "weight": 1.0, "jsonl": path}])
    out = tmp_path / "out.jsonl"

    report = prepare_source(spec, spec.sources[0], tokenizer, str(out))

    assert report.target_tokens == 10000
    assert report.tokens >= 10000
    assert report.documents < 400  # stopped early instead of consuming everything
    assert not report.exhausted
    assert len(list(read_jsonl(str(out)))) == report.documents


def test_keep_rate_is_not_diluted_by_the_tokenization_batch(tmp_path, tokenizer):
    """Records pulled into the batch but never emitted must not count as read,
    or a clean source looks like it was heavily filtered."""
    path = _corpus(tmp_path, "clean", 4000)
    spec = _spec(tmp_path, [{"name": "s", "weight": 1.0, "jsonl": path}])

    report = prepare_source(spec, spec.sources[0], tokenizer, None)

    assert not report.rejected and not report.exact_duplicates
    assert report.seen == report.documents
    assert report.keep_rate == pytest.approx(1.0)


def test_prepare_source_flags_a_source_that_runs_dry(tmp_path, tokenizer):
    """Silently under-filling a slot would quietly change the mixture."""
    path = _corpus(tmp_path, "small", 5)
    spec = _spec(tmp_path, [{"name": "s", "weight": 1.0, "jsonl": path}], total_tokens=10_000_000)

    report = prepare_source(spec, spec.sources[0], tokenizer, None)

    assert report.exhausted
    assert report.fill < 0.01
    assert "ran out of data" in render_sources([report])


def test_prepare_source_drops_exact_duplicates(tmp_path, tokenizer):
    path = tmp_path / "dup.jsonl"
    write_jsonl(str(path), [{"text": "同一篇文档的内容重复出现了很多次。"}] * 10)
    spec = _spec(tmp_path, [{"name": "s", "weight": 1.0, "jsonl": str(path)}], total_tokens=10_000_000)

    report = prepare_source(spec, spec.sources[0], tokenizer, None)

    assert report.documents == 1
    assert report.exact_duplicates == 9


def test_prepare_source_attributes_filter_rejections(tmp_path, tokenizer):
    path = tmp_path / "mixed.jsonl"
    write_jsonl(str(path), [
        {"text": "这是一段足够长的正常中文文本，用来通过长度过滤。"},
        {"text": "短"},
        {"text": "hello world in english only"},
        {"other_field": "no text at all"},
    ])
    spec = _spec(
        tmp_path,
        [{
            "name": "s", "weight": 1.0, "jsonl": str(path),
            "filters": {"min_chars": 10, "min_cjk_ratio": 0.5},
        }],
        total_tokens=10_000_000,
    )

    report = prepare_source(spec, spec.sources[0], tokenizer, None)

    assert report.documents == 1
    assert report.rejected["too_short"] == 1
    assert report.rejected["low_cjk_ratio"] == 1
    assert report.rejected["missing_text_field"] == 1
    assert report.seen == 4
    assert report.keep_rate == pytest.approx(0.25)


def test_prepare_source_respects_max_records(tmp_path, tokenizer):
    path = _corpus(tmp_path, "capped", 100)
    spec = _spec(
        tmp_path,
        [{"name": "s", "weight": 1.0, "jsonl": path, "max_records": 7}],
        total_tokens=10_000_000,
    )
    assert prepare_source(spec, spec.sources[0], tokenizer, None).documents == 7


def test_finalize_splits_and_decontaminates(tmp_path, tokenizer):
    leaked = "一个水池有两个进水管，甲管单独注满需要六小时，乙管单独注满需要四小时。"
    source = tmp_path / "sources" / "s.jsonl"
    write_jsonl(str(source), [
        *({"text": f"干净的训练文档编号 {i}，内容是关于园艺和土壤的介绍。"} for i in range(200)),
        {"text": "习题集：" + leaked + " 请计算同时开需要多久。"},
    ])
    spec = _spec(tmp_path, [{"name": "s", "weight": 1.0, "jsonl": str(source)}],
                 val_fraction=0.1, holdout_fraction=0.1)

    info = finalize(spec, [str(source)], str(tmp_path), build_eval_index([leaked]))

    assert info["contaminated_removed"] == 1
    assert info["eval_items"] == 1
    assert sum(info["counts"].values()) == 200
    for name in (TRAIN, VAL, HOLDOUT):
        assert (tmp_path / f"{name}.jsonl").exists()
    assert info["counts"][TRAIN] > info["counts"][VAL]


def test_finalize_without_an_eval_index_keeps_everything(tmp_path):
    source = tmp_path / "s.jsonl"
    write_jsonl(str(source), [{"text": f"文档 {i}"} for i in range(20)])
    spec = _spec(tmp_path, [{"name": "s", "weight": 1.0, "jsonl": str(source)}])

    info = finalize(spec, [str(source)], str(tmp_path), None)

    assert sum(info["counts"].values()) == 20
    assert info["contaminated_removed"] == 0


def test_source_report_serializes(tmp_path, tokenizer):
    path = _corpus(tmp_path, "s", 20)
    spec = _spec(tmp_path, [{"name": "s", "weight": 1.0, "jsonl": path}])
    report = prepare_source(spec, spec.sources[0], tokenizer, None)

    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["name"] == "s"
    assert payload["tokens"] > 0
    assert isinstance(payload["rejected"], dict)

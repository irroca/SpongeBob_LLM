import json
import random

import pytest

from envs import ArithmeticEnv, load_tasks, make_env
from envs.arithmetic import (
    extract_tag,
    gold_completion,
    has_valid_format,
    normalize_number,
    parse_answer,
)
from envs.generate_data import build_rows, write_jsonl


def _completion(think: str, answer: str) -> str:
    return f"<think>{think}</think><answer>{answer}</answer>"


def test_extract_tag_requires_closing_tag():
    assert extract_tag("<answer>42</answer>", "answer") == "42"
    assert extract_tag("<answer>42", "answer") is None
    assert extract_tag("no tags here", "answer") is None


def test_valid_format_requires_think_before_answer():
    assert has_valid_format(_completion("1+1", "2"))
    assert not has_valid_format("<answer>2</answer><think>1+1</think>")
    assert not has_valid_format("<think>1+1</think><answer>2")
    assert not has_valid_format("<answer>2</answer>")
    # An answer nested inside the think block is not a valid two-section output.
    assert not has_valid_format("<think>maybe <answer>2</answer></think>")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("42", "42"),
        (" 42 ", "42"),
        ("007", "7"),
        ("-15", "-15"),
        ("1,024", "1024"),
        ("４２", "42"),  # fullwidth digits
        ("42。", "42"),
        ("forty two", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_parse_answer_strict_vs_fallback():
    text = "<think>17 + 8 = 25</think>答案 25"  # answer tag missing entirely
    assert parse_answer(text, strict=True) is None
    assert parse_answer(text, strict=False) == "25"
    # With a tag present both modes agree, and the tag wins over stray numbers.
    tagged = "<think>3 + 4 = 7</think><answer>7</answer> 999"
    assert parse_answer(tagged, strict=True) == "7"
    assert parse_answer(tagged, strict=False) == "7"


def test_reward_perfect_completion():
    env = ArithmeticEnv(seed=0, format_weight=0.2)
    task = env.sample_task()
    reward = env.score(env.gold_completion(task), task)
    assert reward.correct and reward.format_ok
    assert reward.total == pytest.approx(1.2)
    assert not reward.hacked_format


def test_reward_format_only_is_detected_as_hacking():
    """Right shape, wrong number: the model collected the format reward only."""
    env = ArithmeticEnv(seed=0, format_weight=0.2)
    task = env.sample_task()
    wrong = _completion("随便写点", str(int(task.answer) + 1))
    reward = env.score(wrong, task)

    assert reward.format_ok and not reward.correct
    assert reward.hacked_format
    assert reward.total == pytest.approx(0.2)


def test_strict_mode_gates_accuracy_on_the_answer_tag():
    task = ArithmeticEnv(seed=3).sample_task()
    bare = f"答案是 {task.answer}"

    strict = ArithmeticEnv(seed=3, strict=True).score(bare, task)
    lenient = ArithmeticEnv(seed=3, strict=False).score(bare, task)

    assert strict.total == 0.0 and not strict.correct
    assert lenient.correct and not lenient.format_ok
    assert lenient.total == pytest.approx(1.0)


def test_sampling_is_seed_reproducible_and_respects_bounds():
    a = ArithmeticEnv(seed=7, ops=("+", "-"), min_digits=1, max_digits=2).sample(20)
    b = ArithmeticEnv(seed=7, ops=("+", "-"), min_digits=1, max_digits=2).sample(20)
    assert [t.question for t in a] == [t.question for t in b]
    assert all(t.meta["a"] <= 99 and t.meta["b"] <= 99 for t in a)


def test_subtraction_stays_non_negative_by_default():
    env = ArithmeticEnv(seed=1, ops=("-",), max_digits=2)
    assert all(int(t.answer) >= 0 for t in env.sample(50))


def test_negative_results_allowed_when_configured():
    env = ArithmeticEnv(seed=1, ops=("-",), max_digits=2, allow_negative=True)
    assert any(int(t.answer) < 0 for t in env.sample(50))


def test_gold_completion_always_scores_full_reward():
    """The cold-start data the env generates must be worth full reward under its own rules."""
    env = ArithmeticEnv(seed=11, max_digits=3)
    for task in env.sample(40):
        reward = env.score(env.gold_completion(task), task)
        assert reward.correct, (task.question, env.gold_completion(task))
        assert reward.format_ok


def test_corrupt_completion_never_beats_gold():
    env = ArithmeticEnv(seed=5)
    rng = random.Random(5)
    for task in env.sample(40):
        gold = env.score(env.gold_completion(task), task)
        rejected = env.score(env.corrupt_completion(task, rng), task)
        assert rejected.total < gold.total


def test_reject_unsupported_op():
    with pytest.raises(ValueError):
        ArithmeticEnv(ops=("*",))


def test_gold_completion_multi_digit_decomposition():
    text = gold_completion(37, "+", 25)
    assert "37 + 20 = 57" in text
    assert "57 + 5 = 62" in text
    assert "<answer>62</answer>" in text


def test_build_rows_emits_expected_schemas():
    env = ArithmeticEnv(seed=2)
    rng = random.Random(2)

    sft = build_rows(env, "sft", 3, rng)
    pref = build_rows(env, "preference", 3, rng)
    evals = build_rows(env, "eval", 3, rng)

    assert all(r["conversations"][0]["role"] == "user" for r in sft)
    assert all(r["conversations"][1]["role"] == "assistant" for r in sft)
    assert all({"prompt", "chosen", "rejected"} <= set(r) for r in pref)
    assert all({"question", "answer"} <= set(r) for r in evals)


def test_build_rows_rejects_unknown_split():
    with pytest.raises(ValueError):
        build_rows(ArithmeticEnv(seed=0), "rlhf", 1, random.Random(0))


def test_generated_eval_set_roundtrips_through_load_tasks(tmp_path):
    path = tmp_path / "eval.jsonl"
    rows = build_rows(ArithmeticEnv(seed=4), "eval", 5, random.Random(4))
    assert write_jsonl(str(path), rows) == 5

    tasks = load_tasks(str(path))

    assert [t.question for t in tasks] == [r["question"] for r in rows]
    assert tasks[0].meta["op"] in ("+", "-")
    assert json.loads(path.read_text("utf-8").splitlines()[0])["answer"] == rows[0]["answer"]


def test_registry_exposes_arithmetic_env():
    env = make_env("arithmetic", seed=0)
    assert isinstance(env, ArithmeticEnv)
    with pytest.raises(KeyError):
        make_env("does-not-exist")


def test_render_mentions_question_and_required_tags():
    env = ArithmeticEnv(seed=0)
    task = env.sample_task()
    rendered = env.render(task)
    assert task.question in rendered
    assert "<think>" in rendered and "<answer>" in rendered

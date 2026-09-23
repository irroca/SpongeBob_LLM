import math

import pytest
import torch

from config import LLMConfig
from evaluate import evaluate_lm, evaluate_preference
from model import Whetstone

VOCAB = 32


def _model(seed=0):
    torch.manual_seed(seed)
    return Whetstone(LLMConfig(
        dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
        vocab_size=VOCAB, max_seq_len=64, dropout=0.0,
    ))


def _lm_batches(n_batches=3, bs=2, seq=8, mask_value=1):
    torch.manual_seed(1)
    return [
        (
            torch.randint(0, VOCAB, (bs, seq)),
            torch.randint(0, VOCAB, (bs, seq)),
            torch.full((bs, seq), mask_value),
        )
        for _ in range(n_batches)
    ]


def test_evaluate_lm_reports_loss_ppl_and_token_count():
    stats = evaluate_lm(_model(), _lm_batches(), "cpu")

    assert stats["batches"] == 3
    assert stats["tokens"] == 3 * 2 * 8
    assert stats["ppl"] == pytest.approx(math.exp(stats["loss"]), rel=1e-6)
    # An untrained model over a uniform vocabulary sits near ln(V).
    assert stats["loss"] == pytest.approx(math.log(VOCAB), abs=0.6)


def test_evaluate_lm_is_token_weighted_not_batch_averaged():
    """Batch-averaging would let a batch of two scored tokens count as much as a
    batch of sixteen, so the number would drift with batch composition."""
    model = _model()
    torch.manual_seed(2)
    big = (torch.randint(0, VOCAB, (1, 8)), torch.randint(0, VOCAB, (1, 8)), torch.ones(1, 8, dtype=torch.long))
    small_mask = torch.zeros(1, 8, dtype=torch.long)
    small_mask[0, 0] = 1
    small = (big[0].clone(), big[1].clone(), small_mask)

    combined = evaluate_lm(model, [big, small], "cpu")
    only_big = evaluate_lm(model, [big], "cpu")
    only_small = evaluate_lm(model, [small], "cpu")

    assert combined["tokens"] == 9
    expected = (only_big["loss"] * 8 + only_small["loss"] * 1) / 9
    assert combined["loss"] == pytest.approx(expected, rel=1e-5)


def test_evaluate_lm_respects_the_mask():
    model = _model()
    torch.manual_seed(3)
    scored = evaluate_lm(model, _lm_batches(1, mask_value=1), "cpu")
    ignored = evaluate_lm(model, _lm_batches(1, mask_value=0), "cpu")

    assert scored["tokens"] == 16
    assert ignored["tokens"] == 0
    assert math.isnan(ignored["loss"])


def test_evaluate_lm_caps_batches():
    assert evaluate_lm(_model(), _lm_batches(10), "cpu", max_batches=2)["batches"] == 2


def test_evaluate_lm_restores_training_mode():
    model = _model()
    model.train()
    evaluate_lm(model, _lm_batches(1), "cpu")
    assert model.training


def test_evaluate_lm_does_not_blow_up_on_a_diverged_model():
    """A diverged run must still record the number that proves it diverged."""
    model = _model()
    with torch.no_grad():
        model.output.weight.mul_(1e4)
    stats = evaluate_lm(model, _lm_batches(1), "cpu")
    assert stats["ppl"] == float("inf")
    assert not math.isnan(stats["loss"])


def _preference_batches(n=2, bs=2, seq=8):
    torch.manual_seed(4)
    return [
        tuple(
            torch.randint(0, VOCAB, (bs, seq)) if i % 3 != 2 else torch.ones(bs, seq, dtype=torch.long)
            for i in range(6)
        )
        for _ in range(n)
    ]


def test_preference_accuracy_is_one_half_when_policy_equals_reference():
    """With policy == ref every margin is exactly zero, so nothing is ranked
    above anything: accuracy must be 0, not a coin flip."""
    model = _model()
    stats = evaluate_preference(model, model, _preference_batches(), "cpu", beta=0.1)

    assert stats["margin"] == pytest.approx(0.0, abs=1e-6)
    assert stats["accuracy"] == 0.0
    assert stats["pairs"] == 4


def test_preference_accuracy_detects_a_policy_that_prefers_chosen():
    policy, ref = _model(seed=0), _model(seed=1)
    stats = evaluate_preference(policy, ref, _preference_batches(), "cpu", beta=0.1)

    assert 0.0 <= stats["accuracy"] <= 1.0
    # margin is exactly the reward gap, so the components must reconstruct it.
    assert stats["margin"] == pytest.approx(stats["chosen_reward"] - stats["rejected_reward"], rel=1e-5)


def test_preference_margin_scales_with_beta():
    policy, ref = _model(seed=0), _model(seed=1)
    small = evaluate_preference(policy, ref, _preference_batches(), "cpu", beta=0.1)
    large = evaluate_preference(policy, ref, _preference_batches(), "cpu", beta=0.5)

    assert large["margin"] == pytest.approx(small["margin"] * 5, rel=1e-4)
    assert large["accuracy"] == small["accuracy"]  # scaling cannot reorder pairs


def test_preference_handles_an_empty_loader():
    model = _model()
    stats = evaluate_preference(model, model, [], "cpu", beta=0.1)
    assert stats["pairs"] == 0 and math.isnan(stats["accuracy"])

import math

import pytest
import torch

from losses import (
    approx_kl,
    grpo_advantages,
    grpo_policy_loss,
    token_entropy,
    token_logprobs,
    zero_variance_groups,
)


def test_token_logprobs_matches_manual_gather():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    targets = torch.randint(0, 7, (2, 3))

    result = token_logprobs(logits, targets)

    expected = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(result, expected, atol=1e-6)


def test_token_entropy_of_uniform_logits_is_log_vocab():
    logits = torch.zeros(1, 2, 8)
    assert torch.allclose(token_entropy(logits), torch.full((1, 2), math.log(8)), atol=1e-6)


def test_advantages_are_zero_mean_within_each_group():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.2, 1.2, 1.2, 0.2]])
    adv = grpo_advantages(rewards)
    assert torch.allclose(adv.mean(dim=-1), torch.zeros(2), atol=1e-5)
    # Higher-reward rollouts get positive advantage, lower ones negative.
    assert (adv[0][rewards[0] == 1.0] > 0).all()
    assert (adv[0][rewards[0] == 0.0] < 0).all()


def test_std_normalization_rescales_but_keeps_sign():
    rewards = torch.tensor([[1.0, 0.0], [10.0, 0.0]])
    z = grpo_advantages(rewards, normalize_std=True)
    raw = grpo_advantages(rewards, normalize_std=False)

    assert torch.sign(z).equal(torch.sign(raw))
    # z-scoring erases the fact that group 1 had a 10x larger reward spread...
    assert torch.allclose(z[0].abs(), z[1].abs(), atol=1e-4)
    # ...while Dr. GRPO's unnormalized advantage preserves it.
    assert raw[1].abs().max() > 5 * raw[0].abs().max()


def test_zero_variance_group_produces_no_gradient_signal():
    rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 0.5]])

    silent = zero_variance_groups(rewards)
    adv = grpo_advantages(rewards)

    assert silent.tolist() == [True, False]
    assert torch.allclose(adv[0], torch.zeros(3))
    assert adv[1].abs().sum() > 0


def test_advantages_reject_non_grouped_input():
    with pytest.raises(ValueError):
        grpo_advantages(torch.tensor([1.0, 0.0]))


def _flat_inputs(n=2, t=3):
    logprobs = torch.full((n, t), -1.0)
    return logprobs, logprobs.clone(), torch.ones(n, t)


def test_on_policy_loss_equals_negative_mean_advantage():
    """First pass over fresh rollouts: ratio == 1, so the clip never binds."""
    logprobs, old, mask = _flat_inputs()
    advantages = torch.tensor([1.0, -1.0])

    loss, metrics = grpo_policy_loss(logprobs, old, advantages, mask)

    assert loss == pytest.approx(0.0, abs=1e-6)
    assert metrics["ratio_mean"] == pytest.approx(1.0)
    assert metrics["clip_frac"] == pytest.approx(0.0)


def test_positive_advantage_gradient_raises_token_logprob():
    logprobs = torch.full((1, 3), -1.0, requires_grad=True)
    old = torch.full((1, 3), -1.0)

    loss, _ = grpo_policy_loss(logprobs, old, torch.tensor([1.0]), torch.ones(1, 3))
    loss.backward()

    # Descending the loss increases the log-prob of a better-than-average rollout.
    assert (logprobs.grad < 0).all()


def test_clipping_caps_the_update_for_large_ratios():
    old = torch.zeros(1, 2)
    logprobs = torch.full((1, 2), 1.0)  # ratio = e ~= 2.72, far above 1 + eps
    advantages = torch.tensor([1.0])

    loss, metrics = grpo_policy_loss(logprobs, old, advantages, torch.ones(1, 2), clip_eps_high=0.2)

    assert loss == pytest.approx(-1.2, abs=1e-5)
    assert metrics["clip_frac"] == pytest.approx(1.0)


def test_clip_higher_allows_more_upside_than_symmetric_clipping():
    old = torch.zeros(1, 2)
    logprobs = torch.full((1, 2), 1.0)
    advantages = torch.tensor([1.0])

    symmetric, _ = grpo_policy_loss(logprobs, old, advantages, torch.ones(1, 2), clip_eps_high=0.2)
    higher, _ = grpo_policy_loss(logprobs, old, advantages, torch.ones(1, 2), clip_eps_high=0.4)

    assert higher < symmetric


def test_masked_tokens_are_excluded_from_the_loss():
    logprobs = torch.tensor([[0.0, 5.0]])
    old = torch.zeros(1, 2)
    advantages = torch.tensor([1.0])

    masked, _ = grpo_policy_loss(logprobs, old, advantages, torch.tensor([[1.0, 0.0]]))
    only_first, _ = grpo_policy_loss(
        logprobs[:, :1], old[:, :1], advantages, torch.ones(1, 1)
    )

    assert masked == pytest.approx(float(only_first))


def test_aggregation_modes_use_different_denominators():
    """Two rollouts of very different length: seq_mean equalizes them, token_mean
    lets the long one dominate, dr_grpo divides by a constant budget."""
    logprobs = torch.zeros(2, 4)
    old = torch.zeros(2, 4)
    advantages = torch.tensor([1.0, 1.0])
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

    seq, _ = grpo_policy_loss(logprobs, old, advantages, mask, aggregation="seq_mean")
    token, _ = grpo_policy_loss(logprobs, old, advantages, mask, aggregation="token_mean")
    dr, _ = grpo_policy_loss(
        logprobs, old, advantages, mask, aggregation="dr_grpo", max_completion_len=4
    )

    assert seq == pytest.approx(-1.0)  # (1/2) * (1/1 + 4/4)
    assert token == pytest.approx(-1.0)  # 5 tokens / 5 tokens
    assert dr == pytest.approx(-5.0 / 8.0)  # 5 tokens / (2 * 4)


def test_dr_grpo_requires_a_length_budget():
    with pytest.raises(ValueError):
        grpo_policy_loss(
            torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([1.0]), torch.ones(1, 2),
            aggregation="dr_grpo",
        )


def test_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError):
        grpo_policy_loss(
            torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([1.0]), torch.ones(1, 2),
            aggregation="mean",
        )


def test_advantage_batch_shape_is_validated():
    with pytest.raises(ValueError):
        grpo_policy_loss(torch.zeros(2, 2), torch.zeros(2, 2), torch.tensor([1.0]), torch.ones(2, 2))


@pytest.mark.parametrize("aggregation", ["seq_mean", "token_mean", "dr_grpo"])
def test_micro_batched_loss_matches_single_pass(aggregation):
    """Micro-batching only stays correct if every chunk divides by the *batch*
    normalizer; this is what lets a rollout batch be split on a small machine."""
    torch.manual_seed(0)
    n, t = 6, 5
    logprobs = torch.randn(n, t)
    old = torch.randn(n, t)
    advantages = torch.randn(n)
    mask = torch.ones(n, t)
    mask[0, 3:] = 0
    mask[4, 1:] = 0

    kwargs = dict(aggregation=aggregation, max_completion_len=t)
    full, _ = grpo_policy_loss(logprobs, old, advantages, mask, **kwargs)

    normalizer = {
        "seq_mean": float(n),
        "token_mean": float(mask.sum()),
        "dr_grpo": float(n * t),
    }[aggregation]
    chunked = sum(
        grpo_policy_loss(
            logprobs[s : s + 2], old[s : s + 2], advantages[s : s + 2], mask[s : s + 2],
            normalizer=normalizer, **kwargs,
        )[0]
        for s in range(0, n, 2)
    )

    assert float(chunked) == pytest.approx(float(full), abs=1e-6)


def test_approx_kl_is_zero_for_identical_policies():
    logprobs = torch.tensor([[-1.0, -2.0]])
    assert approx_kl(logprobs, logprobs.clone(), torch.ones(1, 2)) == pytest.approx(0.0, abs=1e-7)


def test_approx_kl_is_non_negative_and_grows_with_divergence():
    torch.manual_seed(0)
    logprobs = torch.log_softmax(torch.randn(4, 6), dim=-1)
    near = logprobs + 0.05 * torch.randn_like(logprobs)
    far = logprobs + 0.5 * torch.randn_like(logprobs)
    mask = torch.ones(4, 6)

    small = approx_kl(logprobs, near, mask)
    large = approx_kl(logprobs, far, mask)

    assert small >= 0 and large >= 0
    assert large > small


def test_approx_kl_respects_mask():
    logprobs = torch.tensor([[0.0, 0.0]])
    ref = torch.tensor([[0.0, 3.0]])

    masked = approx_kl(logprobs, ref, torch.tensor([[1.0, 0.0]]))
    unmasked = approx_kl(logprobs, ref, torch.ones(1, 2))

    assert masked == pytest.approx(0.0, abs=1e-7)
    assert unmasked > 0

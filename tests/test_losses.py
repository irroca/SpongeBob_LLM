import torch
import torch.nn.functional as F

from losses import dpo_loss, kd_loss, masked_cross_entropy, sequence_logprobs


def test_masked_cross_entropy_ignores_zeros():
    logits = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (2, 4))
    mask = torch.zeros(2, 4)
    mask[:, :2] = 1
    loss = masked_cross_entropy(logits, targets, mask)
    assert torch.isfinite(loss)
    # Compare to manual
    ce = F.cross_entropy(logits.view(-1, 10), targets.view(-1), reduction="none").view(2, 4)
    expected = (ce * mask).sum() / mask.sum()
    assert torch.allclose(loss, expected)


def test_kd_loss_finite_and_temperature_changes():
    torch.manual_seed(0)
    student = torch.randn(2, 5, 20)
    teacher = torch.randn(2, 5, 20)
    mask = torch.ones(2, 5)
    loss_t2 = kd_loss(student, teacher, temperature=2.0, mask=mask)
    loss_t4 = kd_loss(student, teacher, temperature=4.0, mask=mask)
    assert torch.isfinite(loss_t2) and torch.isfinite(loss_t4)
    assert not torch.allclose(loss_t2, loss_t4)


def test_kd_alpha_blend_moves_with_alpha():
    torch.manual_seed(0)
    student = torch.randn(2, 5, 20, requires_grad=True)
    teacher = torch.randn(2, 5, 20)
    targets = torch.randint(0, 20, (2, 5))
    mask = torch.ones(2, 5)
    ce = masked_cross_entropy(student, targets, mask)
    kd = kd_loss(student, teacher.detach(), temperature=2.0, mask=mask)
    loss0 = (1 - 0.0) * ce + 0.0 * kd
    loss1 = (1 - 1.0) * ce + 1.0 * kd
    assert torch.allclose(loss0, ce)
    assert torch.allclose(loss1, kd)


def test_sequence_logprobs_matches_manual_token_sum():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 6)
    targets = torch.randint(0, 6, (2, 4))
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])

    result = sequence_logprobs(logits, targets, mask)

    log_probs = F.log_softmax(logits, dim=-1)
    expected = torch.zeros(2)
    for b in range(2):
        for t in range(4):
            if mask[b, t] == 1:
                expected[b] += log_probs[b, t, targets[b, t]]

    assert result.shape == (2,)
    assert torch.allclose(result, expected, atol=1e-6)


def test_sequence_logprobs_masked_positions_ignored():
    torch.manual_seed(1)
    logits = torch.randn(1, 3, 5)
    targets = torch.randint(0, 5, (1, 3))
    full_mask = torch.ones(1, 3)
    partial_mask = torch.tensor([[1.0, 1.0, 0.0]])

    full_result = sequence_logprobs(logits, targets, full_mask)
    partial_result = sequence_logprobs(logits, targets, partial_mask)

    log_probs = F.log_softmax(logits, dim=-1)
    last_token_logp = log_probs[0, 2, targets[0, 2]]

    assert torch.allclose(full_result, partial_result + last_token_logp, atol=1e-6)
    assert not torch.allclose(full_result, partial_result)


def test_dpo_loss_prefers_higher_chosen_advantage():
    # Construct logps where chosen advantage is larger in case A than B
    beta = 0.1
    # case A: strong preference margin
    loss_a = dpo_loss(
        policy_chosen_logps=torch.tensor([ -1.0]),
        policy_rejected_logps=torch.tensor([-5.0]),
        ref_chosen_logps=torch.tensor([-2.0]),
        ref_rejected_logps=torch.tensor([-2.0]),
        beta=beta,
    )
    # case B: weaker / inverted margin
    loss_b = dpo_loss(
        policy_chosen_logps=torch.tensor([ -4.0]),
        policy_rejected_logps=torch.tensor([-1.0]),
        ref_chosen_logps=torch.tensor([-2.0]),
        ref_rejected_logps=torch.tensor([-2.0]),
        beta=beta,
    )
    assert loss_a < loss_b
    assert torch.isfinite(loss_a) and torch.isfinite(loss_b)

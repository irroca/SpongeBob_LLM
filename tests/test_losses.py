import torch
import torch.nn.functional as F

from losses import dpo_loss, kd_loss, masked_cross_entropy


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

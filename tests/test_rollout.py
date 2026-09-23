import pytest
import torch
from transformers import AutoTokenizer

from Config import LLMConfig
from envs import ArithmeticEnv
from envs.base import Reward, Task
from losses import token_logprobs
from model import SpongeBob
from rollout import (
    build_prompt_ids,
    collate_rollouts,
    completion_masks,
    compute_logprobs,
    decode_examples,
    generate_group,
    resolve_micro_batch,
    rollout_stats,
    run_rollouts,
    select_rollouts,
    Rollout,
)

EOS, PAD = 2, 0


def _tokenizer():
    return AutoTokenizer.from_pretrained("./spongebob_tokenizer")


def _tiny_model(vocab_size=10, **kwargs):
    defaults = dict(
        dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64,
        vocab_size=vocab_size, dropout=0.0,
    )
    defaults.update(kwargs)
    return SpongeBob(LLMConfig(**defaults)).eval()


def _rollout(prompt, completion, mask, rewards=None):
    completion = torch.tensor(completion, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    n = completion.size(0)
    return Rollout(
        task=Task(question="1 + 1", answer="2"),
        prompt_ids=torch.tensor(prompt, dtype=torch.long),
        completions=completion,
        completion_mask=mask,
        texts=["x"] * n,
        rewards=rewards or [Reward(1.0, 1.0, 0.0, True, True, "2")] * n,
        truncated=torch.zeros(n, dtype=torch.bool),
    )


def test_completion_mask_covers_through_first_eos():
    tokens = torch.tensor([[5, 6, EOS, PAD, PAD]])
    mask, truncated = completion_masks(tokens, EOS)
    assert mask.tolist() == [[1, 1, 1, 0, 0]]
    assert truncated.tolist() == [False]


def test_completion_mask_keeps_everything_when_truncated():
    """No EOS means the budget ran out; every sampled token still counts."""
    tokens = torch.tensor([[5, 6, 7]])
    mask, truncated = completion_masks(tokens, EOS)
    assert mask.tolist() == [[1, 1, 1]]
    assert truncated.tolist() == [True]


def test_completion_mask_handles_immediate_eos_and_repeated_eos():
    tokens = torch.tensor([[EOS, PAD, PAD], [4, EOS, EOS]])
    mask, truncated = completion_masks(tokens, EOS)
    assert mask.tolist() == [[1, 0, 0], [1, 1, 0]]
    assert truncated.tolist() == [False, False]


def test_collate_aligns_masked_targets_with_completion_tokens():
    """The loss must land on exactly the sampled tokens, shifted for next-token
    prediction, no matter how long the prompt is."""
    rollouts = [
        _rollout(prompt=[11, 12, 13], completion=[[7, 8, EOS, PAD]], mask=[[1, 1, 1, 0]]),
        _rollout(prompt=[21], completion=[[9, EOS, PAD, PAD]], mask=[[1, 1, 0, 0]]),
    ]

    batch = collate_rollouts(rollouts, pad_token_id=PAD)

    assert batch.n_sequences == 2
    assert batch.targets[0][batch.mask[0].bool()].tolist() == [7, 8, EOS]
    assert batch.targets[1][batch.mask[1].bool()].tolist() == [9, EOS]
    # The last prompt token predicts the first completion token.
    first_scored = batch.mask[0].nonzero()[0].item()
    assert batch.input_ids[0, first_scored].item() == 13
    assert batch.n_tokens == 5


def test_collate_right_pads_to_the_longest_sequence():
    rollouts = [
        _rollout(prompt=[1, 2, 3, 4], completion=[[7, EOS]], mask=[[1, 1]]),
        _rollout(prompt=[1], completion=[[7, EOS]], mask=[[1, 1]]),
    ]

    batch = collate_rollouts(rollouts, pad_token_id=PAD)

    assert batch.input_ids.shape == (2, 5)  # width 6 (4+2) minus the shift
    assert batch.mask[1, -1].item() == 0  # padding is never scored
    assert batch.input_ids[1, -1].item() == PAD


def test_collate_rewards_keep_group_structure():
    rewards = [Reward(1.0, 1.0, 0.0, True, True, "2"), Reward(0.0, 0.0, 0.0, False, False, None)]
    rollouts = [
        _rollout([1], [[7, EOS], [8, EOS]], [[1, 1], [1, 1]], rewards=rewards),
        _rollout([1], [[7, EOS], [8, EOS]], [[1, 1], [1, 1]], rewards=rewards[::-1]),
    ]

    batch = collate_rollouts(rollouts, pad_token_id=PAD)

    assert batch.group_size == 2
    assert batch.rewards.shape == (2, 2)
    assert batch.rewards.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_collate_rejects_mixed_group_sizes():
    rollouts = [
        _rollout([1], [[7, EOS]], [[1, 1]]),
        _rollout([1], [[7, EOS], [8, EOS]], [[1, 1], [1, 1]]),
    ]
    with pytest.raises(ValueError):
        collate_rollouts(rollouts, pad_token_id=PAD)


def test_collate_requires_at_least_one_rollout():
    with pytest.raises(ValueError):
        collate_rollouts([], pad_token_id=PAD)


def test_compute_logprobs_micro_batching_matches_single_pass():
    torch.manual_seed(0)
    model = _tiny_model()
    input_ids = torch.randint(0, 10, (6, 8))
    targets = torch.randint(0, 10, (6, 8))

    chunked = compute_logprobs(model, input_ids, targets, micro_batch_size=2)
    with torch.no_grad():
        full = token_logprobs(model(input_ids).logits, targets)

    assert chunked.shape == (6, 8)
    assert torch.allclose(chunked, full, atol=1e-5)


def test_compute_logprobs_restores_training_mode():
    model = _tiny_model()
    model.train()
    compute_logprobs(model, torch.randint(0, 10, (2, 4)), torch.randint(0, 10, (2, 4)), 2)
    assert model.training


def test_generate_group_tiles_one_prompt_into_a_group():
    torch.manual_seed(0)
    model = _tiny_model()
    prompt = torch.tensor([1, 2, 3])

    completions = generate_group(
        model, prompt, group_size=4, max_new_tokens=5,
        temperature=1.0, top_p=1.0, eos_token_id=-1, pad_token_id=PAD,
    )

    assert completions.shape == (4, 5)  # eos never fires, so the full budget is used


def test_generated_ids_can_feed_a_training_forward():
    """generate() runs under inference_mode; unless the ids are cloned out they
    cannot be used as embedding indices in a graph that needs backward."""
    torch.manual_seed(0)
    model = _tiny_model()
    completions = generate_group(
        model, torch.tensor([1, 2]), group_size=2, max_new_tokens=3,
        temperature=1.0, top_p=1.0, eos_token_id=-1, pad_token_id=PAD,
    )

    logits = model(completions).logits
    logits.sum().backward()

    assert model.tok_embeddings.weight.grad is not None


def test_generate_group_rejects_empty_budget():
    with pytest.raises(ValueError):
        generate_group(
            _tiny_model(), torch.tensor([1]), group_size=2, max_new_tokens=0,
            temperature=1.0, top_p=1.0, eos_token_id=EOS, pad_token_id=PAD,
        )


def test_build_prompt_ids_ends_with_the_assistant_turn():
    tokenizer = _tokenizer()
    env = ArithmeticEnv(seed=0)
    task = env.sample_task()

    ids = build_prompt_ids(tokenizer, env, task)
    text = tokenizer.decode(ids.tolist())

    assert task.question in text
    assert env.system_prompt in text
    assert text.rstrip().endswith("<s>assistant")  # generation prompt, nothing sampled yet


def test_run_rollouts_scores_decoded_text_with_the_env():
    """End-to-end reward path: sampled ids -> decoded text -> env reward."""
    tokenizer = _tokenizer()
    env = ArithmeticEnv(seed=0)
    task = env.sample_task()
    gold_ids = tokenizer(env.gold_completion(task), add_special_tokens=False).input_ids
    row = gold_ids + [EOS, PAD, PAD]

    class StubPolicy:
        training = False

        def eval(self):
            return self

        def train(self):
            return self

        def generate(self, input_ids, **kwargs):
            return torch.tensor([row] * input_ids.size(0), dtype=torch.long)

    rollouts = run_rollouts(
        StubPolicy(), tokenizer, env, [task],
        group_size=3, max_new_tokens=len(row),
    )

    assert len(rollouts) == 1 and rollouts[0].group_size == 3
    assert all(r.correct and r.format_ok for r in rollouts[0].rewards)
    assert rollouts[0].completion_mask[0].sum().item() == len(gold_ids) + 1  # through EOS
    assert not rollouts[0].truncated.any()


def test_rollout_stats_reports_hacking_and_silent_groups():
    hacked = Reward(0.2, 0.0, 0.2, True, False, "9")
    correct = Reward(1.2, 1.0, 0.2, True, True, "2")
    rollouts = [
        _rollout([1], [[7, EOS], [8, EOS]], [[1, 1], [1, 1]], rewards=[correct, hacked]),
        _rollout([1], [[7, EOS], [8, EOS]], [[1, 1], [1, 1]], rewards=[hacked, hacked]),
    ]

    stats = rollout_stats(rollouts)

    assert stats["accuracy"] == pytest.approx(0.25)
    assert stats["format_rate"] == pytest.approx(1.0)
    assert stats["hack_rate"] == pytest.approx(0.75)
    assert stats["silent_group_frac"] == pytest.approx(0.5)  # second group is all-hacked
    assert stats["completion_len"] == pytest.approx(2.0)
    assert stats["truncated_frac"] == pytest.approx(0.0)


def test_select_rollouts_drops_flagged_groups():
    rollouts = [_rollout([1], [[7, EOS]], [[1, 1]]) for _ in range(3)]
    kept = select_rollouts(rollouts, torch.tensor([True, False, True]))
    assert len(kept) == 2


def test_decode_examples_respects_limit():
    rollouts = [_rollout([1], [[7, EOS], [8, EOS]], [[1, 1], [1, 1]])]
    assert len(decode_examples(rollouts, limit=1)) == 1
    assert len(decode_examples(rollouts, limit=5)) == 2


@pytest.mark.parametrize("requested,expected", [(0, 8), (-1, 8), (3, 3), (100, 8)])
def test_resolve_micro_batch(requested, expected):
    assert resolve_micro_batch(8, requested) == expected

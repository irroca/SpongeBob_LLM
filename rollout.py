"""On-policy rollout plumbing for GRPO.

GRPO needs, per prompt, a *group* of sampled completions plus the log-probs the
policy assigned to them. Three things make this fiddly and they all live here:

1. **No padded prompt batching.** The model applies RoPE from ``start_pos`` and
   has no left-padding offset, so prompts of different lengths cannot share a
   generate call. A group is the same prompt repeated ``group_size`` times, so
   every row is naturally the same length and no padding is needed.
2. **Completion masks.** ``generate`` emits the real EOS token and then pads,
   per row. Training must credit tokens up to and including that EOS and
   nothing after it.
3. **Log-probs are recomputed, not harvested from sampling.** Temperature and
   top-p reshape the sampling distribution; the importance ratio needs the
   unmodified policy distribution over the tokens that were actually emitted.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from envs.base import Reward, Task, TaskEnv
from losses import token_logprobs


@dataclass
class Rollout:
    """One prompt and the group of completions sampled for it."""

    task: Task
    prompt_ids: torch.Tensor  # (P,)
    completions: torch.Tensor  # (G, T)
    completion_mask: torch.Tensor  # (G, T), 1 through the first EOS inclusive
    texts: list[str]
    rewards: list[Reward]
    truncated: torch.Tensor  # (G,) bool, True when no EOS was emitted

    @property
    def group_size(self) -> int:
        return self.completions.size(0)

    @property
    def reward_tensor(self) -> torch.Tensor:
        return torch.tensor([r.total for r in self.rewards], dtype=torch.float32)


@dataclass
class RolloutBatch:
    """Flat, right-padded training view of several rollout groups.

    Rows are group-major: group ``g`` owns rows ``[g*group_size, (g+1)*group_size)``.
    """

    input_ids: torch.Tensor  # (N, L-1)
    targets: torch.Tensor  # (N, L-1)
    mask: torch.Tensor  # (N, L-1), completion tokens only
    rewards: torch.Tensor  # (n_groups, group_size)
    group_size: int
    rollouts: list[Rollout] = field(default_factory=list)

    @property
    def n_sequences(self) -> int:
        return self.input_ids.size(0)

    @property
    def n_tokens(self) -> float:
        return float(self.mask.sum().item())


def build_prompt_ids(tokenizer, env: TaskEnv, task: Task, device: str = "cpu") -> torch.Tensor:
    """Render a task through the chat template and tokenize it. Returns (P,)."""
    messages = []
    if env.system_prompt:
        messages.append({"role": "system", "content": env.system_prompt})
    messages.append({"role": "user", "content": env.render(task)})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return torch.tensor(ids, dtype=torch.long, device=device)


def completion_masks(tokens: torch.Tensor, eos_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask completion tokens up to and including the first EOS of each row.

    Returns ``(mask, truncated)`` where ``truncated`` marks rows that never
    emitted EOS (they hit the ``max_new_tokens`` budget instead).
    """
    is_eos = tokens == eos_token_id
    eos_before = is_eos.long().cumsum(dim=-1) - is_eos.long()
    mask = (eos_before == 0).long()
    return mask, ~is_eos.any(dim=-1)


@torch.no_grad()
def generate_group(
    model,
    prompt_ids: torch.Tensor,
    group_size: int,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Sample ``group_size`` completions for one prompt. Returns (G, T) new tokens only."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    batched = prompt_ids.unsqueeze(0).expand(group_size, -1).contiguous()
    generated = model.generate(
        batched,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
        repetition_penalty=1.0,
        use_cache=True,
        pad_token_id=pad_token_id,
    )
    # generate() runs under inference_mode; clone so the ids can feed a training
    # forward pass (embedding backward saves its index tensor).
    return generated.clone()


def run_rollouts(
    model,
    tokenizer,
    env: TaskEnv,
    tasks: Sequence[Task],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
) -> list[Rollout]:
    """Sample a group of completions per task and score them with the env."""
    was_training = model.training
    model.eval()
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    rollouts = []
    try:
        for task in tasks:
            prompt_ids = build_prompt_ids(tokenizer, env, task, device)
            completions = generate_group(
                model,
                prompt_ids,
                group_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
            mask, truncated = completion_masks(completions, eos_id)
            texts = [
                tokenizer.decode(row[m.bool()].tolist(), skip_special_tokens=True)
                for row, m in zip(completions, mask)
            ]
            rollouts.append(
                Rollout(
                    task=task,
                    prompt_ids=prompt_ids,
                    completions=completions,
                    completion_mask=mask,
                    texts=texts,
                    rewards=[env.score(text, task) for text in texts],
                    truncated=truncated,
                )
            )
    finally:
        if was_training:
            model.train()
    return rollouts


def collate_rollouts(rollouts: Sequence[Rollout], pad_token_id: int) -> RolloutBatch:
    """Right-pad every prompt+completion into one training batch.

    Padding sits strictly after the completion, and the loss mask covers
    completion tokens only, so trailing pads are inert under a causal model —
    no attention mask is needed for the training forward pass.
    """
    if not rollouts:
        raise ValueError("collate_rollouts requires at least one rollout")
    group_sizes = {r.group_size for r in rollouts}
    if len(group_sizes) != 1:
        raise ValueError(f"all rollouts must share a group size, got {sorted(group_sizes)}")

    device = rollouts[0].completions.device
    rows = [
        (r.prompt_ids, r.completions[i], r.completion_mask[i])
        for r in rollouts
        for i in range(r.group_size)
    ]
    width = max(prompt.size(0) + comp.size(0) for prompt, comp, _ in rows)

    sequences = torch.full((len(rows), width), pad_token_id, dtype=torch.long, device=device)
    full_mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for i, (prompt, comp, comp_mask) in enumerate(rows):
        p, t = prompt.size(0), comp.size(0)
        sequences[i, :p] = prompt
        sequences[i, p : p + t] = comp
        full_mask[i, p : p + t] = comp_mask

    return RolloutBatch(
        input_ids=sequences[:, :-1],
        targets=sequences[:, 1:],
        mask=full_mask[:, 1:],
        rewards=torch.stack([r.reward_tensor for r in rollouts]).to(device),
        group_size=rollouts[0].group_size,
        rollouts=list(rollouts),
    )


def compute_logprobs(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    micro_batch_size: int,
    ctx=None,
) -> torch.Tensor:
    """Frozen-graph token log-probs for a whole rollout batch, in micro-batches."""
    ctx = ctx if ctx is not None else nullcontext()
    was_training = model.training
    model.eval()
    chunks = []
    try:
        with torch.no_grad():
            for start in range(0, input_ids.size(0), micro_batch_size):
                stop = start + micro_batch_size
                with ctx:
                    logits = model(input_ids[start:stop]).logits
                chunks.append(token_logprobs(logits.float(), targets[start:stop]))
    finally:
        if was_training:
            model.train()
    return torch.cat(chunks, dim=0)


def rollout_stats(rollouts: Sequence[Rollout], eps: float = 1e-8) -> dict:
    """Reward/accuracy/format/length diagnostics for one rollout batch.

    ``silent_group_frac`` is the share of groups whose rewards are all equal:
    their advantages are exactly zero, so they produce no gradient at all.
    """
    rewards = [r for roll in rollouts for r in roll.rewards]
    n = max(len(rewards), 1)
    lengths = torch.cat([roll.completion_mask.sum(dim=-1).float() for roll in rollouts])
    truncated = torch.cat([roll.truncated for roll in rollouts])
    group_rewards = torch.stack([roll.reward_tensor for roll in rollouts])
    spread = group_rewards.max(dim=-1).values - group_rewards.min(dim=-1).values
    return {
        "reward_mean": sum(r.total for r in rewards) / n,
        "reward_std": float(group_rewards.std(dim=-1, unbiased=False).mean()),
        "accuracy": sum(float(r.correct) for r in rewards) / n,
        "format_rate": sum(float(r.format_ok) for r in rewards) / n,
        "hack_rate": sum(float(r.hacked_format) for r in rewards) / n,
        "completion_len": float(lengths.mean()),
        "truncated_frac": float(truncated.float().mean()),
        "silent_group_frac": float((spread.abs() <= eps).float().mean()),
    }


def select_rollouts(
    rollouts: Sequence[Rollout],
    keep: torch.Tensor,
) -> list[Rollout]:
    """Keep only the groups flagged ``True`` in ``keep`` (DAPO dynamic sampling)."""
    return [roll for roll, flag in zip(rollouts, keep.tolist()) if flag]


def decode_examples(rollouts: Sequence[Rollout], limit: int = 2) -> list[dict]:
    """A few (prompt, completion, reward) triples for eyeballing training progress."""
    out: list[dict] = []
    for roll in rollouts:
        for text, reward in zip(roll.texts, roll.rewards):
            if len(out) >= limit:
                return out
            out.append(
                {
                    "question": roll.task.question,
                    "gold": roll.task.answer,
                    "completion": text,
                    "reward": reward.total,
                    "correct": reward.correct,
                    "format_ok": reward.format_ok,
                }
            )
    return out


def resolve_micro_batch(total: int, requested: Optional[int]) -> int:
    return total if not requested or requested <= 0 else min(requested, total)

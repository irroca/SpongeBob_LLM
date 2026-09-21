"""GRPO / RLVR training: sample a group per prompt, score it with a rule-based
environment, and update on group-relative advantages.

Unlike the other stages there is no dataset of targets — the environment
generates prompts and grades completions, so the only supervision is a scalar
reward. The loop per step is: rollout -> reward -> advantage -> clipped policy
gradient (+ optional KL to a frozen reference) -> update.

Variant switches map to the published ablations:

* ``--normalize_advantage_std False``           -> Dr. GRPO advantages
* ``--aggregation token_mean``                  -> DAPO token-level loss
* ``--aggregation dr_grpo``                     -> Dr. GRPO constant normalizer
* ``--clip_eps_high > --clip_eps_low``          -> DAPO clip-higher
* ``--filter_zero_variance True``               -> DAPO dynamic sampling
* ``--kl_coeff 0``                              -> no reference anchor (DAPO / Dr. GRPO default)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch import optim
from transformers import AutoTokenizer

from Config import LLMConfig
from envs import available_envs, load_tasks, make_env
from losses import (
    AGGREGATIONS,
    approx_kl,
    grpo_advantages,
    grpo_policy_loss,
    token_entropy,
    token_logprobs,
    zero_variance_groups,
)
from model import SpongeBob
from rollout import (
    build_prompt_ids,
    collate_rollouts,
    compute_logprobs,
    decode_examples,
    resolve_micro_batch,
    rollout_stats,
    run_rollouts,
    select_rollouts,
)
from train_utils import (
    add_common_train_args,
    build_autocast_scaler,
    get_lr,
    init_wandb_if_needed,
    load_weights,
    optimizer_step,
    save_checkpoint,
    set_seed,
    str2bool,
)


def batch_normalizer(aggregation: str, n_sequences: int, n_tokens: float, max_new_tokens: int) -> float:
    """Denominator for the whole rollout batch.

    Passing it to every micro-batch keeps the summed micro-batch losses equal
    to the loss of a single full-batch pass.
    """
    if aggregation == "seq_mean":
        return float(n_sequences)
    if aggregation == "token_mean":
        return max(n_tokens, 1.0)
    return float(n_sequences * max_new_tokens)


def policy_update(policy, optimizer, scaler, batch, advantages, old_logprobs, ref_logprobs, args, ctx):
    """One or more passes over a rollout batch; returns averaged metrics."""
    normalizer = batch_normalizer(
        args.aggregation, batch.n_sequences, batch.n_tokens, args.max_new_tokens
    )
    micro = resolve_micro_batch(batch.n_sequences, args.micro_batch_size)
    policy.train()
    stats = {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "entropy": 0.0, "clip_frac": 0.0, "ratio_mean": 0.0}

    for _ in range(args.ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, batch.n_sequences, micro):
            stop = start + micro
            with ctx:
                logits = policy(batch.input_ids[start:stop]).logits
                logprobs = token_logprobs(logits, batch.targets[start:stop])
                chunk_mask = batch.mask[start:stop]
                loss, chunk_metrics = grpo_policy_loss(
                    logprobs,
                    old_logprobs[start:stop],
                    advantages[start:stop],
                    chunk_mask,
                    clip_eps_low=args.clip_eps_low,
                    clip_eps_high=args.clip_eps_high,
                    aggregation=args.aggregation,
                    normalizer=normalizer,
                    max_completion_len=args.max_new_tokens,
                )
                kl = torch.zeros((), device=loss.device)
                if ref_logprobs is not None and args.kl_coeff > 0:
                    kl = approx_kl(
                        logprobs,
                        ref_logprobs[start:stop],
                        chunk_mask,
                        aggregation=args.aggregation,
                        normalizer=normalizer,
                        max_completion_len=args.max_new_tokens,
                    )
                    loss = loss + args.kl_coeff * kl

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            with torch.no_grad():
                weight = float(chunk_mask.sum().clamp_min(1))
                entropy = float((token_entropy(logits) * chunk_mask).sum() / weight)
            stats["loss"] += float(loss.detach())
            stats["policy_loss"] += chunk_metrics["policy_loss"]
            stats["kl"] += float(kl.detach())
            stats["entropy"] += entropy * weight
            stats["clip_frac"] += chunk_metrics["clip_frac"] * weight
            stats["ratio_mean"] += chunk_metrics["ratio_mean"] * weight

        optimizer_step(policy, optimizer, scaler, args.grad_clip)

    token_weight = max(batch.n_tokens * args.ppo_epochs, 1.0)
    for key in ("entropy", "clip_frac", "ratio_mean"):
        stats[key] /= token_weight
    for key in ("loss", "policy_loss", "kl"):
        stats[key] /= args.ppo_epochs
    return stats


@torch.no_grad()
def evaluate(policy, tokenizer, env, tasks, args) -> dict:
    """Greedy decode on a fixed task set; reports accuracy/format/hacking rates."""
    rollouts = run_rollouts(
        policy,
        tokenizer,
        env,
        tasks,
        group_size=1,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        device=args.device,
    )
    stats = rollout_stats(rollouts)
    return {f"eval_{k}": v for k, v in stats.items() if k != "silent_group_frac"}


def build_env(args):
    kwargs = {"seed": args.seed}
    if args.env == "arithmetic":
        kwargs.update(
            ops=tuple(op.strip() for op in args.env_ops.split(",") if op.strip()),
            min_digits=args.env_min_digits,
            max_digits=args.env_max_digits,
            format_weight=args.format_weight,
            strict=args.strict_answer,
        )
    return make_env(args.env, **kwargs)


def check_length_budget(tokenizer, env, args) -> None:
    probe = build_prompt_ids(tokenizer, env, env.sample_task(), "cpu")
    needed = probe.size(0) + args.max_new_tokens
    if needed > args.max_seq_len:
        raise ValueError(
            f"prompt ({probe.size(0)} tokens) + --max_new_tokens ({args.max_new_tokens}) "
            f"= {needed} exceeds --max_seq_len ({args.max_seq_len}); RoPE has no positions left"
        )


def main():
    parser = argparse.ArgumentParser(description="GRPO training with verifiable rewards")
    add_common_train_args(
        parser,
        batch_size=4,
        learning_rate=1e-6,
        log_step=1,
        save_step=50,
        max_seq_len=512,
        data_path="",
        wandb_project="SpongeBob-GRPO",
        skip=("epochs", "accumulation_steps", "num_workers"),
    )
    parser.add_argument("--policy_path", type=str, required=True, help="Init policy (usually SFT)")
    parser.add_argument("--ref_path", type=str, default=None, help="Frozen KL reference; default=policy_path")
    parser.add_argument("--rl_steps", type=int, default=20, help="Number of policy updates")
    parser.add_argument("--group_size", type=int, default=4, help="Rollouts sampled per prompt (G)")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--micro_batch_size", type=int, default=0, help="0 = whole rollout batch at once")
    parser.add_argument("--ppo_epochs", type=int, default=1, help="Reuse passes per rollout batch")

    parser.add_argument("--clip_eps_low", type=float, default=0.2)
    parser.add_argument("--clip_eps_high", type=float, default=0.2)
    parser.add_argument("--kl_coeff", type=float, default=0.0)
    parser.add_argument("--aggregation", type=str, default="seq_mean", choices=list(AGGREGATIONS))
    parser.add_argument("--normalize_advantage_std", type=str2bool, default=True)
    parser.add_argument("--filter_zero_variance", type=str2bool, default=False)

    parser.add_argument("--env", type=str, default="arithmetic", choices=available_envs())
    parser.add_argument("--env_ops", type=str, default="+,-")
    parser.add_argument("--env_min_digits", type=int, default=1)
    parser.add_argument("--env_max_digits", type=int, default=2)
    parser.add_argument("--format_weight", type=float, default=0.2)
    parser.add_argument("--strict_answer", type=str2bool, default=True)

    parser.add_argument("--eval_path", type=str, default="", help="Held-out task JSONL; empty = sample one")
    parser.add_argument("--eval_size", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=10, help="0 disables periodic eval")
    parser.add_argument("--metrics_path", type=str, default="", help="Default: {save_dir}/grpo_metrics.jsonl")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    metrics_path = args.metrics_path or os.path.join(args.save_dir, "grpo_metrics.jsonl")

    tokenizer = AutoTokenizer.from_pretrained("./spongebob_tokenizer")
    args.lm_config = LLMConfig(max_seq_len=args.max_seq_len, vocab_size=tokenizer.vocab_size)

    policy = SpongeBob(args.lm_config).to(args.device)
    load_weights(args.policy_path, policy, args.device, strict=False)

    ref = None
    if args.kl_coeff > 0:
        ref = SpongeBob(args.lm_config).to(args.device)
        load_weights(args.ref_path or args.policy_path, ref, args.device, strict=False)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    env = build_env(args)
    check_length_budget(tokenizer, env, args)

    # Prompts come from the env generator unless a fixed pool is supplied.
    prompt_pool = load_tasks(args.data_path) if args.data_path else None
    if args.eval_path:
        eval_tasks = load_tasks(args.eval_path)[: args.eval_size]
    else:
        eval_env = build_env(args)
        eval_env.reseed(args.seed + 10_000)
        eval_tasks = eval_env.sample(args.eval_size)

    optimizer = optim.AdamW(policy.parameters(), lr=args.learning_rate)
    ctx, scaler = build_autocast_scaler(args.device, args.dtype)
    wandb = init_wandb_if_needed(args, run_name=f"grpo-g{args.group_size}-b{args.batch_size}")

    print(
        f"GRPO: env={args.env} G={args.group_size} prompts/step={args.batch_size} "
        f"steps={args.rl_steps} agg={args.aggregation} kl={args.kl_coeff} "
        f"std_norm={args.normalize_advantage_std} filter_zero_var={args.filter_zero_variance}"
    )

    rng = torch.Generator().manual_seed(args.seed)
    for step in range(1, args.rl_steps + 1):
        started = time.time()
        lr = get_lr(step, args.rl_steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if prompt_pool:
            idx = torch.randint(len(prompt_pool), (args.batch_size,), generator=rng).tolist()
            tasks = [prompt_pool[i] for i in idx]
        else:
            tasks = env.sample(args.batch_size)

        rollouts = run_rollouts(
            policy,
            tokenizer,
            env,
            tasks,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
        record = {"step": step, "lr": lr, **rollout_stats(rollouts)}

        if args.filter_zero_variance:
            keep = ~zero_variance_groups(torch.stack([r.reward_tensor for r in rollouts]))
            rollouts = select_rollouts(rollouts, keep)
        record["groups_used"] = len(rollouts)

        if not rollouts:
            # Every group agreed, so every advantage is zero: nothing to learn from.
            record.update(loss=0.0, policy_loss=0.0, kl=0.0, entropy=0.0, clip_frac=0.0, ratio_mean=1.0)
        else:
            batch = collate_rollouts(rollouts, tokenizer.pad_token_id)
            advantages = grpo_advantages(
                batch.rewards, normalize_std=args.normalize_advantage_std
            ).reshape(-1)
            micro = resolve_micro_batch(batch.n_sequences, args.micro_batch_size)
            old_logprobs = compute_logprobs(policy, batch.input_ids, batch.targets, micro, ctx)
            ref_logprobs = (
                compute_logprobs(ref, batch.input_ids, batch.targets, micro, ctx)
                if ref is not None
                else None
            )
            record.update(
                policy_update(
                    policy, optimizer, scaler, batch, advantages, old_logprobs, ref_logprobs, args, ctx
                )
            )

        record["sec"] = round(time.time() - started, 2)
        if args.eval_every and (step % args.eval_every == 0 or step == args.rl_steps):
            record.update(evaluate(policy, tokenizer, env, eval_tasks, args))

        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if wandb is not None:
            wandb.log(record)

        if step % args.log_step == 0:
            print(
                f"step {step}/{args.rl_steps} reward={record['reward_mean']:.3f} "
                f"acc={record['accuracy']:.2f} fmt={record['format_rate']:.2f} "
                f"hack={record['hack_rate']:.2f} silent={record['silent_group_frac']:.2f} "
                f"len={record['completion_len']:.1f} kl={record['kl']:.4f} "
                f"ent={record['entropy']:.3f} loss={record['loss']:.4f} lr={lr:.2e} "
                f"({record['sec']}s)"
            )
            if "eval_accuracy" in record:
                print(f"  eval: acc={record['eval_accuracy']:.3f} fmt={record['eval_format_rate']:.3f}")
            for example in decode_examples(rollouts, limit=1):
                print(f"  sample[{example['question']}={example['gold']}]: {example['completion']!r}")

        if args.save_step and step % args.save_step == 0:
            save_checkpoint(
                f"{args.save_dir}/latest_checkpoint.pth",
                policy, optimizer, scaler, 0, step, step, record["loss"], args.lm_config,
            )

    final_path = f"{args.save_dir}/grpo_final.pth"
    torch.save(policy.state_dict(), final_path)
    print(f"Saved {final_path}; metrics -> {metrics_path}")


if __name__ == "__main__":
    main()

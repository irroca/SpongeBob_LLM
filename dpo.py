"""Direct Preference Optimization (DPO) training."""

from __future__ import annotations

import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Config import LLMConfig
from dataset import PreferenceDataset
from losses import dpo_loss, sequence_logprobs
from model import SpongeBob
from train_utils import (
    add_common_train_args,
    build_autocast_scaler,
    flush_pending_grads,
    get_lr,
    init_wandb_if_needed,
    load_train_state,
    load_weights,
    optimizer_step,
    save_checkpoint,
    set_seed,
)


def train_epoch(epoch, start_step, global_step, policy, ref, optimizer, scaler, loader, args, ctx, wandb):
    policy.train()
    ref.eval()
    pending = False
    current_loss = 0.0
    for step, batch in enumerate(loader):
        if step < start_step:
            continue
        cX, cY, cM, rX, rY, rM = [t.to(args.device) for t in batch]
        lr = get_lr(max(global_step, 1), args.total_steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with ctx:
            policy_chosen_logits = policy(cX).logits
            policy_rejected_logits = policy(rX).logits
            with torch.no_grad():
                ref_chosen_logits = ref(cX).logits
                ref_rejected_logits = ref(rX).logits

            loss = dpo_loss(
                sequence_logprobs(policy_chosen_logits, cY, cM),
                sequence_logprobs(policy_rejected_logits, rY, rM),
                sequence_logprobs(ref_chosen_logits, cY, cM),
                sequence_logprobs(ref_rejected_logits, rY, rM),
                beta=args.beta,
            )
            loss = loss / args.accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        current_loss = loss.item() * args.accumulation_steps
        pending = True
        if (step + 1) % args.accumulation_steps == 0:
            optimizer_step(policy, optimizer, scaler, args.grad_clip)
            pending = False
            global_step += 1

        if step % args.log_step == 0:
            print(
                f"Epoch[{epoch+1}/{args.epochs}] ({step}/{len(loader)}) "
                f"dpo_loss={current_loss:.4f} "
                f"lr={optimizer.param_groups[-1]['lr']:.7f} global_step={global_step}"
            )
            if wandb is not None:
                wandb.log(
                    {
                        "dpo_loss": current_loss,
                        "lr": optimizer.param_groups[-1]["lr"],
                        "global_step": global_step,
                    }
                )

        if global_step > 0 and global_step % args.save_step == 0:
            save_checkpoint(
                f"{args.save_dir}/latest_checkpoint.pth",
                policy, optimizer, scaler, epoch, step, global_step,
                current_loss, args.lm_config,
            )

    if not flush_pending_grads(policy, optimizer, scaler, args.grad_clip, pending) and pending:
        global_step += 1
    return global_step, current_loss


def main():
    parser = argparse.ArgumentParser(description="DPO preference optimization")
    add_common_train_args(
        parser,
        batch_size=2,
        learning_rate=1e-5,
        wandb_project="SpongeBob-DPO",
        log_step=1,
        max_seq_len=256,
        data_path="tests/fixtures/preference_tiny.jsonl",
    )
    parser.add_argument("--policy_path", type=str, required=True, help="Init policy (usually SFT)")
    parser.add_argument("--ref_path", type=str, default=None, help="Frozen reference; default=policy_path")
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    if args.ref_path is None:
        args.ref_path = args.policy_path

    tokenizer = AutoTokenizer.from_pretrained("./spongebob_tokenizer")
    cfg = LLMConfig(max_seq_len=args.max_seq_len, vocab_size=tokenizer.vocab_size)
    args.lm_config = cfg

    policy = SpongeBob(cfg).to(args.device)
    ref = SpongeBob(cfg).to(args.device)
    load_weights(args.policy_path, policy, args.device, strict=False)
    load_weights(args.ref_path, ref, args.device, strict=False)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    optimizer = optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.learning_rate)
    ctx, scaler = build_autocast_scaler(args.device, args.dtype)

    start_epoch, start_step, global_step = 0, 0, 0
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = load_weights(args.resume_from, policy, args.device, strict=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            start_epoch, start_step, global_step, _ = load_train_state(ckpt, optimizer, scaler)

    wandb = init_wandb_if_needed(args, run_name=f"dpo-bs{args.batch_size}")

    ds = PreferenceDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    args.total_steps = max(1, args.epochs * len(loader) // args.accumulation_steps)

    print(f"DPO: beta={args.beta} steps={args.total_steps}")
    for epoch in range(start_epoch, args.epochs):
        global_step, last_loss = train_epoch(
            epoch, start_step if epoch == start_epoch else 0,
            global_step, policy, ref, optimizer, scaler, loader, args, ctx, wandb,
        )
        start_step = 0
        save_checkpoint(
            f"{args.save_dir}/epoch_{epoch+1}_checkpoint.pth",
            policy, optimizer, scaler, epoch + 1, 0, global_step, last_loss, args.lm_config,
        )

    final_path = f"{args.save_dir}/dpo_final.pth"
    torch.save(policy.state_dict(), final_path)
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()

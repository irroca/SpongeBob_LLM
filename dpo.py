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
    build_autocast_scaler,
    get_lr,
    load_train_state,
    load_weights,
    save_checkpoint,
    set_seed,
)


def train_epoch(epoch, start_step, global_step, policy, ref, optimizer, scaler, loader, args, ctx):
    policy.train()
    ref.eval()
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

        if (step + 1) % args.accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if step % args.log_step == 0:
            print(
                f"Epoch[{epoch+1}/{args.epochs}] ({step}/{len(loader)}) "
                f"dpo_loss={loss.item() * args.accumulation_steps:.4f} "
                f"lr={optimizer.param_groups[-1]['lr']:.7f} global_step={global_step}"
            )

        if global_step > 0 and global_step % args.save_step == 0:
            save_checkpoint(
                f"{args.save_dir}/latest_checkpoint.pth",
                policy, optimizer, scaler, epoch, step, global_step,
                loss.item() * args.accumulation_steps, args.lm_config,
            )
    return global_step


def main():
    parser = argparse.ArgumentParser(description="DPO preference optimization")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_step", type=int, default=1)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--data_path", type=str, default="tests/fixtures/preference_tiny.jsonl")
    parser.add_argument("--policy_path", type=str, required=True, help="Init policy (usually SFT)")
    parser.add_argument("--ref_path", type=str, default=None, help="Frozen reference; default=policy_path")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
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

    ds = PreferenceDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    args.total_steps = max(1, args.epochs * len(loader) // args.accumulation_steps)

    print(f"DPO: beta={args.beta} steps={args.total_steps}")
    for epoch in range(start_epoch, args.epochs):
        global_step = train_epoch(
            epoch, start_step if epoch == start_epoch else 0,
            global_step, policy, ref, optimizer, scaler, loader, args, ctx,
        )
        start_step = 0
        save_checkpoint(
            f"{args.save_dir}/epoch_{epoch+1}_checkpoint.pth",
            policy, optimizer, scaler, epoch + 1, 0, global_step, 0.0, args.lm_config,
        )

    final_path = f"{args.save_dir}/dpo_final.pth"
    torch.save(policy.state_dict(), final_path)
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()

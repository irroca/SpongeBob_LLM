"""Supervised fine-tuning from a pretrained checkpoint."""

from __future__ import annotations

import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Config import LLMConfig
from dataset import SFTDataset
from losses import masked_cross_entropy
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


def train_epoch(epoch, start_step, global_step, model, optimizer, scaler, loader, args, ctx, wandb):
    model.train()
    pending = False
    current_loss = 0.0
    for step, (X, Y, loss_mask) in enumerate(loader):
        if step < start_step:
            continue
        X, Y, loss_mask = X.to(args.device), Y.to(args.device), loss_mask.to(args.device)
        lr = get_lr(max(global_step, 1), args.total_steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with ctx:
            out = model(X)
            loss = masked_cross_entropy(out.logits, Y, loss_mask) / args.accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        current_loss = loss.item() * args.accumulation_steps
        pending = True
        if (step + 1) % args.accumulation_steps == 0:
            optimizer_step(model, optimizer, scaler, args.grad_clip)
            pending = False
            global_step += 1

        if step % args.log_step == 0:
            print(
                f"Epoch[{epoch+1}/{args.epochs}] ({step}/{len(loader)}) "
                f"loss={current_loss:.4f} lr={optimizer.param_groups[-1]['lr']:.7f} "
                f"global_step={global_step}"
            )
            if wandb is not None:
                wandb.log({"loss": current_loss, "lr": optimizer.param_groups[-1]["lr"], "global_step": global_step})

        if global_step > 0 and global_step % args.save_step == 0:
            save_checkpoint(
                f"{args.save_dir}/latest_checkpoint.pth",
                model, optimizer, scaler, epoch, step, global_step, current_loss, args.lm_config,
            )

    if not flush_pending_grads(model, optimizer, scaler, args.grad_clip, pending) and pending:
        global_step += 1
    return global_step, current_loss


def main():
    parser = argparse.ArgumentParser()
    add_common_train_args(
        parser,
        learning_rate=1e-4,
        wandb_project="SpongeBob-SFT",
        data_path="datasets/sft_512.jsonl",
    )
    parser.add_argument("--pretrained_path", type=str, default="./results/pretrain_final.pth")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("./spongebob_tokenizer")
    args.lm_config = LLMConfig(max_seq_len=args.max_seq_len, vocab_size=tokenizer.vocab_size)
    model = SpongeBob(args.lm_config).to(args.device)

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from {args.pretrained_path}")
        load_weights(args.pretrained_path, model, args.device, strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    ctx, scaler = build_autocast_scaler(args.device, args.dtype)

    start_epoch, start_step, global_step = 0, 0, 0
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = load_weights(args.resume_from, model, args.device, strict=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            start_epoch, start_step, global_step, _ = load_train_state(ckpt, optimizer, scaler)

    print(f"LLM parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")

    wandb = init_wandb_if_needed(args, run_name=f"sft-bs{args.batch_size}")

    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    args.total_steps = max(1, args.epochs * len(loader) // args.accumulation_steps)

    for epoch in range(start_epoch, args.epochs):
        global_step, last_loss = train_epoch(
            epoch, start_step if epoch == start_epoch else 0,
            global_step, model, optimizer, scaler, loader, args, ctx, wandb,
        )
        start_step = 0
        save_checkpoint(
            f"{args.save_dir}/epoch_{epoch+1}_checkpoint.pth",
            model, optimizer, scaler, epoch + 1, 0, global_step, last_loss, args.lm_config,
        )

    torch.save(model.state_dict(), f"{args.save_dir}/sft_final.pth")
    print("SFT Training completed!")


if __name__ == "__main__":
    main()

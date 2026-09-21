"""Causal LM pretraining."""

from __future__ import annotations

import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import PretrainDataset
from losses import masked_cross_entropy
from model import SpongeBob
from train_utils import (
    add_common_train_args,
    add_model_args,
    build_autocast_scaler,
    describe_model,
    flush_pending_grads,
    get_lr,
    init_wandb_if_needed,
    load_train_state,
    load_weights,
    optimizer_step,
    optimizer_step,
    resolve_model_config,
    save_checkpoint,
    save_final_weights,
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

    if flush_pending_grads(model, optimizer, scaler, args.grad_clip, pending):
        global_step += 1
    return global_step, current_loss


def main():
    parser = argparse.ArgumentParser()
    add_common_train_args(
        parser,
        learning_rate=5e-4,
        wandb_project="SpongeBob-Pretrain",
        data_path="datasets/pretrain.jsonl",
    )
    add_model_args(parser)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    args.lm_config = resolve_model_config(
        args, tokenizer.vocab_size, checkpoint_path=args.resume_from
    )
    model = SpongeBob(args.lm_config).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    ctx, scaler = build_autocast_scaler(args.device, args.dtype)

    start_epoch, start_step, global_step = 0, 0, 0
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = load_weights(args.resume_from, model, args.device, strict=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            start_epoch, start_step, global_step, _ = load_train_state(ckpt, optimizer, scaler)

    print(describe_model(model, args.lm_config, "pretrain"))

    wandb = init_wandb_if_needed(args, run_name=f"pretrain-bs{args.batch_size}")

    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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

    save_final_weights(f"{args.save_dir}/pretrain_final.pth", model, args.lm_config)
    print("Training completed!")


if __name__ == "__main__":
    main()

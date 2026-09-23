"""Causal LM pretraining."""

from __future__ import annotations

import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import PretrainDataset
from evaluate import evaluate_lm
from losses import masked_cross_entropy
from model import Whetstone
from runlog import RunRecorder
from train_utils import (
    add_common_train_args,
    add_model_args,
    build_autocast_scaler,
    build_val_loader,
    describe_model,
    flush_pending_grads,
    get_lr,
    init_wandb_if_needed,
    load_train_state,
    load_weights,
    optimizer_step,
    resolve_model_config,
    save_checkpoint,
    save_final_weights,
    set_seed,
    should_evaluate,
)


def train_epoch(
    epoch, start_step, global_step, model, optimizer, scaler, loader, args, ctx, wandb,
    recorder=None, val_loader=None,
):
    model.train()
    pending = False
    current_loss = 0.0
    grad_norm = 0.0
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
        if recorder is not None:
            recorder.add_tokens(int(loss_mask.sum()))
        if (step + 1) % args.accumulation_steps == 0:
            grad_norm = optimizer_step(model, optimizer, scaler, args.grad_clip)
            pending = False
            global_step += 1

        if step % args.log_step == 0:
            print(
                f"Epoch[{epoch+1}/{args.epochs}] ({step}/{len(loader)}) "
                f"loss={current_loss:.4f} lr={optimizer.param_groups[-1]['lr']:.7f} "
                f"global_step={global_step}"
            )
            if recorder is not None:
                recorder.log(global_step, epoch=epoch + 1, loss=current_loss, lr=lr, grad_norm=grad_norm)
            if wandb is not None:
                wandb.log({"loss": current_loss, "lr": optimizer.param_groups[-1]["lr"], "global_step": global_step})

        if val_loader is not None and should_evaluate(global_step, args):
            stats = evaluate_lm(model, val_loader, args.device, ctx, args.val_batches or None)
            print(f"  val: loss={stats['loss']:.4f} ppl={stats['ppl']:.2f} ({stats['tokens']} tokens)")
            if recorder is not None:
                recorder.log_eval(global_step, epoch=epoch + 1, **stats)
            if wandb is not None:
                wandb.log({"val_loss": stats["loss"], "val_ppl": stats["ppl"], "global_step": global_step})

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
        wandb_project="Whetstone-Pretrain",
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
    model = Whetstone(args.lm_config).to(args.device)
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
    val_loader = build_val_loader(PretrainDataset, args, tokenizer)
    args.total_steps = max(1, args.epochs * len(loader) // args.accumulation_steps)

    with RunRecorder.start(
        "pretrain", args, config=args.lm_config, model=model,
        data_paths=[args.data_path, args.val_data_path],
    ) as recorder:
        print(f"run: {recorder.run_dir}")
        for epoch in range(start_epoch, args.epochs):
            global_step, last_loss = train_epoch(
                epoch, start_step if epoch == start_epoch else 0,
                global_step, model, optimizer, scaler, loader, args, ctx, wandb,
                recorder=recorder, val_loader=val_loader,
            )
            start_step = 0
            save_checkpoint(
                f"{args.save_dir}/epoch_{epoch+1}_checkpoint.pth",
                model, optimizer, scaler, epoch + 1, 0, global_step, last_loss, args.lm_config,
            )

        if val_loader is not None:
            stats = evaluate_lm(model, val_loader, args.device, ctx, args.val_batches or None)
            print(f"final val: loss={stats['loss']:.4f} ppl={stats['ppl']:.2f}")
            recorder.log_eval(global_step, epoch=args.epochs, **stats)

        save_final_weights(f"{args.save_dir}/pretrain_final.pth", model, args.lm_config)
        recorder.finish(status="completed", steps=global_step)
    print("Training completed!")


if __name__ == "__main__":
    main()

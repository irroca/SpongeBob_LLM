"""Real knowledge distillation: frozen teacher + student CE + temperature KL."""

from __future__ import annotations

import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Config import LLMConfig
from dataset import SFTDataset
from losses import kd_loss, masked_cross_entropy
from model import SpongeBob
from train_utils import (
    build_autocast_scaler,
    get_lr,
    load_train_state,
    load_weights,
    save_checkpoint,
    set_seed,
    str2bool,
)


def train_epoch(epoch, start_step, global_step, student, teacher, optimizer, scaler, loader, args, ctx):
    student.train()
    teacher.eval()
    for step, (X, Y, loss_mask) in enumerate(loader):
        if step < start_step:
            continue
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # global_step counts optimizer updates (1-based after first update)
        lr = get_lr(max(global_step, 1), args.total_steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with ctx:
            student_out = student(X)
            with torch.no_grad():
                teacher_out = teacher(X)
            ce = masked_cross_entropy(student_out.logits, Y, loss_mask)
            kd = kd_loss(
                student_out.logits,
                teacher_out.logits.detach(),
                temperature=args.temperature,
                mask=loss_mask,
            )
            loss = ((1.0 - args.alpha) * ce + args.alpha * kd) / args.accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if step % args.log_step == 0:
            print(
                f"Epoch[{epoch+1}/{args.epochs}] ({step}/{len(loader)}) "
                f"loss={loss.item() * args.accumulation_steps:.4f} "
                f"ce={ce.item():.4f} kd={kd.item():.4f} lr={optimizer.param_groups[-1]['lr']:.7f} "
                f"global_step={global_step}"
            )

        if global_step > 0 and global_step % args.save_step == 0:
            save_checkpoint(
                f"{args.save_dir}/latest_checkpoint.pth",
                student,
                optimizer,
                scaler,
                epoch,
                step,
                global_step,
                loss.item() * args.accumulation_steps,
                args.lm_config,
            )

    return global_step


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation (teacher -> student)")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_step", type=int, default=1)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--data_path", type=str, default="tests/fixtures/sft_tiny.jsonl")
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--student_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="KD mix weight")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    args.lm_config = LLMConfig(max_seq_len=args.max_seq_len)

    tokenizer = AutoTokenizer.from_pretrained("./spongebob_tokenizer")
    cfg = LLMConfig(max_seq_len=args.max_seq_len, vocab_size=tokenizer.vocab_size)
    args.lm_config = cfg

    teacher = SpongeBob(cfg).to(args.device)
    student = SpongeBob(cfg).to(args.device)

    load_weights(args.teacher_path, teacher, args.device, strict=False)
    load_weights(args.student_path, student, args.device, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=args.learning_rate)
    ctx, scaler = build_autocast_scaler(args.device, args.dtype)

    start_epoch, start_step, global_step, _ = 0, 0, 0, float("inf")
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = load_weights(args.resume_from, student, args.device, strict=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            start_epoch, start_step, global_step, _ = load_train_state(ckpt, optimizer, scaler)

    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    args.total_steps = max(1, args.epochs * len(loader) // args.accumulation_steps)

    print(f"KD: alpha={args.alpha} T={args.temperature} steps={args.total_steps}")
    for epoch in range(start_epoch, args.epochs):
        global_step = train_epoch(
            epoch, start_step if epoch == start_epoch else 0,
            global_step, student, teacher, optimizer, scaler, loader, args, ctx,
        )
        start_step = 0
        save_checkpoint(
            f"{args.save_dir}/epoch_{epoch+1}_checkpoint.pth",
            student, optimizer, scaler, epoch + 1, 0, global_step, 0.0, args.lm_config,
        )

    final_path = f"{args.save_dir}/distill_final.pth"
    torch.save(student.state_dict(), final_path)
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()

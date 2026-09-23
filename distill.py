"""Real knowledge distillation: frozen teacher + student CE + temperature KL."""

from __future__ import annotations

import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import SFTDataset
from evaluate import evaluate_lm
from losses import kd_loss, masked_cross_entropy
from model import Whetstone
from runlog import RunRecorder
from train_utils import (
    MODEL_ARCH_FIELDS,
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
    optimizer_step,
    resolve_model_config,
    save_checkpoint,
    save_final_weights,
    set_seed,
    should_evaluate,
)


def train_epoch(
    epoch, start_step, global_step, student, teacher, optimizer, scaler, loader, args, ctx, wandb,
    recorder=None, val_loader=None,
):
    student.train()
    teacher.eval()
    pending = False
    current_loss = 0.0
    grad_norm = 0.0
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

        current_loss = loss.item() * args.accumulation_steps
        pending = True
        if recorder is not None:
            recorder.add_tokens(int(loss_mask.sum()))
        if (step + 1) % args.accumulation_steps == 0:
            grad_norm = optimizer_step(student, optimizer, scaler, args.grad_clip)
            pending = False
            global_step += 1

        if step % args.log_step == 0:
            print(
                f"Epoch[{epoch+1}/{args.epochs}] ({step}/{len(loader)}) "
                f"loss={current_loss:.4f} "
                f"ce={ce.item():.4f} kd={kd.item():.4f} lr={optimizer.param_groups[-1]['lr']:.7f} "
                f"global_step={global_step}"
            )
            if recorder is not None:
                recorder.log(
                    global_step, epoch=epoch + 1, loss=current_loss,
                    ce=ce.item(), kd=kd.item(), lr=lr, grad_norm=grad_norm,
                )
            if wandb is not None:
                wandb.log(
                    {
                        "loss": current_loss,
                        "ce": ce.item(),
                        "kd": kd.item(),
                        "lr": optimizer.param_groups[-1]["lr"],
                        "global_step": global_step,
                    }
                )

        if val_loader is not None and should_evaluate(global_step, args):
            stats = evaluate_lm(student, val_loader, args.device, ctx, args.val_batches or None)
            print(f"  val: loss={stats['loss']:.4f} ppl={stats['ppl']:.2f} ({stats['tokens']} tokens)")
            if recorder is not None:
                recorder.log_eval(global_step, epoch=epoch + 1, **stats)

        if global_step > 0 and global_step % args.save_step == 0:
            save_checkpoint(
                f"{args.save_dir}/latest_checkpoint.pth",
                student,
                optimizer,
                scaler,
                epoch,
                step,
                global_step,
                current_loss,
                args.lm_config,
            )

    if flush_pending_grads(student, optimizer, scaler, args.grad_clip, pending):
        global_step += 1
    return global_step, current_loss


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation (teacher -> student)")
    add_common_train_args(
        parser,
        batch_size=4,
        learning_rate=1e-4,
        wandb_project="Whetstone-Distill",
        log_step=1,
        max_seq_len=256,
        data_path="tests/fixtures/sft_tiny.jsonl",
    )
    add_model_args(parser)
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--student_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="KD mix weight")
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # Teacher and student get independent architectures resolved from their own
    # checkpoints; the --dim/--n_layers flags describe the *student* only, since
    # that is the model being trained. KD across sizes only needs a shared vocab,
    # which resolve_model_config enforces against the tokenizer.
    teacher_cfg = resolve_model_config(
        argparse.Namespace(**{f: None for f in MODEL_ARCH_FIELDS}, max_seq_len=args.max_seq_len),
        tokenizer.vocab_size,
        checkpoint_path=args.teacher_path,
    )
    args.lm_config = resolve_model_config(
        args, tokenizer.vocab_size, checkpoint_path=args.resume_from or args.student_path
    )

    teacher = Whetstone(teacher_cfg).to(args.device)
    student = Whetstone(args.lm_config).to(args.device)
    print(describe_model(teacher, teacher_cfg, "teacher"))
    print(describe_model(student, args.lm_config, "student"))

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

    wandb = init_wandb_if_needed(args, run_name=f"distill-bs{args.batch_size}")

    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_val_loader(SFTDataset, args, tokenizer)
    args.total_steps = max(1, args.epochs * len(loader) // args.accumulation_steps)

    print(f"KD: alpha={args.alpha} T={args.temperature} steps={args.total_steps}")
    recorder = RunRecorder.start(
        "distill", args, config=args.lm_config, model=student,
        data_paths=[args.data_path, args.val_data_path],
        extra={"teacher": {"path": args.teacher_path, **{
            k: getattr(teacher_cfg, k) for k in ("dim", "n_layers", "n_heads", "n_kv_heads")
        }}, "kd": {"alpha": args.alpha, "temperature": args.temperature}},
    )
    print(f"run: {recorder.run_dir}")
    for epoch in range(start_epoch, args.epochs):
        global_step, last_loss = train_epoch(
            epoch, start_step if epoch == start_epoch else 0,
            global_step, student, teacher, optimizer, scaler, loader, args, ctx, wandb,
            recorder=recorder, val_loader=val_loader,
        )
        start_step = 0
        save_checkpoint(
            f"{args.save_dir}/epoch_{epoch+1}_checkpoint.pth",
            student, optimizer, scaler, epoch + 1, 0, global_step, last_loss, args.lm_config,
        )

    if val_loader is not None:
        stats = evaluate_lm(student, val_loader, args.device, ctx, args.val_batches or None)
        print(f"final val: loss={stats['loss']:.4f} ppl={stats['ppl']:.2f}")
        recorder.log_eval(global_step, epoch=args.epochs, **stats)

    final_path = f"{args.save_dir}/distill_final.pth"
    save_final_weights(final_path, student, args.lm_config)
    recorder.finish(status="completed", steps=global_step)
    print(f"Saved {final_path}; run -> {recorder.run_dir}")


if __name__ == "__main__":
    main()

import math
import os
import tempfile

import torch
import torch.nn as nn

from Config import LLMConfig
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


def test_str2bool():
    assert str2bool("True") is True
    assert str2bool("false") is False
    assert str2bool(True) is True


def test_get_lr_warmup_nonzero_and_continuous():
    total = 100
    lr = 1e-3
    # step is 1-based optimizer update index in our API
    lr1 = get_lr(1, total, lr, warmup_ratio=0.1)
    assert lr1 > 0
    lr_end_warmup = get_lr(10, total, lr, warmup_ratio=0.1)
    lr_start_cosine = get_lr(11, total, lr, warmup_ratio=0.1)
    # continuous: cosine side should not jump above warmup end by a huge margin
    assert abs(lr_end_warmup - lr) / lr < 1e-6
    assert lr_start_cosine <= lr * 1.01
    lr_mid = get_lr(50, total, lr, warmup_ratio=0.1)
    assert lr_mid < lr
    lr_last = get_lr(100, total, lr, warmup_ratio=0.1)
    assert lr_last <= lr_mid + 1e-12


def test_scaler_only_for_fp16_on_cuda_device_string():
    # On CPU, autocast disabled; scaler should be None regardless for safety when not cuda
    _, scaler_cpu = build_autocast_scaler("cpu", "float16")
    assert scaler_cpu is None
    _, scaler_bf16 = build_autocast_scaler("cuda:0", "bfloat16")
    # Even if CUDA absent, API should not enable GradScaler for bf16
    assert scaler_bf16 is None or scaler_bf16.is_enabled() is False


def test_load_weights_accepts_raw_and_wrapped_checkpoint():
    set_seed(0)
    cfg = LLMConfig(dim=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=32, vocab_size=128)
    model = SpongeBob(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as td:
        raw_path = os.path.join(td, "raw.pth")
        ckpt_path = os.path.join(td, "ckpt.pth")
        torch.save(model.state_dict(), raw_path)
        save_checkpoint(ckpt_path, model, opt, None, epoch=1, step=2, global_step=3, loss=0.5, config=cfg)

        model2 = SpongeBob(cfg)
        load_weights(raw_path, model2, "cpu", strict=True)
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

        model3 = SpongeBob(cfg)
        ckpt = load_weights(ckpt_path, model3, "cpu", strict=True)
        assert "model_state_dict" in ckpt
        epoch, step, global_step, loss = load_train_state(ckpt, opt, None)
        assert (epoch, step, global_step, loss) == (1, 2, 3, 0.5)
        for p1, p2 in zip(model.parameters(), model3.parameters()):
            assert torch.allclose(p1, p2)

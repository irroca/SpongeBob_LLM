# Experiments (fill on your GPU)

This template is for **local GPU runs** with real data. The cloud/CPU smoke on `tests/fixtures/*` only proves the pipeline runs.

## Setup

- GPU, CUDA PyTorch matching your driver
- Data paths (example):
  - `datasets/pretrain.jsonl`
  - `datasets/sft.jsonl`
  - `datasets/preference.jsonl`

## Suggested protocol

| Stage | Key hparams | Checkpoint | Metrics to record |
|-------|-------------|------------|-------------------|
| Pretrain | lr, seq, steps | `pretrain_final.pth` | train loss, eval PPL |
| SFT | lr, epochs | `sft_final.pth` | loss, qualitative chat |
| KD | α, T | `distill_final.pth` | CE, KD term, PPL vs SFT |
| DPO | β | `dpo_final.pth` | DPO loss, win-rate vs SFT (manual or judge) |

## Results (owner fills)

### Loss / PPL

| Model | Eval PPL | Notes |
|-------|----------|-------|
| Pretrain | | |
| SFT | | |
| KD | | |
| DPO | | |

### Qualitative examples

| Prompt | SFT | KD | DPO |
|--------|-----|----|-----|
| | | | |

### Ablations (optional)

- GQA on/off (`n_kv_heads`)
- KD α ∈ {0.3, 0.5, 0.7}, T ∈ {1, 2, 4}
- DPO β ∈ {0.05, 0.1, 0.5}

## Commands (GPU example)

```bash
python pretrain.py --data_path datasets/pretrain.jsonl --device cuda:0 --dtype bfloat16 ...
python SFT.py --pretrained_path results/pretrain_final.pth --device cuda:0 --dtype bfloat16 ...
python distill.py --teacher_path results/sft_final.pth --student_path results/sft_final.pth --device cuda:0 ...
python dpo.py --policy_path results/sft_final.pth --device cuda:0 ...
```

Note: GradScaler is enabled only for `float16`; prefer `bfloat16` on modern GPUs without a scaler.

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
| GRPO | G, β_KL, clip, aggregation | `grpo_final.pth` | eval accuracy, format rate, hack rate, silent-group fraction, entropy, completion length |

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
python grpo.py --policy_path results/sft_final.pth --device cuda:0 --dtype bfloat16 ...
```

Note: GradScaler is enabled only for `float16`; prefer `bfloat16` on modern GPUs without a scaler.

---

# Mini-RLVR (GRPO) experiment plan

The RL stage is where this project stops being a re-implementation exercise, so it gets its own
protocol. Everything below runs on one task family (`envs/arithmetic.py`), which is what makes the
SFT / DPO / GRPO numbers comparable.

## Setup

```bash
python3 -m envs.generate_data --split sft        --n 4000 --out datasets/arith_sft.jsonl
python3 -m envs.generate_data --split preference --n 2000 --out datasets/arith_pref.jsonl
python3 -m envs.generate_data --split eval       --n 200  --out datasets/arith_eval.jsonl --seed 777
```

**Cold start is mandatory.** A policy that never emits `<think>/<answer>` scores 0 on every rollout,
so every group has zero reward variance and the gradient is exactly 0. Run SFT on the gold CoT data
first, and stop it *before* it saturates — a fully converged SFT policy has low entropy and produces
no reward variance either.

## Metrics

Every step appends one JSON line to `{save_dir}/grpo_metrics.jsonl`. Plot at minimum:

| Metric | What it answers |
|--------|-----------------|
| `eval_accuracy` | Is the model actually getting better at the task? |
| `format_rate` vs `accuracy` | Did it learn the shape or the arithmetic? |
| `hack_rate` | Fraction of well-formed wrong answers — the reward-hacking curve |
| `silent_group_frac` | Share of groups with no reward variance (wasted compute) |
| `grad_norm` | Whether a step carried signal at all (`loss` is ~0 by construction for `seq_mean`) |
| `entropy` | Collapse detector; watch it against `kl` when `--kl_coeff 0` |
| `completion_len` | Length drift / reward-shaping side effects |

## Ablation grid (fill in from your runs)

| Run | Change | eval acc | format | hack | silent | entropy @ end |
|-----|--------|----------|--------|------|--------|---------------|
| SFT only | baseline, no RL | | | | — | |
| DPO only | offline pairs from the same env | | | | — | |
| GRPO | default (`seq_mean`, std-norm, no KL) | | | | | |
| GRPO + KL | `--kl_coeff 0.02` | | | | | |
| Dr. GRPO | `--normalize_advantage_std False --aggregation dr_grpo` | | | | | |
| DAPO-lite | `--filter_zero_variance True --clip_eps_high 0.28 --aggregation token_mean` | | | | | |
| Format-only reward | `--format_weight 0.2 --accuracy off` (edit env weights) | | | | | |
| Lenient parse | `--strict_answer False` | | | | | |

Questions worth answering with this grid, in rough order of how interesting the answer is:

1. Does accuracy move at all, or does the policy only climb the format reward?
2. How large is `silent_group_frac`, and does dropping those groups (`--filter_zero_variance`)
   buy real throughput at the same step count?
3. Does the std divisor in the advantage matter at this scale (GRPO vs Dr. GRPO)?
4. Without a KL anchor, does entropy collapse before accuracy improves?
5. Does `--strict_answer False` reveal correct answers hidden behind format failures?

## Recorded CPU smoke (not a quality claim)

A 29M policy, cold-started on single-digit arithmetic, then 200 GRPO steps on CPU. This exists to
show the loop runs end to end and that the diagnostics behave as the algorithm predicts — the task
and model are far too small for the numbers to mean anything about capability.

| | value |
|---|---|
| Policy | SFT on 300 single-digit gold-CoT samples, 4 epochs (final loss ≈ 0.37) |
| GRPO | G=8, 4 prompts/step, 200 steps, lr 1e-5, `seq_mean`, no KL |
| Throughput | ≈ 1.8 s/step on CPU |

Observed (see "Results" below for the filled values): format rate saturates near 1.0 almost
immediately while accuracy stays low, so `hack_rate` is high from the start — the model learned the
shape of the answer long before the arithmetic. Steps where `silent_group_frac` hits 1.0 log exactly
`grad_norm = 0`.

### Results

| Step | eval accuracy | eval format rate | eval hack rate |
|------|---------------|------------------|----------------|
| 0 (SFT init) | | | |
| 50 | | | |
| 100 | | | |
| 200 | | | |

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
python sft.py --pretrained_path results/pretrain_final.pth --device cuda:0 --dtype bfloat16 ...
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

## Recorded CPU runs (implementation validation, **not** a quality claim)

Three runs on the CPU dev box. They exist to show the loop runs end to end and that the
diagnostics behave the way the algorithm predicts. A 29M model trained on a few hundred samples
cannot do arithmetic, so none of these numbers say anything about capability.

Reproduce with:

```bash
python3 -m envs.generate_data --split sft  --n 300 --out datasets/arith1_sft.jsonl  --seed 0   --max_digits 1
python3 -m envs.generate_data --split eval --n 40  --out datasets/arith1_eval.jsonl --seed 777 --max_digits 1

# strong cold start (run A) / weak cold start (runs B, C)
python3 sft.py --data_path datasets/arith1_sft.jsonl --epochs 4 --batch_size 16 --learning_rate 5e-4 \
  --max_seq_len 128 --save_dir results_1d   --device cpu --dtype float32
python3 sft.py --data_path datasets/arith1_sft.jsonl --epochs 1 --batch_size 16 --learning_rate 3e-4 \
  --max_seq_len 128 --save_dir results_weak --device cpu --dtype float32

python3 grpo.py --policy_path <sft_final.pth> --env_max_digits 1 --rl_steps 150 \
  --group_size 8 --batch_size 4 --max_new_tokens 48 --max_seq_len 160 --micro_batch_size 8 \
  --eval_path datasets/arith1_eval.jsonl --eval_size 24 --eval_every 25 \
  --device cpu --dtype float32 --save_dir <out>   # + the per-run flags below

python3 analyze_grpo.py results_weak/grpo_metrics.jsonl results_weak_kl/grpo_metrics.jsonl --window 25
```

| Run | Cold start | lr | KL | Throughput |
|-----|-----------|-----|-----|-----------|
| A | 4-epoch SFT (loss ≈ 0.37) | 1e-5 | 0 | ≈ 1.8 s/step |
| B | 1-epoch SFT (loss ≈ 2.1) | 5e-5 | 0 | ≈ 1.9 s/step |
| C | 1-epoch SFT (loss ≈ 2.1) | 5e-5 | 0.02 | ≈ 2.7 s/step (extra ref forward) |

### Run A — too small a step to change behavior

Over 100 steps the greedy eval never moved (`acc` 0.208, `format` 1.000, identical completion
length every time), but the weights did change: mean |Δ| ≈ 1e-4 against a weight scale of ≈ 0.022.
At lr 1e-5 with roughly 40% of groups silent, the update is real but far too small to flip an
argmax. **A flat eval curve is not by itself evidence that the optimizer is broken** — check the
weight delta and `grad_norm` before debugging the algorithm.

### Runs B vs C — the KL anchor is what keeps the run alive

Sampled (temperature 1.0) format rate, averaged over 25-step windows:

| steps | B (no KL) | C (KL 0.02) |
|-------|-----------|-------------|
| 0–24 | 0.161 | 0.255 |
| 25–49 | **0.296** (peak) | 0.475 |
| 50–74 | 0.000 | 0.637 |
| 75–99 | 0.000 | 0.845 |
| 100–124 | 0.000 | 0.885 |
| 125–149 | 0.000 | **0.915** |

Run B collapsed at around step 50 and **never recovered**: reward went to 0 everywhere, so every
group had zero variance, so the advantage was zero, so `grad_norm` was exactly 0 for the remaining
100 steps. Its entropy had fallen from 3.75 to ≈ 1.3 and the policy degenerated into repeating a
single token:

```text
<think><think><think><think><think><gan</think><think>< conclusion</think><answer><think
```

Run C, same learning rate and seed with only `--kl_coeff 0.02` added, held entropy near 2.0 and
climbed monotonically. This is the clearest single result in the repo: **zero reward is an
absorbing state for GRPO**, and the KL term's job here is not gentle regularization, it is
preventing the run from dying.

### What both runs actually learned: the format, and only the format

Accuracy stayed at **0.000 for all 150 steps in every run**, while `hack_rate` tracked
`format_rate` exactly — i.e. every single well-formed completion had a wrong answer. Run C's
greedy eval format rate reached 1.000 by step 150 with these completions:

```text
<think>右</think><answer>理论 government才能ogle</think><answer> exercis store</answer>...
```

The policy maximized the only reward component it could reach. This is the reason the env reports
`accuracy` and `format` separately and derives `hacked_format` from them: a single scalar reward
curve would have looked like steady progress.

Note also that `silent_group_frac` traces a **U shape** in run C: 0.25 early (nothing works, no
variance), 0.02 in the middle (real learning signal), then back to 0.47 at the end — once the format
reward saturates, every rollout scores the same 0.2 again and the gradient dies a second time. At
this model scale the only way out is for the accuracy component to start producing variance, which
29M parameters trained on 300 samples never manage.

### Follow-ups worth running on a GPU

1. A cold start strong enough that accuracy has nonzero variance (bigger model or far more SFT data)
   — without it, GRPO can only optimize format.
2. `--filter_zero_variance True` on run C: the U shape says roughly half the late-run compute is
   producing exactly zero gradient.
3. An entropy floor or `--clip_eps_high` (DAPO clip-higher) as an alternative to the KL anchor, to
   test whether the collapse in run B is specifically a KL problem or a general exploration problem.

# Resume Portfolio Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SpongeBob LLM interview-defensible: fix correctness bugs, add shared train/loss utils, real KD + DPO, CPU tests/CI, and an honest README.

**Architecture:** Keep flat CLI scripts; extract `train_utils.py` and `losses.py`; fix `model.py` KV-cache/mask; rewrite `distill.py` as true KD; add `dpo.py` + preference dataset; pytest + GitHub Actions.

**Tech Stack:** Python 3.12, PyTorch (CPU), transformers, tokenizers, pytest

## Global Constraints

- CPU-only verification; tiny fixtures under `tests/fixtures/`
- No API server / quantization / fake enterprise features
- Claims in README ⊆ implemented reality (~29M default model)
- Teacher/student/ref models share architecture + tokenizer vocab
- TDD where new behavior is introduced; do not regress the 16+ existing unit tests

## File Map

| File | Responsibility | Status (as of plan refresh) |
|------|----------------|------------------------------|
| `train_utils.py` | seed, AMP/scaler, LR, checkpoint I/O | Done (committed) |
| `model.py` | Fixed KV-cache / RoPE / attention_mask | Done (committed) |
| `chat.py` / `eval_ppl.py` | `load_weights`, `repetition_penalty` | Done (committed) |
| `tests/test_*.py` + fixtures | CPU unit tests | Partial — losses tests uncommitted |
| `losses.py` | masked CE, KD, DPO, sequence_logprobs | Implemented, **uncommitted** |
| `distill.py` | Real KD CLI | Rewritten, **uncommitted** |
| `dpo.py` | DPO CLI | Implemented, **uncommitted** |
| `dataset.py` | SFT mask fix + PreferenceDataset | Modified, **uncommitted** |
| `pretrain.py` / `SFT.py` | Thin loops via train_utils | Modified, **uncommitted** |
| `README.md` / `docs/experiments.md` | Honest docs | **uncommitted** |
| `LICENSE` / `.github/workflows/ci.yml` | MIT + CI | **Missing** |
| `AGENTS.md` / `requirements.txt` | Agent notes + pytest dep | Needs update |

---

### Task 1: `train_utils` + checkpoint roundtrip — DONE

**Files:** `train_utils.py`, `tests/test_checkpoint.py`

**Interfaces produced:**
- `set_seed(seed: int) -> None`
- `str2bool(v) -> bool`
- `build_autocast_scaler(device: str, dtype: str) -> tuple[context, GradScaler|None]`
- `get_lr(step: int, total_steps: int, lr: float, warmup_ratio: float = 0.1) -> float`
- `save_checkpoint(path, model, optimizer, scaler, epoch, step, global_step, loss, config) -> None`
- `load_weights(path, model, device, strict=False) -> dict`
- `load_train_state(checkpoint, optimizer, scaler) -> tuple[epoch, step, global_step, loss]`

- [x] Implement + tests + commit `cf2b3db`

---

### Task 2: Model attention / KV-cache / RoPE / attention_mask — DONE

**Files:** `model.py`, `tests/test_attention_kv.py`

- [x] Prefill / decode+cache / multi-token+cache / GQA / RoPE bounds + commit `2fce07a`

---

### Task 3: Dataset mask + chat/eval loading — DONE (committed core)

**Files:** `dataset.py` (mask), `chat.py`, `eval_ppl.py`, `tests/fixtures/*`, `tests/test_datasets.py`

- [x] Commit `ac9ba89` (fixtures + chat/eval + initial dataset tests)
- [ ] **Step (carry into Task 5):** Ensure working-tree `dataset.py` PreferenceDataset + updated `test_datasets.py` are included when committing KD/DPO work

---

### Task 4: `losses.py` + real KD `distill.py` — IN PROGRESS (code present, uncommitted)

**Files:**
- Create/keep: `losses.py`
- Create/keep: `tests/test_losses.py`
- Rewrite: `distill.py`

**Interfaces:**
- `masked_cross_entropy(logits, targets, mask) -> Tensor`
- `kd_loss(student_logits, teacher_logits, temperature: float, mask=None) -> Tensor`
- `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta: float) -> Tensor`
- `sequence_logprobs(logits, targets, mask) -> Tensor`  # shape (B,), token-sum

**KD objective:** `(1-alpha)*ce + alpha*kd_loss` on SFT assistant masks; frozen teacher.

- [ ] **Step 1: Confirm unit tests pass**

```bash
python -m pytest tests/test_losses.py -v
```

Expected: all PASS (finite KD, T sensitivity, alpha blend, DPO monotonic toy).

- [ ] **Step 2: Confirm `distill.py` uses real KD** (no special-token fake weighting)

Verify `distill.py` imports `kd_loss`, `masked_cross_entropy`, loads teacher with `requires_grad_(False)`, and saves `distill_final.pth`.

- [ ] **Step 3: CPU smoke (1 epoch on fixture)**

```bash
# Assumes a student/teacher weight exists; if not, run a 1-step SFT first:
python SFT.py --data_path tests/fixtures/sft_tiny.jsonl \
  --pretrained_path "" --epochs 1 --batch_size 2 --max_seq_len 128 \
  --save_dir results --device cpu --dtype float32 --num_workers 0 || true
```

If `SFT.py` requires pretrained file, initialize by short pretrain then SFT:

```bash
python pretrain.py --data_path tests/fixtures/pretrain_tiny.jsonl \
  --epochs 1 --batch_size 2 --max_seq_len 128 --save_dir results \
  --device cpu --dtype float32 --num_workers 0 --log_step 1 --save_step 9999

python SFT.py --data_path tests/fixtures/sft_tiny.jsonl \
  --pretrained_path results/pretrain_final.pth \
  --epochs 1 --batch_size 2 --max_seq_len 128 --save_dir results \
  --device cpu --dtype float32 --num_workers 0 --log_step 1 --save_step 9999

python distill.py --data_path tests/fixtures/sft_tiny.jsonl \
  --teacher_path results/sft_final.pth --student_path results/sft_final.pth \
  --alpha 0.5 --temperature 2.0 --epochs 1 --batch_size 2 --max_seq_len 128 \
  --save_dir results --device cpu --dtype float32 --num_workers 0 --log_step 1
```

Expected: prints finite `loss` / `ce` / `kd`; writes `results/distill_final.pth`.

- [ ] **Step 4: Commit**

```bash
git add losses.py tests/test_losses.py distill.py
git commit -m "feat: real knowledge distillation and shared loss helpers"
```

---

### Task 5: PreferenceDataset + `dpo.py` — IN PROGRESS (code present, uncommitted)

**Files:**
- Modify: `dataset.py` (PreferenceDataset)
- Create: `dpo.py`
- Fixture: `tests/fixtures/preference_tiny.jsonl` (already present)
- Extend: `tests/test_datasets.py`

**Canonical preference JSONL:**
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

Dataset returns `(cX, cY, cMask, rX, rY, rMask)` with assistant-only masks via chat template.

- [ ] **Step 1: Run dataset + loss tests**

```bash
python -m pytest tests/test_datasets.py tests/test_losses.py -v
```

Expected: PASS including preference response-mask coverage.

- [ ] **Step 2: CPU smoke DPO**

```bash
python dpo.py --data_path tests/fixtures/preference_tiny.jsonl \
  --policy_path results/sft_final.pth --beta 0.1 \
  --epochs 1 --batch_size 2 --max_seq_len 128 --save_dir results \
  --device cpu --dtype float32 --num_workers 0 --log_step 1
```

Expected: finite `dpo_loss`; writes `results/dpo_final.pth`.

- [ ] **Step 3: Commit**

```bash
git add dataset.py dpo.py tests/test_datasets.py tests/fixtures/preference_tiny.jsonl
git commit -m "feat: add DPO training stage and preference dataset"
```

---

### Task 6: Thin pretrain/SFT + full e2e CPU smoke — IN PROGRESS

**Files:** `pretrain.py`, `SFT.py` (working tree already refactored onto `train_utils`)

**Requirements for both scripts:**
- Use `set_seed`, `build_autocast_scaler`, `get_lr`, `save_checkpoint`, `load_weights`, `load_train_state`
- `global_step` increments on **optimizer updates** only; LR uses that counter
- GradScaler only when dtype is fp16
- Final weights: `pretrain_final.pth` / `sft_final.pth` as raw `state_dict`

- [ ] **Step 1: Commit refactors**

```bash
git add pretrain.py SFT.py
git commit -m "refactor: share train_utils across pretrain and SFT"
```

- [ ] **Step 2: Full pipeline smoke on fixtures**

```bash
rm -rf results && mkdir -p results
python pretrain.py --data_path tests/fixtures/pretrain_tiny.jsonl --epochs 1 --batch_size 2 --max_seq_len 128 --save_dir results --device cpu --dtype float32 --num_workers 0 --log_step 1 --save_step 9999
python SFT.py --data_path tests/fixtures/sft_tiny.jsonl --pretrained_path results/pretrain_final.pth --epochs 1 --batch_size 2 --max_seq_len 128 --save_dir results --device cpu --dtype float32 --num_workers 0 --log_step 1 --save_step 9999
python distill.py --data_path tests/fixtures/sft_tiny.jsonl --teacher_path results/sft_final.pth --student_path results/sft_final.pth --alpha 0.5 --temperature 2.0 --epochs 1 --batch_size 2 --max_seq_len 128 --save_dir results --device cpu --dtype float32 --num_workers 0 --log_step 1
python dpo.py --data_path tests/fixtures/preference_tiny.jsonl --policy_path results/sft_final.pth --beta 0.1 --epochs 1 --batch_size 2 --max_seq_len 128 --save_dir results --device cpu --dtype float32 --num_workers 0 --log_step 1
python eval_ppl.py --model_path results/pretrain_final.pth --dataset_path tests/fixtures/pretrain_tiny.jsonl --max_seq_len 128 --device cpu --batch_size 2
printf '海绵宝宝喜欢做什么？\nquit\n' | python chat.py --save_dir results --model_mode 1 --device cpu --max_new_tokens 32
# Also verify checkpoint (not only final) loads:
printf 'hi\nquit\n' | python chat.py --save_dir results --model_mode 1 --device cpu --max_new_tokens 8
python -m pytest tests/ -q
```

Expected: each stage completes; chat loads; pytest all green.

- [ ] **Step 3: Commit any smoke fixes** if needed

---

### Task 7: Docs, LICENSE, CI, AGENTS.md

**Files:**
- Rewrite/keep: `README.md` (honest version already in working tree)
- Create: `docs/experiments.md` (present, uncommitted)
- Create: `LICENSE` (MIT text)
- Create: `.github/workflows/ci.yml`
- Modify: `AGENTS.md`, `requirements.txt` (add `pytest`)

- [ ] **Step 1: Add MIT LICENSE**

Create `/workspace/LICENSE` with standard MIT license body; copyright holder `irroca` (repo owner) or omit name line as `Copyright (c) 2026 SpongeBob LLM contributors`.

- [ ] **Step 2: Add CI workflow** `.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install deps
        run: |
          pip install --index-url https://download.pytorch.org/whl/cpu torch
          pip install -r requirements.txt
          pip install pytest
      - name: Test
        run: python -m pytest tests/ -q
```

- [ ] **Step 3: Update `requirements.txt`** to include `pytest>=8.0` (or keep CI-only install — prefer listing in requirements for local `pip install -r`).

- [ ] **Step 4: Update `AGENTS.md`** Cursor Cloud section: mention `pytest`, `distill.py` real KD, `dpo.py`, fixtures path `tests/fixtures/`, and that README is source of truth for commands.

- [ ] **Step 5: Finalize README check** — no claims of API/quant/billion-params/enterprise; document KD/DPO formulas; point to `docs/experiments.md`; include resume-oriented 3–5 bullet suggestions (Chinese).

Example resume bullets to include under a short 「简历表述建议」 section:

```text
- 从零实现 ~29M Llama 风格 LM（RoPE / RMSNorm / SwiGLU / 可选 GQA / 权重共享）
- 完整 Pretrain→SFT→KD→DPO 流水线；统一 checkpoint 加载与 CPU 单测覆盖 KV-cache
- 实现温度 KL 知识蒸馏与 DPO（response token-sum log-prob + 冻结 reference）
```

- [ ] **Step 6: Commit docs/CI**

```bash
git add README.md docs/experiments.md LICENSE .github/workflows/ci.yml AGENTS.md requirements.txt
git commit -m "docs: honest README, experiments template, MIT license, and CI"
```

- [ ] **Step 7: Push + update PR** on `cursor/resume-portfolio-optimization-7827`

---

## Spec Coverage Checklist

- [x] KV-cache mask → Task 2
- [x] RoPE bounds → Task 2
- [x] attention_mask → Task 2/3
- [x] train_utils / AMP / LR → Task 1/6
- [x] load_weights dual format → Task 1/3
- [x] SFT mask off-by-one → Task 3
- [ ] Real KD → Task 4 (code ready, needs smoke + commit)
- [ ] DPO → Task 5 (code ready, needs smoke + commit)
- [ ] Thin pretrain/SFT + e2e → Task 6
- [ ] Honest README / experiments / LICENSE / CI / AGENTS → Task 7
- [x] No vaporware → Global Constraints + README rewrite

## Self-Review Notes

- Plan matches approved spec `docs/superpowers/specs/2026-08-10-resume-portfolio-optimization-design.md`.
- Remaining work is primarily **verify + commit + LICENSE/CI/AGENTS**, not greenfield design.
- Do not invent API/quant features while finishing Task 7.

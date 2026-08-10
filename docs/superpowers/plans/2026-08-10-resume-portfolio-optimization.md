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
- TDD: failing test before production code for each behavior

## File Map

| File | Responsibility |
|------|----------------|
| `train_utils.py` | seed, AMP/scaler, LR, checkpoint save/load, `load_weights` |
| `losses.py` | masked CE, KD loss, DPO loss |
| `model.py` | Transformer + fixed attention/generate |
| `dataset.py` | Pretrain, SFT (mask fix), PreferenceDataset |
| `pretrain.py` / `SFT.py` | Thin training loops using train_utils |
| `distill.py` | Real KD CLI |
| `dpo.py` | DPO CLI |
| `chat.py` / `eval_ppl.py` | Fixed loading + generation args |
| `tests/*` | CPU unit tests |
| `README.md` / `docs/experiments.md` / `LICENSE` / `.github/workflows/ci.yml` | Docs + CI |

---

### Task 1: `train_utils` + checkpoint roundtrip

**Files:**
- Create: `train_utils.py`
- Create: `tests/test_checkpoint.py`
- Modify: (wire later in Task 6; tests import utils directly)

**Interfaces:**
- Produces:
  - `set_seed(seed: int) -> None`
  - `str2bool(v) -> bool`
  - `build_autocast_scaler(device: str, dtype: str) -> tuple[contextmanager_or_null, GradScaler|None]`
  - `get_lr(step: int, total_steps: int, lr: float, warmup_ratio: float = 0.1) -> float`
  - `save_checkpoint(path, model, optimizer, scaler, epoch, step, global_step, loss, config) -> None`
  - `load_weights(path, model, device, strict=False) -> dict`  # returns raw checkpoint dict if any
  - `load_train_state(checkpoint, optimizer, scaler) -> tuple[epoch, step, global_step, loss]`

- [ ] **Step 1: Write failing tests** in `tests/test_checkpoint.py` for `get_lr` continuity, `load_weights` accepting both formats, scaler fp16-only policy.

- [ ] **Step 2: Run** `python -m pytest tests/test_checkpoint.py -v` → FAIL (import error)

- [ ] **Step 3: Implement** `train_utils.py`

- [ ] **Step 4: pytest PASS**

- [ ] **Step 5: Commit** `feat: add train_utils with robust checkpoint loading`

---

### Task 2: Model attention / KV-cache / RoPE / attention_mask

**Files:**
- Modify: `model.py`
- Create: `tests/test_attention_kv.py`

**Interfaces:**
- `Attention.forward` uses kv_len for mask
- `SpongeBob.forward(..., attention_mask: Optional[Tensor] = None)`
- Clear error if `start_pos + seq_len > pos_cis.shape[0]`

- [ ] **Step 1: Failing tests** — prefill; decode+cache; multi-token+cache; GQA shapes; long RoPE raises

- [ ] **Step 2: pytest FAIL** (multi-token+cache currently errors)

- [ ] **Step 3: Fix model.py**

- [ ] **Step 4: pytest PASS**

- [ ] **Step 5: Commit** `fix: correct KV-cache causal mask and RoPE bounds`

---

### Task 3: Dataset mask + chat/eval loading fixes

**Files:**
- Modify: `dataset.py`, `chat.py`, `eval_ppl.py`
- Create: `tests/test_datasets.py`
- Create: `tests/fixtures/sft_tiny.jsonl`, `tests/fixtures/pretrain_tiny.jsonl`

- [ ] **Step 1: Failing test** — SFT loss mask includes first assistant content token

- [ ] **Step 2: Fix `_generate_loss_mask`**; fix chat `repetition_penalty`; use `load_weights`; token truncate; `str2bool` / store_true; eval_ppl uses `load_weights` + pass `attention_mask` into model if supported

- [ ] **Step 3: pytest PASS + commit** `fix: SFT loss mask and checkpoint-aware chat/eval`

---

### Task 4: `losses.py` + real KD `distill.py`

**Files:**
- Create: `losses.py`
- Create: `tests/test_losses.py`
- Rewrite: `distill.py`

**Interfaces:**
- `masked_cross_entropy(logits, targets, mask) -> Tensor`
- `kd_loss(student_logits, teacher_logits, temperature: float) -> Tensor`  # KL * T^2, mean over unmasked positions via optional mask
- `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta: float) -> Tensor`

KD total: `(1-alpha)*ce + alpha*kd_loss`

- [ ] **Step 1: Failing loss unit tests** (KD finite; alpha/T sensitivity; DPO toy monotonicity)

- [ ] **Step 2: Implement losses.py**

- [ ] **Step 3: Rewrite distill.py** as teacher-student KD using SFTDataset + train_utils

- [ ] **Step 4: Smoke 1–2 steps on fixture; commit** `feat: real knowledge distillation + loss helpers`

---

### Task 5: PreferenceDataset + `dpo.py`

**Files:**
- Modify: `dataset.py` (add PreferenceDataset)
- Create: `dpo.py`
- Create: `tests/fixtures/preference_tiny.jsonl`
- Extend: `tests/test_datasets.py`

- [ ] **Step 1: Failing tests** for preference response masks

- [ ] **Step 2: Implement PreferenceDataset + dpo.py** (frozen ref, beta, save `dpo_final.pth`)

- [ ] **Step 3: Smoke + commit** `feat: add DPO training stage`

---

### Task 6: Thin pretrain/SFT + e2e CPU smoke

**Files:**
- Modify: `pretrain.py`, `SFT.py` to use train_utils (LR, scaler, checkpoint, load_weights)

- [ ] **Step 1: Refactor loops** — optimizer-update-based global_step for LR; fp16-only scaler

- [ ] **Step 2: Run tiny smoke** pretrain→SFT→KD→DPO on fixtures (1 epoch, small batch, cpu)

- [ ] **Step 3: Commit** `refactor: share train_utils across training scripts`

---

### Task 7: Docs, LICENSE, CI, AGENTS.md

**Files:**
- Rewrite: `README.md`
- Create: `docs/experiments.md`, `LICENSE`, `.github/workflows/ci.yml`
- Modify: `AGENTS.md`, `requirements.txt` (add pytest)

- [ ] **Step 1: Honest README + experiments template + MIT LICENSE**

- [ ] **Step 2: CI workflow** installs CPU torch + deps, runs pytest

- [ ] **Step 3: Commit + open/update PR**

---

## Spec Coverage Checklist

- [x] KV-cache mask → Task 2
- [x] RoPE bounds → Task 2
- [x] attention_mask → Task 2/3
- [x] train_utils / AMP / LR → Task 1/6
- [x] load_weights dual format → Task 1/3
- [x] SFT mask off-by-one → Task 3
- [x] Real KD → Task 4
- [x] DPO → Task 5
- [x] Tests + CI → Tasks 1–5, 7
- [x] Honest README / experiments / LICENSE → Task 7
- [x] No vaporware → Global Constraints

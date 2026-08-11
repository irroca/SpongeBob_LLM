# Solidify Current Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make existing Pretrain/SFT/KD/DPO/eval/chat features interview-solid: fix mask/accum/PPL/generate/attn_mask bugs, thin shared helpers, thicker tests, honest docs.

**Architecture:** Keep flat CLIs; extend `train_utils.py` with `optimizer_step` + accum flush + common argparse/wandb; fix `dataset.assistant_loss_mask` and `model.generate`/attention; no Trainer class.

**Tech Stack:** Python 3.12, PyTorch CPU, transformers, pytest

## Global Constraints

- CPU-only verification; fixtures under `tests/fixtures/`
- No new research features (MoE/MLA/speculative decoding/etc.)
- No full Trainer framework / Hydra / distributed / serving
- Resume+shuffle: document caveat only (do not fake perfect DataLoader resume)
- TDD for each bugfix; do not regress existing 18+ tests
- Branch: `cursor/solidify-current-features-7827`

## File Map

| File | Change |
|------|--------|
| `dataset.py` | Fix `assistant_loss_mask` rules |
| `train_utils.py` | `optimizer_step`, flush helper, `add_common_train_args`, `init_wandb_if_needed` |
| `pretrain.py` / `SFT.py` / `distill.py` / `dpo.py` | Use step/flush; unify CLI/wandb; real epoch loss |
| `model.py` | Batch generate + pad_token_id; attn_mask+cache pad direction |
| `eval_ppl.py` | BOS/EOS wrap like PretrainDataset |
| `tests/test_datasets.py` | Mask regression cases |
| `tests/test_train_utils.py` | New — flush clears grads |
| `tests/test_generate.py` | New — B=1/B=2 |
| `tests/test_attention_kv.py` | Cache + chunk mask |
| `tests/test_eval_ppl.py` | New — wrapping helper |
| `README.md` / `AGENTS.md` | Limits: resume, PPL, generate batching |

---

### Task 1: Fix `assistant_loss_mask` + dataset

**Files:** `dataset.py`, `tests/test_datasets.py`

**Normative mask rules:**
1. Mark from first assistant content token through end of `</s>\n` span (inclusive of EOS span tokens).
2. Never mark `pad_token_id` positions (caller must pass pad id — extend signature to `assistant_loss_mask(..., pad_token_id=0)` or detect pads if ids provided).
3. If no EOS found (truncated): mark content start .. last non-pad index only.

**Preferred signature:**
```python
def assistant_loss_mask(input_ids, bos_id, eos_id, max_length, pad_token_id=0):
```

Update `SFTDataset` / `PreferenceDataset` call sites to pass `tokenizer.pad_token_id`.

- [ ] **Step 1:** Add failing tests in `tests/test_datasets.py`:
  - After normal reply, positions after EOS that are pad must be 0 (currently fails with `+1`).
  - Truncated sequence without EOS: trailing pads are 0; some assistant content still 1.
  - Keep existing first-assistant-token == 1 assertions green after fix.

- [ ] **Step 2:** `pytest tests/test_datasets.py -v` → expect RED on new pad-leak tests.

- [ ] **Step 3:** Implement fix — replace
  `for j in range(start, min(end + len(eos_id) + 1, max_length)):`
  with logic that marks `[start, end+len(eos_id))` capped by length, then zeros any `input_ids[j]==pad_token_id`, and for missing EOS uses last non-pad.

- [ ] **Step 4:** pytest PASS; commit `fix: prevent pad tokens in assistant loss masks`

---

### Task 2: `optimizer_step` + epoch accum flush

**Files:** `train_utils.py`, `pretrain.py`, `SFT.py`, `distill.py`, `dpo.py`, `tests/test_train_utils.py`

**Interfaces:**
```python
def optimizer_step(model, optimizer, scaler, grad_clip: float) -> None:
    """unscale (if scaler), clip, step, update, zero_grad."""

def flush_pending_grads(model, optimizer, scaler, grad_clip: float, pending: bool) -> bool:
    """If pending, call optimizer_step and return False (cleared). Else return pending."""
```

Training loop pattern:
- `pending = False`
- each micro-batch after backward: `pending = True`
- on accum boundary: `optimizer_step(...); pending = False; global_step += 1`
- end of epoch: `if pending: optimizer_step(...); pending=False; global_step += 1`

- [ ] **Step 1:** Failing test: create tiny param, run N backwards without step where N % accum != 0, assert `.grad` nonzero; call `flush_pending_grads(..., pending=True)`; assert grads cleared / None after zero_grad.

- [ ] **Step 2:** Implement helpers; refactor four scripts to use them; pass last real loss into epoch `save_checkpoint` (not `0.0`).

- [ ] **Step 3:** pytest PASS; commit `fix: flush leftover grads at epoch end`

---

### Task 3: `eval_ppl` BOS/EOS wrapping

**Files:** `eval_ppl.py`, `tests/test_eval_ppl.py`

Extract helper:
```python
def wrap_pretrain_text(text: str, tokenizer) -> str:
    return f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"
```
Use in `calculate_ppl` / before tokenize.

- [ ] **Step 1:** Test wrap adds bos/eos strings; test that raw vs wrapped token ids differ.

- [ ] **Step 2:** Wire into eval path; commit `fix: align eval_ppl inputs with PretrainDataset`

---

### Task 4: Batch-safe `generate` + `pad_token_id`

**Files:** `model.py`, `tests/test_generate.py`

Behavior:
- `finished` bool tensor `[B]`
- Sample next tokens for unfinished rows only (or sample all then overwrite finished rows with `pad_token_id`)
- Stop when `finished.all()` or hit `max_new_tokens`
- Never call `.item()` on multi-element EOS check

- [ ] **Step 1:** Failing tests — B=2 generate raises today; after fix returns without error; finished rows get pad_token_id.

- [ ] **Step 2:** Implement; pytest PASS; commit `fix: support batched generate with per-sequence EOS`

---

### Task 5: KV-cache + chunk `attention_mask`

**Files:** `model.py` (`Attention.forward`), `tests/test_attention_kv.py`

When `past_key_value` present and `attention_mask.shape[-1] == q_len`:
`key_mask = cat([ones(B, past_len), attention_mask], dim=-1)`.
If `shape[-1] == kv_len`, use as-is. Reject other lengths with clear error.

- [ ] **Step 1:** Failing/expanding test for multitoken+cache+chunk mask.

- [ ] **Step 2:** Fix padding direction; commit `fix: pad attention_mask on the left for KV cache`

---

### Task 6: CLI/wandb unify + docs

**Files:** `train_utils.py` (`add_common_train_args`, `init_wandb_if_needed`), four train scripts, `README.md`, `AGENTS.md`

- Unify `--device` default to `"cuda" if torch.cuda.is_available() else "cpu"`.
- Add wandb flags + logging to `distill.py` / `dpo.py`.
- Document: resume+shuffle caveat; eval_ppl pretrain-style wrapping; generate batching/EOS behavior; known limits.

- [ ] **Step 1:** Implement helpers + wire scripts.

- [ ] **Step 2:** Update README/AGENTS.

- [ ] **Step 3:** `pytest tests/ -q` all green; short fixture smoke optional; commit `chore: unify train CLI/wandb and document limits`

- [ ] **Step 4:** Push + open/update PR for `cursor/solidify-current-features-7827`

---

## Spec Coverage Checklist

- [x] §1.1 mask → Task 1
- [x] §1.2 accum flush → Task 2
- [x] §1.3 eval_ppl → Task 3
- [x] §1.4 generate → Task 4
- [x] §1.5 attn_mask+cache → Task 5
- [x] §2 thin shared / wandb / CLI / epoch loss / docs → Tasks 2 & 6
- [x] §3 tests → Tasks 1–5
- [x] Non-goals respected → Global Constraints

# Solidify Current Features — Design

**Date:** 2026-08-10  
**Status:** Section 1 approved; Section 2 included per user “继续计划”  
**Scope:** Make currently implemented Pretrain / SFT / KD / DPO / eval / chat **interview-solid**.  
**Non-goals:** New LLM research (MoE, MLA, speculative decoding, GQA default change as a “new advance”, etc.).

## Goals

1. Fix remaining correctness bugs in existing features.
2. Add a thin shared training helper layer (no full Trainer framework).
3. Thicken tests for claimed behaviors and known failure modes.
4. Document known limits honestly; small API/CLI cleanup.

## Constraints

- CPU-verifiable with fixtures + unit tests.
- Keep flat CLI entrypoints (`python3 pretrain.py`, etc.).
- YAGNI: no Hydra, no distributed, no serving.

---

## 1. Correctness (approved)

### 1.1 Assistant loss mask (`dataset.assistant_loss_mask`)

**Bugs:** (a) `+1` past EOS marks a pad/`<unk>` position; (b) missing closing `</s>` (truncation) marks all trailing pads.

**Rules (normative):**

1. Locate each `<s>assistant\n` … `</s>\n` span.
2. Mark loss=1 for tokens from first assistant **content** token through the end of the `</s>\n` span (inclusive of EOS tokens that belong to the span).
3. Never mark `pad_token_id` positions as 1.
4. If EOS is not found before end of sequence (truncated): mark from content start through the last **non-pad** index only; do not extend into padding.

Update docstring to match. Expand unit tests for: normal span, trailing pads after EOS, truncated assistant without EOS.

### 1.2 Gradient accumulation epoch flush

In `pretrain.py`, `SFT.py`, `distill.py`, `dpo.py`: after each epoch’s micro-batch loop, if there are leftover accumulated grads (`len(loader) % accumulation_steps != 0` or any pending backward without step), perform one optimizer step and `zero_grad`.

Extract to `train_utils.py`:

- `optimizer_step(model, optimizer, scaler, grad_clip) -> None`
- `maybe_flush_accum(model, optimizer, scaler, grad_clip, pending: bool) -> bool`  
  or simply always call `flush_pending_grads(...)` at epoch end when `accumulation_steps > 1` and last window incomplete — tracked via a `grads_pending` flag set on every backward and cleared on step.

`global_step` increments on flush steps too (they are real optimizer updates).

### 1.3 `eval_ppl` distribution match

When evaluating pretrain-style text, wrap each sample as  
`f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"`  
matching `PretrainDataset`. Document that SFT/chat-format PPL is out of scope for this script unless `--text_field` content is already raw LM text.

Add a small test or fixture assertion that wrapped encoding differs from raw and that the eval path uses wrapping.

### 1.4 `generate` batching

- Support `batch_size >= 1`.
- Per-sequence finished flags; finished rows get `pad_token_id` appended (or stay fixed) and are excluded from active sampling; stop when all finished or `max_new_tokens` reached.
- Use `pad_token_id` argument for finished-row fill (no longer dead).
- EOS check must not call `.item()` on multi-element tensors.
- Tests: B=1 completes; B=2 does not raise; finished sequences stop growing meaningfully (pad or frozen).

### 1.5 KV-cache + padding `attention_mask`

When `past_key_value` is used and caller passes an `attention_mask` for the **current chunk** only (`q_len`), pad the mask on the **left** (past keys = visible/`1`) and apply the provided mask to the **new** key positions. If caller passes full `kv_len`, use as-is.

Test: multi-token continuation with cache + chunk mask; pad positions get `-inf` on scores for those keys.

---

## 2. Thin shared layer + consistency

### 2.1 `train_utils` additions (not a Trainer class)

| Helper | Purpose |
|--------|---------|
| `optimizer_step(...)` | unscale / clip / step / update / zero_grad |
| epoch-end flush helper | see §1.2 |
| `add_common_train_args(parser)` | `--device` default `cuda` if available else `cpu` (unify away from mixed `cuda:0` / `cuda`); `--dtype`; `--seed`; optional `--use_wandb` / `--wandb_project` |
| `init_wandb_if_needed(args)` | lazy swanlab import; used by all four train scripts |

Wire `--use_wandb` into `distill.py` and `dpo.py` the same way as pretrain/SFT (log scalar loss + lr + global_step).

### 2.2 Resume semantics (honest, not fake-perfect)

**Do not** implement full deterministic DataLoader resume in this iteration (high cost).

Instead:

- Document in README + AGENTS: with `shuffle=True`, `--resume_from` restores model/opt/scaler/step counters but **does not** replay the exact remaining shuffle order; suitable for crash recovery, not bitwise reproducibility.
- Optionally set `shuffle=False` when `resume_from` is set **or** document that users should pass a fixed seed and accept approximate resume — prefer **document only** to avoid surprising behavior change.

Add a unit/integration test that `load_train_state` roundtrips epoch/step/global_step (already partly covered); add a tiny test that epoch-end flush clears `.grad` (can be a focused train_utils or loop-level test without full CLI).

### 2.3 Epoch checkpoint loss

Pass the last logged training loss (or running average) into epoch-end `save_checkpoint` instead of hardcoded `0.0`.

### 2.4 CLI cleanup

- Unify `--device` default via `add_common_train_args`.
- Keep `chat.py` / `eval_ppl.py` consistent (`cpu`/`cuda` autodetection string without forcing `:0` unless user passes it).

---

## 3. Tests to add/extend

| Area | File | Cases |
|------|------|-------|
| Mask | `tests/test_datasets.py` | no pad after EOS; truncated no-EOS no pad leak; first assistant token still 1 |
| Accum flush | `tests/test_train_utils.py` (new) or extend checkpoint tests | pending grads cleared by flush helper |
| PPL wrap | `tests/test_eval_ppl.py` (new) or small helper test | bos/eos wrapping applied |
| Generate | `tests/test_generate.py` (new) | B=1; B=2 no crash; pad_token_id used for finished rows |
| Attn+cache | `tests/test_attention_kv.py` | cache continuation + chunk attention_mask |

Keep suite CPU-fast; no GPU required.

---

## 4. Documentation

- README 「局限」: resume+shuffle caveat; eval_ppl is pretrain-style LM PPL; generate batching behavior summary.
- AGENTS.md: same caveats for cloud agents; point to tests for regression.
- Short 「本轮加固」 note optional in README or `docs/superpowers/specs/` only (avoid changelog spam in README).

---

## 5. Implementation order

1. Fix `assistant_loss_mask` + dataset tests  
2. `train_utils` optimizer_step + flush; wire four scripts; accum test  
3. `eval_ppl` wrapping + test  
4. `generate` batching + tests  
5. attention_mask + cache fix + test  
6. wandb/CLI unify; epoch loss; docs  
7. Full pytest + short fixture smoke  

## 6. Success criteria

- All new + existing tests pass on CPU.  
- Reproduced bugs from audit (pad-in-loss, accum leak, eval without BOS, generate B=2 crash) have failing-then-passing tests or explicit regression tests.  
- No new research features.  
- Docs match behavior (especially resume and PPL).

## 7. Out of scope (next iteration)

MoE, MLA, speculative decoding, FlashAttention binding, multi-GPU, RLHF beyond DPO, changing default architecture size.

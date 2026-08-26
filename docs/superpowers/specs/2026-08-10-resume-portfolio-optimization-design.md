# SpongeBob LLM Resume Portfolio Optimization — Design

**Date:** 2026-08-10  
**Status:** Approved direction (Approach 3 + KD + DPO; CPU/synthetic verification)  
**Audience:** CS/AI grad student targeting LLM/AI internships and junior algo roles

## Goals

Make this personal project **interview-defensible** and **resume-honest**:

1. Fix correctness bugs that would fail a deep dive.
2. Reduce training-script duplication with a small shared layer.
3. Replace fake “distill” with **real Knowledge Distillation**.
4. Add a minimal but correct **DPO** stage.
5. Add CPU unit tests + CI.
6. Rewrite README / experiment docs to match reality (~29M params, CPU smoke, GPU recipe left for the owner).

## Non-Goals (YAGNI)

- REST API server, quantization, TensorBoard façade, “enterprise monitoring”
- Full Hydra/YAML config system or large installable monorepo packaging
- Claiming billion-parameter scale or multi-GPU without implementing it
- Training on real large corpora or publishing GPU metrics in this environment (CPU-only)

## Constraints

- Cloud/dev environment is **CPU-only**; verification = unit tests + tiny synthetic JSONL smoke runs.
- Keep CLI script entrypoints (`python pretrain.py ...`) so the project stays easy to demo.
- Chinese-first README is OK (target market); keep technical terms precise.

---

## 1. Layout & Module Boundaries

```text
spongebob_llm/
├── Config.py              # LLMConfig (stronger validation)
├── model.py               # Fixed Attention/KV-cache/generate/optional attn mask
├── dataset.py             # Pretrain / SFT / Preference(DPO) datasets
├── train_utils.py         # NEW: seed, AMP, LR, checkpoint I/O, load_weights
├── losses.py              # NEW: CE helpers, KD loss, DPO loss (unit-tested)
├── pretrain.py            # Thin entry
├── SFT.py                 # Thin entry
├── distill.py             # REWRITE: real teacher→student KD
├── dpo.py                 # NEW: DPO training entry
├── chat.py                # Fixed kwargs + weight loading
├── eval_ppl.py            # Fixed weight loading + padding-aware eval
├── tests/                 # NEW
│   ├── test_attention_kv.py
│   ├── test_losses.py
│   ├── test_checkpoint.py
│   └── test_datasets.py
├── .github/workflows/ci.yml
├── requirements.txt
├── README.md              # Honest rewrite
├── docs/experiments.md    # GPU experiment template for the owner
└── spongebob_tokenizer/   # Unchanged
```

**Responsibility split**

| Module | Owns |
|--------|------|
| `model.py` | Network + generation correctness |
| `losses.py` | Algorithm formulas, DataLoader-agnostic |
| `train_utils.py` | Shared training engineering |
| `distill.py` / `dpo.py` | Stage CLI + wiring only |

---

## 2. Model Correctness & Training Hardening

### 2.1 Attention / KV-cache

- Causal mask must be shaped for **query length × key length**, not `seq_len × seq_len` where `seq_len` is only the current query chunk.
- Preferred implementation: `F.scaled_dot_product_attention` with correct `attn_mask` / `is_causal` behavior for prefill vs decode.
- Required passing cases:
  1. Prefill, no cache
  2. Single-token decode with cache
  3. Multi-token continuation with cache

### 2.2 RoPE length handling

- Remove broken “dynamic extend” that truncates `pos_cis` without truncating hidden states.
- If `start_pos + seq_len > max_seq_len`: raise a clear error or truncate **inputs** to fit; never desync RoPE vs tokens.

### 2.3 Optional `attention_mask`

- `forward(..., attention_mask=None)` applies padding mask on the key side so batched PPL with padding is not polluted.

### 2.4 `generate`

- Keep streaming generator API.
- Honor `repetition_penalty`, temperature, top_p, eos stop.
- `chat.py` must pass `repetition_penalty=` (not `rp=`).

### 2.5 `train_utils.py`

| API | Behavior |
|-----|----------|
| `set_seed(seed)` | torch (+ numpy if used) |
| `build_autocast_scaler(device, dtype)` | GradScaler **only for fp16**; autocast dtype matches `--dtype` |
| `get_lr(step, total_steps, lr, warmup_ratio=0.1)` | Warmup starts > 0 at step 1; continuous join into cosine; `step` counts **optimizer updates** |
| `save_checkpoint(...)` | Dict with `model_state_dict`, opt, scaler, epoch, step, config, loss |
| `load_weights(path, model, device, strict=False)` | Accepts raw `state_dict` **or** training checkpoint dict |
| `load_train_state(...)` | Resume opt/scaler/epoch/step |

All of `pretrain.py`, `SFT.py`, `distill.py`, `dpo.py`, `chat.py`, `eval_ppl.py` use `load_weights` for model weights.

### 2.6 Dataset / CLI fixes

- Fix SFT assistant loss-mask off-by-one (first assistant content token included).
- Replace `type=bool` argparse with `store_true` or explicit `str2bool`.
- Token-based prompt truncation in `chat.py` (not character slicing).
- Document / set sensible `--save_dir` default (`results`).

---

## 3. Real Knowledge Distillation

### 3.1 Algorithm

Student \(S\), frozen teacher \(T\). Data: **same JSONL as SFT** (`conversations`), with assistant-only token mask when computing CE and KL.

For each batch:

\[
\mathcal{L} = (1-\alpha)\,\mathcal{L}_{\mathrm{CE}}(S(x), y)
+ \alpha \, T^2 \, \mathrm{KL}\big(\mathrm{softmax}(z_T/T) \,\|\, \mathrm{log\_softmax}(z_S/T)\big)
\]

- Temperature \(T \ge 1\), mix weight \(\alpha \in [0,1]\).
- KL computed over vocabulary at masked (assistant) positions only; pad positions excluded.
- Teacher loaded from `--teacher_path`, `eval()` + `torch.no_grad()`.
- Student initialized from `--student_path` (typically SFT checkpoint).
- Typical demo wiring: teacher = larger or earlier SFT/pretrain ckpt; student = smaller or same-size SFT ckpt. Same architecture by default (config must match checkpoint); document that teacher/student must share `vocab_size` and tokenizer.

### 3.2 CLI (`distill.py`)

Essential args: data path, teacher/student paths, `alpha`, `temperature`, seq len, batch size, lr, dtype, device, save dir, resume.

Remove the previous “special token weight” fake-distill behavior (or do not expose it).

### 3.3 CPU smoke

Tiny synthetic SFT-format JSONL; 1–2 update steps; assert finite loss and that KD term changes when `temperature`/`alpha` change (unit test on `losses.kd_loss`).

---

## 4. DPO

### 4.1 Data format (`PreferenceDataset`)

**Canonical JSONL** (one format only — no dual schemas):

```json
{"prompt": "用户问题", "chosen": "更好的回答", "rejected": "更差的回答"}
```

Construction (fixed in code + README):

1. Build chat messages for chosen/rejected separately using the tokenizer chat template:
   - user message = `prompt`
   - assistant message = `chosen` or `rejected`
2. Tokenize full sequences with padding/truncation to `max_seq_len`.
3. Loss / log-probs use a **response mask**: tokens belonging to the assistant span only (same masking idea as SFT).

Do not accept pre-templated raw strings in v1 — keeps fixtures and tests simple.

### 4.2 Algorithm

With policy \(\pi_\theta\) and frozen reference \(\pi_{\mathrm{ref}}\) (usually SFT model):

\[
\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\Big[\log\sigma\Big(
\beta \big(
\log\pi_\theta(y_w|x) - \log\pi_{\mathrm{ref}}(y_w|x)
- (\log\pi_\theta(y_l|x) - \log\pi_{\mathrm{ref}}(y_l|x))
\big)
\Big)\Big]
\]

Implement in `losses.dpo_loss` given per-sequence average or sum log-probs of response tokens (document sum vs mean; prefer **token-sum** log-prob then average over batch — standard).

### 4.3 CLI (`dpo.py`)

Args: preference data, policy init path, ref path (default = policy init), `beta`, train hparams, save `dpo_final.pth`.

### 4.4 CPU smoke

Synthetic 4–8 preference pairs; 1–2 steps; unit test: when chosen logprob rises vs rejected under fixed ref, loss behaves monotonically in a constructed toy logit case.

---

## 5. Tests & CI

### 5.1 Unit tests (CPU)

| File | Cases |
|------|--------|
| `test_attention_kv.py` | Prefill; decode+cache; multi-token+cache shapes/no throw; GQA `n_kv_heads < n_heads` |
| `test_losses.py` | KD finite + α/T sensitivity; DPO sign/monotonic toy case |
| `test_checkpoint.py` | Roundtrip `save_checkpoint` → `load_weights` / `load_train_state` |
| `test_datasets.py` | SFT loss mask covers first assistant token; preference response mask |

### 5.2 CI

`.github/workflows/ci.yml`: on PR/push, install CPU torch + requirements, run `pytest -q`.

### 5.3 Smoke scripts (documented, not necessarily CI)

Commands for 1-epoch tiny pretrain → SFT → KD → DPO on synthetic data under `datasets/` (gitignored) or `tests/fixtures/`.

Prefer **committed tiny fixtures** under `tests/fixtures/` so CI/agents can run without inventing data.

---

## 6. Documentation & Resume Framing

### 6.1 README rewrite (honest)

Must include:

- What it is: from-scratch **~29M** Llama-style LM (RoPE, RMSNorm, SwiGLU, optional GQA, weight tying)
- Pipeline: tokenizer (committed) → pretrain → SFT → **KD** → **DPO** → PPL → chat
- Exact commands matching scripts
- Parameter count formula / default config table
- Explicit **Limitations**: no multi-GPU, no production serving, CPU smoke ≠ quality claim
- Delete/omit: unfinished API, INT8/INT4, TensorBoard-not-implemented, “enterprise”, “billion params”, fake distill narrative

### 6.2 `docs/experiments.md`

Template for the owner’s GPU runs: data prep, hparams, tables for loss/PPL, qualitative examples, ablation ideas (GQA on/off, KD α/T, DPO β). Leave metrics blank or mark “fill after local run”.

### 6.3 Resume bullet suggestions (short)

Provide 3–5 Chinese + English bullets the owner can paste, grounded in implemented facts only.

### 6.4 LICENSE

Add MIT `LICENSE` file to match README claim.

### 6.5 `AGENTS.md`

Update Cursor Cloud section for new modules (`losses.py`, `dpo.py`, pytest) without bloating setup script scope.

---

## 7. Implementation Order

1. `train_utils.py` + checkpoint/load fixes wired into existing scripts  
2. `model.py` KV-cache / RoPE / attention_mask / generate fixes + tests  
3. Dataset mask + chat/eval argparse fixes  
4. `losses.py` + rewrite `distill.py` + tests  
5. `PreferenceDataset` + `dpo.py` + tests  
6. Thin pretrain/SFT to use shared utils  
7. README + experiments.md + LICENSE + CI + AGENTS.md  
8. End-to-end CPU smoke on fixtures; commit; PR

## 8. Success Criteria

- All listed unit tests pass on CPU.
- Synthetic smoke: pretrain → SFT → KD → DPO completes ≥1 optimizer step each and writes loadable weights.
- `chat.py` loads both `*_final.pth` and training checkpoints correctly.
- README claims ⊆ implemented reality.
- No new vaporware files (API/quant/etc.).

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| SDPA behavior differs CPU vs CUDA | Tests use explicit mask path assertions; keep fallback matmul+mask if needed |
| DPO logprob numerical issues | float32 accumulate for token logprobs in loss helper |
| Scope creep | Stick to Non-Goals; no serving stack |
| Interview overclaim after rewrite | Resume bullets only reference tested features |

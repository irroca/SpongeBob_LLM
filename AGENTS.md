# AGENTS.md

## Start here

**Read [`docs/status.md`](docs/status.md) first.** It is the handoff document: where the project
actually stands, which PRs are open, the three engineering items that block real training, and the
ordered next steps. This file covers how the codebase behaves; `status.md` covers what to do next.

Whetstone is a from-scratch PyTorch LLM training/inference codebase (no web server, no
long-running service) targeting bilingual zh/en **verifiable tasks** (arithmetic, code). The core
workflow is a set of CLI scripts:

- `python3 -m datatools.fetch_evals` → `python3 -m datatools.prepare <spec>` → `train_tokenizer.py`
 — the data pipeline, which runs *before* any training. See `docs/corpus-plan.md`.
- `train_tokenizer.py --data <prepared jsonl> --out <dir> --vocab_size N` — train a BPE tokenizer
 on prepared corpus files. It no longer reads a hardcoded path.
- `pretrain.py` → `sft.py` → `distill.py` → `dpo.py` → `grpo.py` — the five-stage training pipeline
 (Pretrain → SFT → real Knowledge Distillation → DPO → GRPO/RLVR). `distill.py` does real KD (frozen
 teacher, CE + temperature-scaled KL on assistant tokens via `losses.kd_loss`), not the old fake
 special-token-weighted version. `dpo.py` runs standard Bradley-Terry DPO with a frozen reference
 model (`losses.dpo_loss` + `sequence_logprobs`). `grpo.py` runs on-policy GRPO against a
 rule-based environment in `envs/` (no reward model, no TRL dependency).
- `eval_ppl.py` — perplexity evaluation. `chat.py` — interactive inference REPL.

**README.md is the source of truth** for CLI commands/flags for every stage (including the CPU
smoke-test walkthrough using `tests/fixtures/`) — consult it and each script's `argparse` block
rather than duplicating commands here.

### Tests

- `tests/` holds CPU-only unit tests (`pytest>=8.0`, already in `requirements.txt`) plus small
 JSONL fixtures under `tests/fixtures/` (`pretrain_tiny.jsonl`, `sft_tiny.jsonl`,
 `preference_tiny.jsonl`) used both by the tests and by the README's CPU smoke-test commands for
 the pretrain/SFT/KD/DPO stages. The GRPO stage needs no fixture: `envs/` generates its own
 prompts and data (`python3 -m envs.generate_data --split {sft,preference,eval}`).
- Run the whole suite with `python -m pytest tests/ -q`. No GPU, network, or external data is
  required.
- CI (`.github/workflows/ci.yml`) installs CPU-wheel `torch` + `requirements.txt` and runs the same
  `pytest tests/ -q` on every push/PR.

### Environment / running caveats (non-obvious)

- **`--device` auto-selects**: `cuda` when `torch.cuda.is_available()` else `cpu`, everywhere
  (all five training scripts, `eval_ppl.py`, `chat.py`). On a CPU-only box a harmless
  `GradScaler ... CUDA is not available. Disabling.` warning is expected.
- **Only `--dtype float16` enables GradScaler.** `bfloat16` needs no loss scaling and will not
  create one (see `build_autocast_scaler`) — that is intended, not a missing feature. Prefer
  `bfloat16` on any modern GPU.
- **Two things will break at real scale and have not been fixed** (details and suggested designs
  in `docs/status.md` §3): `dataset.py` reads an entire JSONL into a Python list, and
  `model.py` materializes the full `(B, heads, q_len, kv_len)` attention score matrix instead of
  using `F.scaled_dot_product_attention`. Both are fine for the CPU tests and for fixtures; both
  are blocking for a ~100M model on a 30GB corpus.
- **No datasets or checkpoints are committed.** Training scripts expect JSONL under `datasets/`,
  which is git-ignored along with `results*/` and `*.pth`. Build real data with
  `datatools.prepare`, or generate synthetic task data with `envs.generate_data`; the committed
  `tests/fixtures/*.jsonl` are enough for a CPU smoke of every stage.
- **`tokenizer/zh_6400/` is a legacy tokenizer**, committed so the CPU tests and smoke runs work
  (vocab 6400, bos `<s>`, eos `</s>`, pad `<unk>`). It was trained on Chinese only and compresses
  code at 2.23 chars/token against 4.00 for English prose, so real bilingual+code training needs a
  retrained ~32k vocabulary (`train_tokenizer.py`). **Checkpoints do not survive a tokenizer
  change** — `resolve_model_config` raises on a `vocab_size` mismatch, which is intended.
- **`chat.py` is an interactive REPL** (`input()`), so pipe input for non-interactive runs, e.g.
 `printf 'question\nquit\n' | python3 chat.py --save_dir results --model_mode 1 --device cpu`.
 `--model_mode` selects the checkpoint: 0=`pretrain*.pth`, 1=`sft*.pth`, 2=`distill*.pth`,
 3=`dpo*.pth`, 4=`grpo*.pth`, and it falls back to `*_final.pth` filenames. `--save_dir` already
 defaults to `results`, matching the other stages' `--save_dir results`.
- **`--use_wandb True` requires `swanlab`** (imported lazily, not installed by default). Leave
 wandb off unless you install it. All five training scripts (`pretrain.py`/`sft.py`/`distill.py`/
 `dpo.py`/`grpo.py`) support `--use_wandb`/`--wandb_project` via `train_utils.init_wandb_if_needed`.
 `grpo.py` additionally appends every step's metrics to `{save_dir}/grpo_metrics.jsonl`, so RL
 curves can be plotted with no tracker installed.
- Installed with a recent major `transformers` (5.x) and `torch` 2.x CPU; the model code (custom
  `PreTrainedModel`/`PretrainedConfig` subclasses) is compatible with these.
- **Model architecture is CLI-configurable, and checkpoints carry their own architecture.**
 `train_utils.add_model_args(parser)` adds `--tokenizer_path` plus `--dim`/`--n_layers`/
 `--n_heads`/`--n_kv_heads`/`--hidden_dim`/`--multiple_of`/`--norm_eps`/`--rope_theta`/`--dropout`
 to all five training scripts and to `eval_ppl.py`/`chat.py`. Every arch flag defaults to `None`
 so `resolve_model_config(args, vocab_size, checkpoint_path=...)` can apply the precedence
 **explicit CLI > checkpoint > `LLMConfig` default**. Never hardcode `LLMConfig(...)` in a script
 again.
  - `n_heads` is **not** recoverable from tensor shapes (`head_dim = dim // n_heads`, so `wq` is
    always `dim x dim`; only the kv/q ratio is visible). `save_final_weights` therefore writes a
    `*.config.json` sidecar next to each bare `*_final.pth`; resolving from shapes alone warns
    that `n_heads` was assumed.
  - A checkpoint/tokenizer `vocab_size` mismatch raises. Retraining the tokenizer invalidates old
    weights — expect this when swapping corpora.
  - `load_weights` now warns on missing/unexpected keys: `strict=False` raises on shape mismatch
    but silently tolerates *absent* keys, which would leave whole layers randomly initialized.
  - `distill.py` resolves teacher and student architectures independently from their own
    checkpoints, so cross-size KD works; they only need a shared vocab.
- **The data pipeline lives in `datatools/`.** `python3 -m datatools.prepare <spec>` runs
 pull → filter → exact-dedup → decontaminate → split → manifest from a mixture spec
 (`configs/mixture_v1.json`). Individual stages are also CLIs: `stats`, `filters` (library only),
 `dedup`, `decontaminate`, `split`, `tokenizer_stats`.
  - `datatools/records.py` is the shared schema layer: it detects `text` / `conversations` /
    `prompt+chosen+rejected` / `question+answer` automatically, so **no tool takes a `--schema`
    flag**. `record_text` joins a record for stats/dedup/split; `record_parts` keeps the pieces
    separate for decontamination (the joined form inserts `=>` and role prefixes that never occur
    in natural text and would block n-gram matches); `prompt_text` isolates the input side.
  - **The whole pipeline is streaming.** Mixture weights are in tokens while corpora are published
    in documents and bytes, so `prepare` tokenizes as it pulls and stops when a source's share is
    met. At 10B tokens the corpus is ~30GB; don't add a stage that materializes it.
  - `datatools/minhash.py` is a self-contained MinHash+LSH implementation on numpy (no
    `datasketch`); its permutation coefficients are bounded so uint64 arithmetic never wraps —
    don't "simplify" that away. LSH proposes candidates and every candidate is verified against
    the full signature, so banding only trades recall for speed. It holds ~1KB per document, so
    near-dedup is bounded to 1–2M docs and is deliberately **not** part of `prepare`'s pass.
  - `datatools/decontaminate.py` is the real contamination check (13-gram + optional LCS 0.6,
    following SmolLM2); `dedup --against` is only exact prompt equality. CJK is split per
    character and latin per word, and eval items shorter than `n` are indexed at their own length.
    Very short answers (< `MIN_GRAM` units) fall back to exact matching — a known, tested limit.
  - `prepare`'s report is meant to be trustworthy: `fill < 100%` plus `ran out of data` means a
    source was silently down-weighted, and `kept%` excludes records pulled into the tokenization
    batch but never emitted. Don't regress either.
  - `datatools/fetch_evals.py` pulls benchmarks into the repo's `{"question","answer"}` schema, so
    one file serves both `prepare`'s `decontaminate.against` and `grpo.py --eval_path`. Converters
    are pure functions tested offline against recorded rows — update the recorded row when a
    field name changes upstream rather than loosening the converter. **A mixture spec with an
    empty `decontaminate.against` silently checks nothing**, so run `fetch_evals
    --decontamination_only --update_spec <spec>` before `prepare`.
- **Common training CLI flags come from `train_utils.add_common_train_args(parser, **overrides)`**
 (`--save_dir`, `--epochs`, `--batch_size`, `--learning_rate`, `--device`, `--use_wandb`,
 `--wandb_project`, `--dtype`, `--num_workers`, `--accumulation_steps`, `--grad_clip`, `--log_step`,
 `--save_step`, `--max_seq_len`, `--data_path`, `--resume_from`, `--seed`). Each script calls it
 first with its own default overrides, then adds its stage-specific extras (e.g. `dpo.py` adds
 `--policy_path`/`--beta`). Don't hand-roll these flags in a script — add/change them in
 `add_common_train_args` so all five scripts stay in sync. A stage that genuinely has no use for a
 shared flag passes `skip=(...)` rather than defining its own (`grpo.py` skips `--epochs`,
 `--accumulation_steps`, `--num_workers` because it is driven by `--rl_steps` over env-sampled
 prompts with no DataLoader). `--device` defaults to `"cuda" if torch.cuda.is_available() else
 "cpu"` everywhere (train scripts, `eval_ppl.py`, `chat.py`).
- **`--resume_from` does not guarantee identical batch order.** Each script's `DataLoader` uses
  `shuffle=True` with no fixed per-epoch seed, so resuming mid-epoch skips the same *number* of
  batches (via `start_step`) but not necessarily the *same* data. This is a known limitation
  (see README's "已知行为与限制"), not something to "fix" without an explicit ask — a real fix
  would need a seeded `Sampler`/checkpointed RNG state, which is out of scope for now.
- **`eval_ppl.py` wraps text pretrain-style** (`bos_token + text + eos_token`, matching
  `dataset.PretrainDataset`) before tokenizing, so PPL is computed on the same input distribution
  the model was trained on — don't strip that wrapping when touching `calculate_ppl`.
- **GRPO/RLVR specifics (`envs/`, `rollout.py`, `grpo.py`)**:
  - A rollout group is *one prompt repeated `--group_size` times*, never a batch of different
    prompts. The model applies RoPE from `start_pos` and has no left-padding offset, so mixed-length
    prompts cannot share a `generate` call. Multiple prompts per step are generated sequentially.
  - Token ids returned by `generate` are produced under `inference_mode` and **must be cloned**
    before they feed a training forward pass, or embedding backward raises on the saved index
    tensor. `rollout.generate_group` already does this.
  - Old/reference log-probs are **recomputed** with a no-grad forward, not captured during
    sampling: temperature/top-p reshape the sampling distribution but the importance ratio needs
    the policy's own distribution. With `--top_p < 1` the data is therefore slightly off-policy.
  - When micro-batching (`--micro_batch_size`), every chunk must divide by the *whole batch's*
    normalizer — that is what `grpo_policy_loss(..., normalizer=...)` is for. There are tests
    asserting chunked loss == full-batch loss for all three aggregations; don't "simplify" it away.
  - **The on-policy `seq_mean` loss value is ≈0 by construction** (ratio ≡ 1 and group advantages
    sum to zero). This is not a bug and not a sign the run is dead — look at `grad_norm`.
  - GRPO needs a policy that already emits the env's output format, otherwise every rollout scores
    0, every group has zero reward variance, and `grad_norm` is exactly 0. Cold start with SFT on
    `envs.generate_data --split sft` output first.
- **`model.generate`/`_stream_generate` supports batch>1 with per-row EOS**: each row tracks its
  own `finished` flag; once a row hits `eos_token_id` it emits that real EOS token on the hit step
  and `pad_token_id` on every step after, while other rows keep generating until they finish or
  `max_new_tokens` is reached. Callers must truncate at each row's own EOS position themselves —
  the returned tensor is not automatically trimmed per row.

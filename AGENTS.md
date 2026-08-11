# AGENTS.md

## Cursor Cloud specific instructions

SpongeBob LLM is a from-scratch PyTorch LLM training/inference codebase (no web server, no
long-running service). The core workflow is a set of CLI scripts:

- `train_tokenizer.py` — train the BPE tokenizer (reads a hardcoded `pretrain.jsonl` in repo root).
- `pretrain.py` → `SFT.py` → `distill.py` → `dpo.py` — the four-stage training pipeline
  (Pretrain → SFT → real Knowledge Distillation → DPO). `distill.py` does real KD (frozen teacher,
  CE + temperature-scaled KL on assistant tokens via `losses.kd_loss`), not the old fake
  special-token-weighted version. `dpo.py` runs standard Bradley-Terry DPO with a frozen reference
  model (`losses.dpo_loss` + `sequence_logprobs`).
- `eval_ppl.py` — perplexity evaluation. `chat.py` — interactive inference REPL.

**README.md is the source of truth** for CLI commands/flags for every stage (including the CPU
smoke-test walkthrough using `tests/fixtures/`) — consult it and each script's `argparse` block
rather than duplicating commands here.

### Tests

- `tests/` holds CPU-only unit tests (`pytest>=8.0`, already in `requirements.txt`) plus small
  JSONL fixtures under `tests/fixtures/` (`pretrain_tiny.jsonl`, `sft_tiny.jsonl`,
  `preference_tiny.jsonl`) used both by the tests and by the README's CPU smoke-test commands for
  all four training stages.
- Run the whole suite with `python -m pytest tests/ -q`. No GPU, network, or external data is
  required.
- CI (`.github/workflows/ci.yml`) installs CPU-wheel `torch` + `requirements.txt` and runs the same
  `pytest tests/ -q` on every push/PR.

### Environment / running caveats (non-obvious)

- **No GPU in Cursor Cloud VMs.** Every script defaults `--device` to `cuda` when
  `torch.cuda.is_available()` else `cpu`, so on this VM it auto-selects `cpu`. You can pass
  `--device cpu` explicitly. `torch` is installed as the CPU wheel; a harmless
  `GradScaler ... CUDA is not available. Disabling.` warning is expected on CPU.
- **No datasets or checkpoints are committed.** Training scripts expect JSONL data under
  `datasets/` (e.g. `datasets/pretrain.jsonl` with `{"text": ...}` lines, `datasets/sft_512.jsonl`
  with `{"conversations": [...]}` lines) and there are no pretrained `.pth` weights in the repo.
  To exercise the pipeline you must create small synthetic JSONL data first. `datasets/`,
  `results/`, and `*.pth` are git-ignored so demo artifacts are not committed.
- **Tokenizer is committed** in `spongebob_tokenizer/` (vocab size 6400, bos `<s>`, eos `</s>`,
  pad `<unk>`). You do NOT need to run `train_tokenizer.py` to run training/inference — but note
  that script reads `pretrain.jsonl` from the repo root (not `datasets/`).
- **`chat.py` is an interactive REPL** (`input()`), so pipe input for non-interactive runs, e.g.
  `printf 'question\nquit\n' | python3 chat.py --save_dir results --model_mode 1 --device cpu`.
  `--model_mode` selects the checkpoint: 0=`pretrain*.pth`, 1=`sft*.pth`, 2=`distill*.pth`,
  3=`dpo*.pth`, and it falls back to `*_final.pth` filenames. `--save_dir` already defaults to
  `results`, matching the other stages' `--save_dir results`.
- **`--use_wandb True` requires `swanlab`** (imported lazily, not installed by default). Leave
  wandb off unless you install it. All four training scripts (`pretrain.py`/`SFT.py`/`distill.py`/
  `dpo.py`) support `--use_wandb`/`--wandb_project` via `train_utils.init_wandb_if_needed`.
- Installed with a recent major `transformers` (5.x) and `torch` 2.x CPU; the model code (custom
  `PreTrainedModel`/`PretrainedConfig` subclasses) is compatible with these.
- **Common training CLI flags come from `train_utils.add_common_train_args(parser, **overrides)`**
  (`--save_dir`, `--epochs`, `--batch_size`, `--learning_rate`, `--device`, `--use_wandb`,
  `--wandb_project`, `--dtype`, `--num_workers`, `--accumulation_steps`, `--grad_clip`, `--log_step`,
  `--save_step`, `--max_seq_len`, `--data_path`, `--resume_from`, `--seed`). Each script calls it
  first with its own default overrides, then adds its stage-specific extras (e.g. `dpo.py` adds
  `--policy_path`/`--beta`). Don't hand-roll these flags in a script — add/change them in
  `add_common_train_args` so all four scripts stay in sync. `--device` defaults to `"cuda" if
  torch.cuda.is_available() else "cpu"` everywhere (train scripts, `eval_ppl.py`, `chat.py`).
- **`--resume_from` does not guarantee identical batch order.** Each script's `DataLoader` uses
  `shuffle=True` with no fixed per-epoch seed, so resuming mid-epoch skips the same *number* of
  batches (via `start_step`) but not necessarily the *same* data. This is a known limitation
  (see README's "已知行为与限制"), not something to "fix" without an explicit ask — a real fix
  would need a seeded `Sampler`/checkpointed RNG state, which is out of scope for now.
- **`eval_ppl.py` wraps text pretrain-style** (`bos_token + text + eos_token`, matching
  `dataset.PretrainDataset`) before tokenizing, so PPL is computed on the same input distribution
  the model was trained on — don't strip that wrapping when touching `calculate_ppl`.
- **`model.generate`/`_stream_generate` supports batch>1 with per-row EOS**: each row tracks its
  own `finished` flag; once a row hits `eos_token_id` it emits that real EOS token on the hit step
  and `pad_token_id` on every step after, while other rows keep generating until they finish or
  `max_new_tokens` is reached. Callers must truncate at each row's own EOS position themselves —
  the returned tensor is not automatically trimmed per row.

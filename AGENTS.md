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
  wandb off unless you install it.
- Installed with a recent major `transformers` (5.x) and `torch` 2.x CPU; the model code (custom
  `PreTrainedModel`/`PretrainedConfig` subclasses) is compatible with these.

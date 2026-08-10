# AGENTS.md

## Cursor Cloud specific instructions

SpongeBob LLM is a from-scratch PyTorch LLM training/inference codebase (no web server, no
long-running service). The core workflow is a set of CLI scripts:

- `train_tokenizer.py` — train the BPE tokenizer (reads a hardcoded `pretrain.jsonl` in repo root).
- `pretrain.py` → `SFT.py` → `distill.py` — the three-stage training pipeline.
- `eval_ppl.py` — perplexity evaluation. `chat.py` — interactive inference REPL.

Standard commands / args live in `README.md` and each script's `argparse` block; consult those
rather than duplicating here.

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
  `--model_mode` selects the checkpoint: 0=`pretrain*.pth`, 1=`sft*.pth`, 2=`distill*.pth`, and it
  falls back to `*_final.pth` filenames. `--save_dir` defaults to `sample_pth` (nonexistent);
  point it at `results`.
- **`--use_wandb True` requires `swanlab`** (imported lazily, not installed by default). Leave
  wandb off unless you install it.
- Installed with a recent major `transformers` (5.x) and `torch` 2.x CPU; the model code (custom
  `PreTrainedModel`/`PretrainedConfig` subclasses) is compatible with these.

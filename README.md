# SpongeBob LLM

从零实现的 **~29M** Llama 风格解码器语言模型：RoPE、RMSNorm、SwiGLU、可选 GQA、词嵌入/输出层权重共享，并包含完整的 **Pretrain → SFT → Knowledge Distillation → DPO** 训练与评估流水线。

> 本仓库定位是**可深挖的学习/作品集项目**，不是生产级大模型平台。默认配置约 **29M 参数**（`dim=512, n_layers=8, vocab=6400`），无多卡并行、无推理服务、无量化实现。

## 环境

```bash
# CPU 环境（如无 GPU）建议先装 CPU 版 PyTorch：
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
```

分词器已提交在 `spongebob_tokenizer/`（BPE，vocab=6400），一般无需重新训练。

## 仓库结构

| 路径 | 说明 |
|------|------|
| `model.py` / `Config.py` | 模型与配置 |
| `dataset.py` | Pretrain / SFT / Preference(DPO) 数据 |
| `train_utils.py` / `losses.py` | 共享训练工具与 CE/KD/DPO loss |
| `pretrain.py` / `SFT.py` / `distill.py` / `dpo.py` | 各阶段训练入口 |
| `eval_ppl.py` / `chat.py` | 困惑度评估与交互式生成 |
| `tests/` | CPU 单元测试与小型 fixtures |
| `docs/experiments.md` | 本地 GPU 实验记录模板 |

## 快速跑通（CPU smoke）

使用仓库内置 fixtures（无需外部数据）：

```bash
python3 pretrain.py --data_path tests/fixtures/pretrain_tiny.jsonl \
  --epochs 1 --batch_size 2 --max_seq_len 128 --save_dir results --device cpu --dtype float32

python3 SFT.py --data_path tests/fixtures/sft_tiny.jsonl \
  --pretrained_path results/pretrain_final.pth \
  --epochs 1 --batch_size 2 --max_seq_len 128 --save_dir results --device cpu --dtype float32

# teacher=pretrain_final.pth, student=sft_final.pth so KL is nonzero from step 1
# (both share the default config, so shapes still match; same-size teacher/student
# is also fine, e.g. --teacher_path results/sft_final.pth --student_path results/sft_final.pth)
python3 distill.py --data_path tests/fixtures/sft_tiny.jsonl \
  --teacher_path results/pretrain_final.pth --student_path results/sft_final.pth \
  --alpha 0.5 --temperature 2.0 --epochs 1 --batch_size 2 --max_seq_len 128 \
  --save_dir results --device cpu --dtype float32

python3 dpo.py --data_path tests/fixtures/preference_tiny.jsonl \
  --policy_path results/sft_final.pth --beta 0.1 \
  --epochs 1 --batch_size 2 --max_seq_len 128 --save_dir results --device cpu --dtype float32

python3 eval_ppl.py --model_path results/pretrain_final.pth \
  --dataset_path tests/fixtures/pretrain_tiny.jsonl --max_seq_len 128 --device cpu

printf '海绵宝宝喜欢做什么？\nquit\n' | python3 chat.py \
  --save_dir results --model_mode 1 --device cpu --max_new_tokens 64
```

`chat.py --model_mode`：`0` pretrain / `1` SFT / `2` KD / `3` DPO。  
`load_weights` 同时支持纯 `state_dict`（`*_final.pth`）与训练 checkpoint（含 `model_state_dict`）。

## 数据格式

**Pretrain** (`{"text": "..."}` JSONL)  
**SFT** (`{"conversations": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`)  
**DPO** (`{"prompt":"...","chosen":"...","rejected":"..."}`)

## 算法要点

- **KD**（`distill.py`）：冻结 teacher，学生优化  
  \((1-\alpha)\mathrm{CE} + \alpha\, T^2 \mathrm{KL}(p_T \| p_S)\)，仅在 assistant token 上计算。
- **DPO**（`dpo.py`）：冻结 reference（默认=初始 SFT），标准 Bradley-Terry / DPO loss，response token-sum log-prob。
- **注意力**：因果 mask 按 `q_len × kv_len` 构造，支持 KV cache 多 token 续写；可选 padding `attention_mask`。

## 训练 CLI / wandb（`train_utils.py`）

四个训练入口（`pretrain.py` / `SFT.py` / `distill.py` / `dpo.py`）共享同一套 CLI 参数，由
`train_utils.add_common_train_args(parser, **overrides)` 统一添加（`--save_dir` / `--epochs` /
`--batch_size` / `--learning_rate` / `--device` / `--use_wandb` / `--wandb_project` / `--dtype` /
`--num_workers` / `--accumulation_steps` / `--grad_clip` / `--log_step` / `--save_step` /
`--max_seq_len` / `--data_path` / `--resume_from` / `--seed`）；每个脚本通过关键字参数覆盖自己的默认值
（如 `distill.py` 用 `wandb_project="SpongeBob-Distill"`），再 `add_argument` 自己的额外参数
（如 `--teacher_path` / `--beta`）。

- `--device` 统一默认 `"cuda" if torch.cuda.is_available() else "cpu"`（四个训练脚本 + `eval_ppl.py` +
  `chat.py` 一致；此前 `pretrain.py`/`SFT.py`/`distill.py`/`dpo.py` 默认写的是 `"cuda:0"`）。
- `--use_wandb True --wandb_project ...`：四个训练脚本现在都支持（`distill.py`/`dpo.py` 是本轮新增，
  之前只有 `pretrain.py`/`SFT.py` 有）。日志由 `train_utils.init_wandb_if_needed(args, run_name=...)`
  统一处理：`use_wandb=False` 时直接返回 `None`（不 import）；为 `True` 时才 `import swanlab as wandb`
  并 `wandb.init(...)`，随后训练循环里 `if wandb is not None: wandb.log({...})`。`swanlab` 是可选依赖
  （见 `requirements.txt`），未安装时打开 `--use_wandb` 会直接抛 `ModuleNotFoundError`。

## 测试

```bash
python3 -m pytest tests/ -q
```

## 默认模型配置

```python
LLMConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=8, vocab_size=6400, max_seq_len=1024)
# ≈ 29M params；设置 n_kv_heads < n_heads 即启用 GQA
```

## 已知行为与限制（本轮 solidify 覆盖）

- **`--resume_from` + shuffle 的顺序不保证**：`DataLoader(..., shuffle=True)` 每次重新创建
  `DataLoader`/新进程时都会用不同的打乱顺序（没有固定/可派生的 per-epoch seed），而
  `train_epoch` 的 resume 逻辑是"跳过前 `start_step` 个 batch"。这只保证**跳过的 batch 数量**
  与上次一致，**不保证**跳过的是同一批数据——同一 epoch 内 resume 后大概率会重复或漏掉一些样本。
  这是当前实现的已知限制，不是 bug；如需严格可复现的 resume，需要自己引入固定 seed 的
  `Sampler`（不在本轮范围内）。
- **`eval_ppl.py` 按 pretrain 方式包裹文本**：`calculate_ppl` 对每条文本先用
  `wrap_pretrain_text` 包上 `bos_token`/`eos_token`（与 `dataset.PretrainDataset` 编码方式一致），
  再 tokenize/padding/truncate 计算困惑度，确保评估输入分布与训练输入分布对齐（旧版本直接对裸文本
  计算，会低估真实 PPL）。
- **`generate` 支持 batch>1 且逐行独立判断 EOS**：`SpongeBob._stream_generate` 维护一个
  `finished` 布尔张量，每行各自判断是否已生成 `eos_token_id`；已结束的行从**下一步**开始持续输出
  `pad_token_id`（命中 EOS 当步仍输出真实 EOS token），其余未结束的行继续正常采样，直到全部行
  `finished` 或达到 `max_new_tokens` 才停止整个循环。因此调用方拿到的输出里，已结束的行末尾会有
  `pad_token_id` 填充，需要自行按 EOS 位置截断（`chat.py` 单条生成时不受影响）。
- **KV cache 下的 `attention_mask` 长度语义**：`model.forward` 在带 `past_key_values` 续写时，若传入
  的 `attention_mask` 长度等于新 chunk 长度（`q_len`），会自动在左侧补 1（等价于假设所有缓存的历史
  key 都可见）扩展到 `kv_len` 再使用；若长度已等于 `kv_len` 则原样使用；其他长度会抛 `ValueError`。
- 无分布式训练、无 FlashAttention 绑定、无服务化 API、无 INT8/INT4。
- fixtures / CPU smoke **不能**代表语言能力；请在自己的 GPU + 真实数据上填 `docs/experiments.md`。
- 旧版伪「特殊 token 加权蒸馏」已移除，现为真实 KD。

## 简历表述建议（可直接改写）

- 从零实现 ~29M Llama 风格 LM（RoPE / RMSNorm / SwiGLU / 可选 GQA），打通 Pretrain→SFT→KD→DPO。
- 修复并单测覆盖 KV-cache 因果 mask、checkpoint 双格式加载、AMP/LR 调度等训练工程问题。
- 实现温度 KD 与 DPO（response-mask logprob），附 CPU 可复现测试与实验模板。

## License

MIT — see [LICENSE](LICENSE).

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

## 测试

```bash
python3 -m pytest tests/ -q
```

## 默认模型配置

```python
LLMConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=8, vocab_size=6400, max_seq_len=1024)
# ≈ 29M params；设置 n_kv_heads < n_heads 即启用 GQA
```

## 局限（写进简历前请读）

- 无分布式训练、无 FlashAttention 绑定、无服务化 API、无 INT8/INT4。
- fixtures / CPU smoke **不能**代表语言能力；请在自己的 GPU + 真实数据上填 `docs/experiments.md`。
- 旧版伪「特殊 token 加权蒸馏」已移除，现为真实 KD。

## 简历表述建议（可直接改写）

- 从零实现 ~29M Llama 风格 LM（RoPE / RMSNorm / SwiGLU / 可选 GQA），打通 Pretrain→SFT→KD→DPO。
- 修复并单测覆盖 KV-cache 因果 mask、checkpoint 双格式加载、AMP/LR 调度等训练工程问题。
- 实现温度 KD 与 DPO（response-mask logprob），附 CPU 可复现测试与实验模板。

## License

MIT — see [LICENSE](LICENSE).

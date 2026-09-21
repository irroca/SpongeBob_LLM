# SpongeBob LLM

从零实现的 **~29M** Llama 风格解码器语言模型：RoPE、RMSNorm、SwiGLU、可选 GQA、词嵌入/输出层权重共享，并包含完整的 **Pretrain → SFT → Knowledge Distillation → DPO → GRPO/RLVR** 训练与评估流水线。

RL 部分不依赖 TRL/veRL：可验证奖励环境、组相对优势、clipped policy loss、k3 KL 全部从零实现，
并把 Dr. GRPO / DAPO 的几个关键改动做成开关而不是分叉代码，方便做消融。

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
| `train_utils.py` / `losses.py` | 共享训练工具与 CE/KD/DPO/GRPO loss |
| `envs/` | 可验证奖励环境（RLVR）与数据生成 |
| `rollout.py` | GRPO 在线采样：分组 rollout、completion mask、logprob |
| `pretrain.py` / `SFT.py` / `distill.py` / `dpo.py` / `grpo.py` | 各阶段训练入口 |
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

python3 grpo.py --policy_path results/sft_final.pth --env arithmetic \
  --rl_steps 3 --group_size 4 --batch_size 2 --max_new_tokens 64 --max_seq_len 192 \
  --micro_batch_size 4 --learning_rate 2e-6 --save_dir results --device cpu --dtype float32

python3 eval_ppl.py --model_path results/pretrain_final.pth \
  --dataset_path tests/fixtures/pretrain_tiny.jsonl --max_seq_len 128 --device cpu

printf '海绵宝宝喜欢做什么？\nquit\n' | python3 chat.py \
  --save_dir results --model_mode 1 --device cpu --max_new_tokens 64
```

> GRPO 的 policy 必须已经会输出 `<think>/<answer>` 格式，否则采样全是 0 奖励、组内零方差、梯度恒为 0。
> 用 `tests/fixtures/sft_tiny.jsonl` 训出来的模型不满足这一点，所以 GRPO 的 smoke 要先用
> `envs.generate_data` 造算术 SFT 数据做冷启动（见下一节）。

`chat.py --model_mode`：`0` pretrain / `1` SFT / `2` KD / `3` DPO / `4` GRPO。  
`load_weights` 同时支持纯 `state_dict`（`*_final.pth`）与训练 checkpoint（含 `model_state_dict`）。

## 数据格式

**Pretrain** (`{"text": "..."}` JSONL)  
**SFT** (`{"conversations": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`)  
**DPO** (`{"prompt":"...","chosen":"...","rejected":"..."}`)  
**RLVR 评测集** (`{"question":"17 + 8","answer":"25"}`，其余字段进 `Task.meta`)

GRPO 阶段没有「目标输出」这种数据——prompt 由环境生成，监督信号只有奖励标量。

## Mini-RLVR：可验证奖励环境 + GRPO

### 环境（`envs/`）

`ArithmeticEnv` 生成多位加减法，策略必须输出：

```text
<think>把 25 拆成 20 和 5：37 + 20 = 57；57 + 5 = 62。</think><answer>62</answer>
```

两个规则奖励，无 reward model、无人工标注：

| 奖励 | 判定 | 默认权重 |
|------|------|----------|
| accuracy | `<answer>` 里的整数等于 ground truth | 1.0 |
| format | think/answer 两对标签都闭合且顺序正确 | 0.2 |

- 默认 `--strict_answer True`：accuracy 以能解析出 `<answer>` 为前提，format 是前置条件而非白送的加分。
  设成 `False` 则退化为「取全文最后一个整数」，用来区分**不会算**和**不会按格式写**。
- `Reward.hacked_format`（格式对、答案错）单独统计，reward hacking 在曲线上直接可见。
- `envs/base.py` 定义 `Task` / `Reward` / `TaskEnv` 接口与注册表；加新环境只需实现
  `sample_task` / `render` / `score`，训练代码不用动。

同一个环境还负责造别的阶段的数据，所以四个后训练阶段能在同一批题目上对比：

```bash
python3 -m envs.generate_data --split sft        --n 2000 --out datasets/arith_sft.jsonl
python3 -m envs.generate_data --split preference --n 1000 --out datasets/arith_pref.jsonl
python3 -m envs.generate_data --split eval       --n 200  --out datasets/arith_eval.jsonl
```

### 算法（`losses.py` + `rollout.py`）

对每个 prompt 采 \(G\) 条回复，用组内均值当 baseline（这就是 GRPO 省掉 critic 的地方）：

\[
A_i = \frac{r_i - \mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r) + \varepsilon}
\]

\[
\mathcal L = -\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t
\min\!\big(\rho_{i,t}A_i,\ \mathrm{clip}(\rho_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})A_i\big)
\;+\; \beta_{\mathrm{KL}}\, D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\mathrm{ref}})
\]

其中 \(\rho_{i,t} = \pi_\theta(o_{i,t}\mid x, o_{i,<t}) / \pi_{\mathrm{old}}(\cdot)\)，KL 用 Schulman 的 k3 估计量（恒非负）。

论文里的几个改动都是**开关**，不是分叉代码，方便做消融：

| 开关 | 对应工作 |
|------|----------|
| `--normalize_advantage_std False` | Dr. GRPO 优势（去掉 std，保留题目难度差异） |
| `--aggregation token_mean` | DAPO token-level loss |
| `--aggregation dr_grpo` | Dr. GRPO 常数分母（去长度偏置） |
| `--clip_eps_high > --clip_eps_low` | DAPO clip-higher |
| `--filter_zero_variance True` | DAPO dynamic sampling（丢掉零方差组） |
| `--kl_coeff 0` | 不锚定 reference（DAPO / Dr. GRPO 默认） |

### 工程上真正踩的坑（`rollout.py`）

- **一个 group = 同一 prompt 重复 G 次**。模型的 RoPE 按 `start_pos` 取位置、没有 left-padding 偏移，
  不同长度的 prompt 不能拼一个 batch 生成；同 prompt 复制天然等长，顺带就是 GRPO 需要的分组结构。
  代价是多个 prompt 只能逐个 generate。
- **completion mask 截到每行自己的第一个 EOS（含）**，复用已有的 batch>1 逐行 EOS 生成逻辑；
  之后的 pad 一律不计入 loss。
- **log-prob 是重算的，不是采样时顺手记的**。temperature / top-p 改变的是采样分布，
  而重要性比需要的是策略自身的分布。注意 `top_p < 1` 时训练数据严格来说是 off-policy 的。
- **`generate` 在 `inference_mode` 下运行**，产出的 token id 必须 `clone()` 出来才能当 embedding 索引
  参与需要 backward 的前向（embedding 反向会保存索引张量）。
- **右侧 padding 是安全的**：因果模型下 pad 只在 completion 之后，且 loss mask 不覆盖它们，
  所以训练前向不需要额外的 attention mask。
- **微批必须共用整批的分母**。`grpo_policy_loss(..., normalizer=...)` 就是为此存在的，
  单测断言三种聚合方式下「分块损失之和 == 单次全批损失」。

### 训练指标（`{save_dir}/grpo_metrics.jsonl`，每步一行）

```text
reward_mean reward_std accuracy format_rate hack_rate silent_group_frac
completion_len truncated_frac entropy kl clip_frac ratio_mean grad_norm adv_abs_mean
```

**为什么要看 `grad_norm` 而不是 `loss`**：on-policy 第一次 pass 时 \(\rho \equiv 1\)，
而组内优势之和为 0，所以 `seq_mean` 聚合下的 loss **恒等于 0**——信号全在梯度里。
（`token_mean` 按长度加权，loss 就不为 0 了，这个差值正是 Dr. GRPO 要去掉的长度偏置，
见 `tests/test_grpo_losses.py::test_on_policy_seq_mean_loss_is_zero_while_token_mean_leaks_length_bias`。）

`silent_group_frac` 是组内奖励全相同的比例：这些组优势全零、梯度恒为 0。CPU smoke 里能直接看到
`silent=1.00` 的那一步 `gnorm=0.0000`，这就是 DAPO dynamic sampling 要解决的问题。

用 `analyze_grpo.py` 做窗口平均并排比多组消融：

```bash
python3 analyze_grpo.py results_a/grpo_metrics.jsonl results_b/grpo_metrics.jsonl --window 25
```

### CPU 上已经跑出来的两个现象（详见 `docs/experiments.md`）

同样的 29M 策略、同样的 lr 和 seed，只差一个 `--kl_coeff 0.02`：

| steps | 无 KL 的 format rate | KL=0.02 的 format rate |
|-------|---------------------|------------------------|
| 25–49 | **0.296**（峰值） | 0.475 |
| 50–74 | 0.000 | 0.637 |
| 125–149 | 0.000 | **0.915** |

1. **零奖励是 GRPO 的吸收态**：无 KL 的那一组在第 50 步左右熵从 3.75 掉到 1.3、策略退化成反复输出
   `<think>`，此后奖励恒 0 → 优势恒 0 → `grad_norm` 恒 0，剩下 100 步完全没有梯度，再也起不来。
2. **它学会的只有格式**：两组的 accuracy 全程 0.000，而 `hack_rate` 与 `format_rate` 完全重合——
   每一条格式正确的输出答案都是错的。策略把唯一够得着的奖励分量刷满了。单看一条 reward 曲线
   会误以为在稳步进步，这正是环境要把 accuracy / format 分开上报的原因。

## 算法要点

- **KD**（`distill.py`）：冻结 teacher，学生优化  
  \((1-\alpha)\mathrm{CE} + \alpha\, T^2 \mathrm{KL}(p_T \| p_S)\)，仅在 assistant token 上计算。
- **DPO**（`dpo.py`）：冻结 reference（默认=初始 SFT），标准 Bradley-Terry / DPO loss，response token-sum log-prob。
- **GRPO**（`grpo.py`）：见上一节。
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
- **GRPO 需要冷启动**：策略必须已经会输出环境要求的格式，否则每条 rollout 都是 0 奖励、
  组内零方差、`grad_norm` 恒为 0。先用 `envs.generate_data --split sft` 的数据做 SFT。
  反过来，SFT 训到完全饱和（熵极低）同样采不出奖励差异，也学不动——冷启动要**够但不过**。
- **一次 `generate` 只处理一个 prompt 的 group**：模型没有 left-padding 的 RoPE 偏移，
  不同长度 prompt 不能同批生成。多个 prompt 只能串行，这是当前实现的吞吐上限。
- **`--top_p < 1` 时训练数据严格来说是 off-policy 的**：采样分布被截断过，而重要性比用的是
  策略自身分布。默认 `--top_p 1.0` 就是为了避免这个偏差。
- **GRPO 的 `loss` 不是进度指标**：on-policy 且 `seq_mean` 聚合时它恒等于 0，看 `grad_norm`。
- 无分布式训练、无 FlashAttention 绑定、无服务化 API、无 INT8/INT4、无 PPO critic、无 PRM。
- fixtures / CPU smoke **不能**代表语言能力；请在自己的 GPU + 真实数据上填 `docs/experiments.md`。
- 旧版伪「特殊 token 加权蒸馏」已移除，现为真实 KD。

## 简历表述建议（可直接改写）

> 只写你真的跑过的部分。RL 那几条在你自己填完 `docs/experiments.md` 之前不要用。

- 从零实现 ~29M Llama 风格 LM（RoPE / RMSNorm / SwiGLU / 可选 GQA），打通
  Pretrain→SFT→KD→DPO→GRPO 全栈后训练流水线。
- 不依赖 TRL/veRL，从零实现 GRPO：组相对优势、clipped 代理目标、k3 KL，
  并把 Dr. GRPO / DAPO 的优势归一化、token-level loss、clip-higher、dynamic sampling
  做成可消融开关。
- 设计规则奖励环境（accuracy + format 双分量），量化 reward hacking（格式对答案错的比例）、
  零方差组占比与熵坍缩，而不是只报一条 reward 曲线。
- 修复并单测覆盖 KV-cache 因果 mask、checkpoint 双格式加载、AMP/LR 调度等训练工程问题；
  RL 侧单测断言微批损失之和等于全批损失、零方差组梯度为零、completion mask 停在各自 EOS。

## License

MIT — see [LICENSE](LICENSE).

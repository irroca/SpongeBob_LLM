# 项目状态与路线（交接文档）

**最后更新**：2026-09-23，云端 agent session 结束时。
**下一步在本地进行**，本文档是新 session 的入口——先读这里，再读 `AGENTS.md`。

---

## 0. 一句话现状

算法链路（五阶段后训练）和数据管线都已实现并有 318 个 CPU 单测覆盖，语料方案和模型规模已定；
**正式训练还没开始**，而且开始之前有三件工程上的事必须先做（见 §3，都是会直接导致 OOM 或
显存不够的硬问题）。

仓库里现有的一切数字都是 29M 玩具规模的**实现验证**，不是能力声明。

---

## 1. 已完成，可以依赖的部分

### 模型与训练
- ~29M Llama 风格解码器（RoPE / RMSNorm / SwiGLU / 可选 GQA / 权重共享），架构**全部走 CLI**
  （`--dim` / `--n_layers` / `--n_heads` / `--n_kv_heads` / ...），不需要改源码做尺寸消融
- 五个阶段都能跑通：`pretrain.py` → `sft.py` → `distill.py` → `dpo.py` → `grpo.py`
- 架构解析优先级 **显式 CLI > checkpoint 记录 > 库默认值**；`*_final.pth` 旁边写
  `*.config.json` sidecar（因为 `n_heads` 无法从张量形状反推）
- 真正的跨尺寸蒸馏（teacher / student 各自从自己的 checkpoint 解析架构）

### RLVR / GRPO（[PR #4](https://github.com/irroca/Whetstone/pull/4)）
- `envs/`：可验证奖励环境，accuracy 与 format 双分量分开上报
- `losses.py`：组相对优势、clipped policy loss、k3 KL；Dr. GRPO / DAPO 的改动都是**开关**
- `rollout.py`：分组 rollout、逐行 completion mask、log-prob 重算
- `analyze_grpo.py`：窗口平均 + 多组并排比较

### 数据管线（[PR #5](https://github.com/irroca/Whetstone/pull/5)）
- `datatools.prepare`：按配比 spec 跑 拉取 → 过滤 → 精确去重 → 去污染 → 划分 → manifest，全程流式
- `datatools.stats` / `filters` / `dedup`（MinHash）/ `decontaminate`（13-gram + LCS）/ `split` /
  `tokenizer_stats` / `fetch_evals`
- `train_tokenizer.py` 已改成 CLI 驱动

### 调研与决策（`docs/corpus-plan.md`）
- 语料候选清单（中英 web / 代码 / 数学 / 书籍 / 可验证任务），含规模、许可证、获取上的坑
- 三份可引用的公开配比（SmolLM3、SmolLM2、CCI3.0-HQ）
- **已定**：模型 ~100M（`--dim 768 --n_layers 12 --n_heads 12 --n_kv_heads 3`，vocab 32k）、
  中文语料 `epfml/FineWeb2-HQ` 的 `cmn_Hani`、仓库公开、正式训练租卡

### 已经跑出来的实验结论（`docs/experiments.md`）
- **零奖励是 GRPO 的吸收态**：无 KL 锚定时策略在第 50 步崩溃，此后奖励恒 0 → 组内零方差 →
  优势恒 0 → `grad_norm` 精确为 0，剩下 100 步完全没有梯度，再也起不来
- **它只学会了格式**：accuracy 全程 0.000，而 `hack_rate` 与 `format_rate` 完全重合
- **eval 曲线平不等于优化器坏了**：另一组 100 步 greedy eval 完全不动，但权重确实在变

---

## 2. 待合并的 PR

| PR | 分支 | 目标 | 状态 |
|----|------|------|------|
| [#4](https://github.com/irroca/Whetstone/pull/4) | `cursor/mini-rlvr-grpo-9ce6` | `main` | ready for review，CI 绿 |
| [#5](https://github.com/irroca/Whetstone/pull/5) | `cursor/data-tooling-and-model-cli-9ce6` | `#4` 的分支 | ready for review，CI 绿 |

**#5 叠在 #4 上**，所以顺序是先合 #4 再合 #5。也可以把 #5 的 base 改成 `main`（#4 的提交已在
#5 的历史里，改完 #5 含全部 16 个 commit，#4 会自动关闭）。

**本地新 session 的第一件事就是决定怎么合这两个 PR**，后续所有工作都建立在它们之上。

---

## 3. 开始正式训练之前必须做的三件工程事

这三件都是在云端 CPU 上跑不出来、但一上真实规模就会立刻爆的问题。按优先级排：

### 3.1 【阻塞】`dataset.py` 把整个 JSONL 读进内存

`PretrainDataset.load_data` / `SFTDataset.load_data` 都是把所有行 `json.loads` 进一个 Python
list。10B token 约 30GB 文本、约 2000 万条文档，这样会直接 OOM。而且每个 `__getitem__` 都在
训练循环里重新 tokenize，纯浪费。

**建议做法**（nanoGPT 风格，成熟且简单）：
1. 新增 `datatools/tokenize_corpus.py`：把 `prepare` 产出的 JSONL 预 tokenize 成一个扁平的
   `uint16` 二进制 + 文档边界索引。vocab 32768 < 65536，所以 `uint16` 够用，
   10B token = 20GB 文件
2. 新增 `dataset.MemmapPretrainDataset`：`np.memmap` 打开 `.bin`，`__getitem__` 按
   `max_seq_len` 切窗口。常数内存、零 tokenize 开销
3. 老的 `PretrainDataset` 保留给小数据和单测用

验收：在 30GB 语料上 RSS 稳定在几百 MB；吞吐比现在快一个数量级（因为不再在循环里 tokenize）。

### 3.2 【阻塞 8GB 本地卡】注意力显式构造完整 score 矩阵

`model.py:121` 是 `scores = (xq @ xk.transpose(-2, -1)) / sqrt(head_dim)`，然后 `masked_fill`
再 `F.softmax(scores.float())`。**没有用 SDPA / FlashAttention**，所以
`(B, n_heads, q_len, kv_len)` 这个张量会被完整物化，softmax 还会升到 fp32。

粗算 ~100M 模型（12 头、12 层）：

| seq_len | 每层 score 张量（bf16，B=1） | 12 层保留的激活 |
|---------|---------------------------|----------------|
| 1024 | 25 MB | ~0.3 GB × B |
| 2048 | 100 MB | ~1.2 GB × B |

8GB 卡上 seq 2048 会先在这里爆。

**建议做法**：把 prefill 路径换成 `F.scaled_dot_product_attention`，保留现有的显式 mask 路径
作为 fallback 和对照（现有单测覆盖了 prefill / decode+cache / 多 token 续写 / GQA /
padding mask 五种情况，可以直接用来验证两条路径等价）。KV cache 续写时 `q_len` 很小，
物化开销可以忽略，不急着改。

注意 `--top_p`、repetition penalty 这些生成逻辑不受影响。

### 3.3 显存与吞吐的实测校准

`docs/corpus-plan.md` 里的「~92 小时」是按 `FLOPs/token ≈ 6N` 加一个假设的 30k token/s 估的，
**没有在真卡上量过**。小模型常被显存带宽和 kernel launch 限制而不是算力，实测可能在 15k–60k
之间浮动。

先做一件小事：固定 `--dim 768 --n_layers 12`，扫 `--batch_size` 和 `--max_seq_len`，
记录「不 OOM 的最大组合」和「token/s」，写进 `docs/experiments.md`。这决定后面所有时间预算。

顺带可以考虑（不阻塞）：gradient checkpointing、`torch.compile`、fused AdamW。

---

## 4. 下一步的实验路线

按依赖顺序，每步都有明确的验收标准。

### 步骤 1：合并 #4 和 #5
见 §2。

### 步骤 2：拉评测集，闭上去污染的环
```bash
python3 -m datatools.fetch_evals --decontamination_only --update_spec configs/mixture_v1.json
```
验收：`configs/mixture_v1.json` 的 `decontaminate.against` 非空（**为空时去污染是静默空转**，
这会让后面所有评测数字失去意义）。

### 步骤 3：小规模跑通数据管线
```bash
python3 -m datatools.prepare configs/mixture_v1.json --scale 0.001 --out_dir datasets/smoke
python3 -m datatools.stats datasets/smoke/train.jsonl --tokenizer tokenizer/zh_6400
```
验收：每个源的 `fill` 都接近 100%（若有源标了 `ran out of data`，说明配比已被悄悄改变，
要调 spec 或换数据源）；`manifest.json` 里的过滤拒绝分布看起来合理，没有哪条规则删掉大半。

**这一步会暴露 spec 里的现实问题**（比如 `starcoderdata` 的 `data_dir` 写法、FineWeb2-HQ
的实际字段名），务必先跑再上规模。

### 步骤 4：消融 #1（中文占比）和 #5（词表大小）
这两个决定其余所有配置，所以先做。代理模型用现有 29M 默认配置，每组 0.3–0.5B token。

- 消融 #1：中文占比 0% / 15% / 30% / 45%（其余按比例缩放）
- 消融 #5：词表 16k / 32k / 48k

评测不能只看 PPL（跨配比不可比，因为 token 分布不同）。要看：
中英各自的 holdout PPL（同一 tokenizer 下才可比）、算术任务准确率（复用
`envs/arithmetic.py` 的评测集）、代码补全执行通过率、**语言混淆率**（中文 prompt 下输出
英文的比例，双语小模型的典型故障）。

需要新写：一个消融编排脚本（生成各组 spec → 跑训练 → 汇总对比表）。可以照 `analyze_grpo.py`
的形式做。

### 步骤 5：正式数据集 + 重训 tokenizer
```bash
python3 -m datatools.prepare configs/mixture_v1.json --out_dir datasets/prepared
python3 train_tokenizer.py --data datasets/prepared/train.jsonl --out tokenizer/v1_32k --vocab_size 32768
python3 -m datatools.tokenizer_stats --probe --tokenizer tokenizer/v1_32k tokenizer/zh_6400
```
验收：新词表在代码上的 `chars/token` 明显高于 2.23（旧词表的值），中文不显著变差。

**注意重训 tokenizer 会让所有旧 checkpoint 失效**——`resolve_model_config` 会在 `vocab_size`
不匹配时直接报错，这是设计如此。本地若还有想留的 29M 权重，先归档。

### 步骤 6：正式预训练（~100M / ~10B token，租卡）
```bash
python3 pretrain.py --dim 768 --n_layers 12 --n_heads 12 --n_kv_heads 3 \
  --tokenizer_path tokenizer/v1_32k --max_seq_len 2048 \
  --data_path datasets/prepared/train.jsonl --save_dir results --dtype bfloat16
```
前提是 §3.1 和 §3.2 已经做完。填 `docs/experiments.md` 的 loss / PPL 表。

### 步骤 7：重建后训练四阶段
SFT / KD / DPO 的数据要基于新语料和可验证任务重做（`envs.generate_data` 负责算术那部分；
代码任务的环境还没写，见 §5）。然后 GRPO——这次 accuracy 应该终于有方差了，这是整个项目
最关键的验证点。

---

## 5. 长期计划

### 阶段 A：把可验证任务做实（当前重心）
- 100M 双语模型，预训练 + 后训练全链路跑通
- 算术任务上 GRPO 让 **accuracy 真正上升**（不只是 format rate）——这是上一轮没做到的
- 填满 `docs/experiments.md` 的消融表

### 阶段 B：第二个环境——代码任务
`envs/base.py` 的 `TaskEnv` 接口已经留好（`sample_task` / `render` / `score`），加环境不用动
训练代码。代码环境的奖励是**执行单元测试**，比算术更接近真实 RLVR，也是 Anthropic
Fellows 那类岗位明确在做的「creating RL environments」。

要点：沙箱执行（子进程 + 超时 + 资源限制）、测试用例来源（`PRIME-RL/Eurus-2-RL-Data` 已经带
测试用例，`fetch_evals` 里已注册）、部分通过的奖励整形（pass@k 还是通过率）。

### 阶段 C：算法侧的消融与技术博客
现在都是开关，可以直接跑：
- Dr. GRPO（去 std 归一化 + 常数分母）vs GRPO
- DAPO（token-level loss + clip-higher + dynamic sampling）
- 难度课程（用 `big_math` 的 `llama8b_solve_rate` 筛中等难度题，避开零方差组）
- 熵下界 vs KL 锚定，验证崩溃是 KL 问题还是一般的探索问题

博客建议题目仍然偏现象而不是教程，例如《在小模型上，GRPO 学会的是计算还是格式？》。
素材已经有了一半（§1 末尾那三条结论）。

### 明确不做
PPO critic、PRM（过程奖励模型）、MoE、多卡并行、推理服务化、量化。

---

## 6. 从云端切到本地：环境差异

| | 云端（之前） | 本地（现在） |
|---|---|---|
| GPU | 无，`torch.cuda.is_available()` 为 False | RTX 5060Ti 8GB，正式训练租卡 |
| `--dtype` | 只能 `float32` | 用 `bfloat16`（Blackwell 原生支持，且不需要 GradScaler）|
| 网络 | 通 HF | 同 |
| 数据 | `datasets/` 是空的 | 需要重新拉，见 §4 步骤 2–3 |

要注意的几点：

- **`--dtype float16` 才会启用 GradScaler**，`bfloat16` 不需要也不会启用
  （`build_autocast_scaler` 里的逻辑），别以为是 bug
- **`big_math` 数据集是 gated 的**（auto-approve），要先在 HF 上接受条款并设 `HF_TOKEN`。
  其余五个评测集不需要
- **磁盘预算**：10B token 的 JSONL 约 30GB，预 tokenize 成 `uint16` 后约 20GB，
  加上 HF 缓存，留 100GB 比较稳妥。不要下全量语料——FineWeb2-HQ 的 `cmn_Hani` 是 784GB，
  我们用 `streaming=True` 只取需要的量
- `datasets/`、`results*/`、`*.pth` 都在 `.gitignore` 里

---

## 7. 新 session 最容易踩的坑

`AGENTS.md` 里有完整清单，这里只列最容易造成「以为是 bug」的几条：

1. **GRPO 的 `loss` 恒为 0**（on-policy + `seq_mean` 聚合时）。这是数学上的必然：ratio ≡ 1 且
   组内优势之和为 0。看 `grad_norm`，别看 loss
2. **eval 曲线完全不动 ≠ 优化器坏了**。先查权重 delta 和 `grad_norm` 再怀疑算法
3. **GRPO 需要冷启动，但不能过**。策略不会输出环境格式 → 全 0 奖励 → 无梯度；
   SFT 训到完全饱和（熵极低）→ 采不出奖励方差 → 同样无梯度
4. **一次 `generate` 只能处理一个 prompt 的 group**。模型没有 left-padding 的 RoPE 偏移
5. **`generate` 产出的 token id 必须 `clone()`** 才能参与需要 backward 的前向
   （inference_mode 张量 + embedding 反向会保存索引）
6. **微批必须共用整批的分母**（`grpo_policy_loss(..., normalizer=...)`），有单测断言
   「分块损失之和 == 单次全批损失」，别「简化」掉
7. **配比 spec 的 `decontaminate.against` 为空时静默什么都不查**
8. **`single_char_frac` 只能在同一书写系统内比较**。中文单字本身就是有意义的单位，
   63% 是正常的，不是碎片化

---

## 8. 未决的问题

1. **§3.1 和 §3.2 谁先做？** 建议 3.1 先（不做就 OOM，连试都试不了），3.2 可以先用
   `--max_seq_len 1024` 绕过，等要上 2048 再改
2. **代码任务的沙箱怎么做？** 阶段 B 的核心设计问题。子进程 + 超时是底线，要不要上容器取决于
   数据源可信度
3. **租什么卡、租多久？** 取决于 §3.3 的实测结果。如果显存宽裕，`docs/corpus-plan.md` 里
   ~185M / 18B token 的档位也在射程内
4. **仓库要不要改名**（GitHub 上现在还是 `SpongeBob_LLM`）。代码里已经全部是 Whetstone 了，
   远端仓库名和 PR 链接里的路径还没改

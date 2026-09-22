# 训练语料方案（候选清单 + 配比 + 消融计划）

**目标能力**：可验证任务（算术 / 代码）为主，中英双语可用。
**状态**：候选清单已核实，配比待消融确认。旧语料全部弃用。

## 已定的决策

| 项 | 决定 | 影响 |
|----|------|------|
| 模型规模 | **~100M**（`--dim 768 --n_layers 12 --n_kv_heads 3`，vocab 32k） | 嵌入层占 25.3%，可接受；token 预算 ~10B |
| 中文语料 | **`epfml/FineWeb2-HQ` 的 `cmn_Hani`** | ODC-By，许可最干净；`CCI3-HQ` 降级为消融 #2 的对照 |
| 仓库 | **公开** | 许可约束是硬的：回避 MAP-CC（NC-ND），保留 ODC-By 署名 |
| 正式训练 | **租卡** | 不受 8GB 限制；若显存充足可把模型提到 ~185M / token 提到 18B |

具体命令：

```bash
python3 pretrain.py --dim 768 --n_layers 12 --n_heads 12 --n_kv_heads 3 \
  --tokenizer_path ./tokenizer_32k --max_seq_len 2048 --data_path datasets/prepared/train.jsonl
```

后续阶段（SFT / KD / DPO / GRPO）不需要重复声明架构——`resolve_model_config` 会从
`--pretrained_path` / `--policy_path` 指向的 checkpoint 自动继承。

---

## 0. 三个硬约束，先读这个

配比不是自由变量。下面三条会直接砍掉一部分选项。

### 0.1 词表必须重训，而且这个决定和模型规模绑死

现有 tokenizer（BPE，vocab 6400，中文语料训出来的）在本仓库实测：

| 类型 | 字符/token |
|------|-----------|
| 中文 | 1.40 |
| 英文 | 4.00 |
| **Python 代码** | **2.23** |
| 数学（LaTeX 混排） | 1.52 |

代码被切得很碎，`grpo_advantages` → `gr|p|o|_|ad|v|ant|ages`（8 个 token）。加入代码语料就必须重训词表，目标 ~32k。

但词表和模型规模是耦合的——嵌入层参数 = `vocab_size × dim`：

| 配置 | 总参数 | 词嵌入 | 嵌入占比 |
|------|--------|--------|----------|
| 当前默认（dim512 L8, vocab 6400） | 29.0M | 3.3M | 11.3% |
| 当前默认 + vocab 32k | 42.5M | 16.8M | **39.5%** |
| dim640 L12, vocab 32k | 72.6M | 21.0M | 28.9% |
| **dim768 L12, vocab 32k** | **99.5M** | 25.2M | **25.3%** |
| dim768 L16, vocab 32k | 124.3M | 25.2M | 20.2% |
| dim960 L16, vocab 32k | 184.8M | 31.5M | 17.0% |

在 29M 上挂 32k 词表，四成参数耗在嵌入层，Transformer 主体只剩 25M——这不划算。

**结论：要做中英 + 代码，模型得放大到 ~100M（dim768 / L12 / vocab 32k）。**
这同时解决了另一个问题：上一轮 GRPO 实验里 29M 模型的 accuracy 全程为 0，根因就是容量不足，
RL 只能去刷格式奖励。100M 是让 accuracy 产生方差的最低门槛。

### 0.2 Token 预算：约 10B，只够跑一次正式训练

按 `FLOPs/token ≈ 6N` 估：

| 模型 | Chinchilla 20x | 过训 100x | 100x 墙上时间* |
|------|---------------|-----------|---------------|
| 29.0M | 0.58B | 2.90B | 27 h |
| **99.5M** | 1.99B | **9.95B** | **92 h** |
| 124.3M | 2.49B | 12.43B | 115 h |

\* 假设 30k token/s，**需在你的 5060Ti 上实测后修正**。小模型常被显存带宽和 kernel launch 限制，
不是算力限制，实测值可能在 15k–60k 之间浮动。

参考：SmolLM2-135M 用了 **2T** token（约 14,800 token/参数），我们的 10B 只是它的 1/200。
所以别期待通用语言能力，目标是"能做两位数算术 + 简单 Python"。

**关键推论：10B token 的正式训练跑一次要三四天，不可能反复重跑。配比消融必须在代理模型上做。**

### 0.3 消融必须在小代理模型上做

这是 SmolLM2/SmolLM3 的标准做法（SmolLM3 在 3B 模型上跑 50–100B token 做配比消融，
正式训练才 11.2T）。我们照搬，按比例缩小：

| 用途 | 模型 | token | 单次耗时* |
|------|------|-------|----------|
| 配比消融 | 29M（现有默认，vocab 6400 或 32k 小词表变体） | 0.3–0.5B | 1.5–4 h |
| 正式训练 | ~100M | 8–10B | 3–4 天 |

一次消融几小时，一晚上能跑 3–4 组，这才有可能真的比较配比而不是拍脑袋。

---

## 1. 候选语料清单

只收开源、可下载、有论文或技术报告的。**许可证一列请在实际下载时再确认一遍**——
数据集卡片会变，而这个项目大概率要公开。

### 1.1 英文 web（教育质量过滤）

| 数据集 | 规模 | 许可证 | 备注 |
|--------|------|--------|------|
| `HuggingFaceFW/fineweb-edu` | 1.3T token | ODC-By 1.0 | 用 Llama3-70B 标注训分类器、阈值 3 筛出，删掉了 FineWeb 的 92%。消融显示优于所有其他开放 web 语料 |
| `HuggingFaceFW/fineweb-edu`（score-2 版） | 5.4T token | ODC-By 1.0 | 阈值放宽到 2，量大但略差 |
| DCLM-baseline | 3.8T token | CC-BY-4.0 | fastText 筛选。SmolLM2 发现它在 HellaSwag/CommonsenseQA 上强于 FineWeb-Edu，两者互补 |
| `HuggingFaceTB/cosmopedia-v2` | 30B token | 合成 | 合成教材/博客/故事。SmolLM2 最后阶段放了 4% |

**首选 FineWeb-Edu**。理由：单一来源、质量最高、消融证据最充分；我们的 token 预算太小，
没有余裕去调 FineWeb-Edu / DCLM 的比例（SmolLM2 在 11T 尺度上才需要调这个）。

### 1.2 中文 web

| 数据集 | 规模 | 许可证 | 坑 |
|--------|------|--------|-----|
| `BAAI/CCI3-HQ` | ~500GB（518GB 文件） | 自定义使用协议 | **HF 上是 gated**，要先同意条款。两阶段混合过滤，0.5B 模型 100B token 实验上优于 CCI3.0 / SkyPile / WanjuanV1 |
| `opencsg/chinese-fineweb-edu-v2` | 188M 条 ≈ 420B token | OpenCSG 社区许可 + Apache 2.0 | **商用需发邮件获许可**。从 MAP-CC / SkyPile / WuDao / Wanjuan / CCI3 等汇总后打分筛出 |
| `HuggingFaceFW/fineweb-2`（`cmn_Hani`） | 543.5B 词 / 636M 文档 / 1.48TB | ODC-By 1.0 | 许可最干净。按语种切分，中文子集独立下载 |
| `epfml/FineWeb2-HQ`（`cmn_Hani`） | 54.2M 文档 / 784GB | ODC-By 1.0 | FineWeb2 按 XLM-RoBERTa 分类器取 top ~10% |
| `m-a-p/MAP-CC` | 800B token | **CC BY-NC-ND 4.0** | ⚠️ 非商用**且禁止衍生**。训练模型算不算"衍生"存在争议，公开项目建议**回避** |

**建议：`FineWeb2-HQ` 的 `cmn_Hani` 作为主力，`CCI3-HQ` 作为对照。**
前者许可证最干净（ODC-By）、已经做过质量分类；后者有更完整的消融证据但需要同意协议。
两个都试一下正好是一组有意义的消融，而且这个对比在小模型双语场景下没人发表过。

### 1.3 代码

| 数据集 | 规模 | 许可证 | 坑 |
|--------|------|--------|-----|
| `HuggingFaceTB/stack-edu` | 125B token / 15 种语言 | 随 The Stack v2（原始仓库许可） | ⚠️ **只含 SWHID，不含文件内容**。要从 Software Heritage S3 逐个拉，批量下载需与 SoftwareHeritage/INRIA 签协议。SmolLM 官方脚本在 16 核 AWS 上拉 python-edu 要 ~6 小时 |
| `bigcode/the-stack-v2` 系列 | 3B+ 文件 / 600+ 语言 | 同上 | 同样只有 ID |
| `bigcode/starcoderdata` | ~250B token | 同上 | **含内容**，可直接下载 |
| `bigcode/the-stack-smol` | 小样本 | 同上 | 含内容，适合先摸清格式 |

**建议：主力用 `starcoderdata` 的 Python + Markdown 子集。**
Stack-Edu 质量更高（分类器阈值 3，Java 用 2），但"只有 ID"这个障碍对单机项目太重。
如果后面觉得代码质量是瓶颈，再去拉 Stack-Edu 的 Python 部分（只一种语言，6 小时可接受）。

语言选择：**只要 Python + Markdown**。理由是 RL 阶段要做代码可验证任务，Python 有现成的
单元测试生态；15 种语言对 100M 模型是纯粹的容量浪费。Stack-Edu 的 Python 是 21.8B token，
远超我们的预算。

### 1.4 数学

| 数据集 | 规模 | 许可证 | 备注 |
|--------|------|--------|------|
| `HuggingFaceTB/finemath`（FineMath4+） | 10B token / 6.7M 文档 | ODC-By | 只留 4–5 分。消融中 GSM8K 提升 2x、MATH 提升 6x（对比 InfiMM-WebMath）|
| FineMath3+ | 34B token / 21.4M 文档 | ODC-By | 放宽到 3–5 分 |
| Infi-WebMath4+ / 3+ | 8.5B / 20.5B token | — | FineMath 分类器应用到 InfiMM-WebMath 上 |
| `LLM360/MegaMath` | 371.6B token | ODC-BY | 目前最大的开放数学预训练集（超过 DeepSeekMath 的 120B）。含 web 279B / 代码域 28.1B / 合成 64.5B |

**建议：FineMath4+ 为主，MegaMath 的合成 Q&A 子集作为补充。**
FineMath4+ 的 10B token 刚好等于我们的全部预算，说明它在质量维度上足够密集，不需要更大的池子。

### 1.5 书籍

| 数据集 | 规模 | 许可证 | 备注 |
|--------|------|--------|------|
| `common-pile/project_gutenberg` | 75,000+ 册 | 公有领域 | Common Pile v0.1 的子集，已去 PG 页眉页脚 |
| `common-pile/pre_1929_books` | 130,000+ 册 | 公有领域（美国 1929 前） | HathiTrust / Internet Archive 的 OCR 文本，有 OCR 噪声 |
| PG-19 | 28,752 册 | Apache-2.0 | DeepMind 长文本基准。**作者明确不建议用于训练通用模型**（语言风格陈旧、历史偏见）|

**建议：书籍只占很小比例（≤5%），主要为了长依赖和正式书面语。**
中文书籍这块**没有许可证干净的开放语料**（MAP-CC 的 `zh-books` 是 NC-ND），
所以中文侧用百科和教育类 web 顶替，这是一个已知缺口而不是遗漏。

### 1.6 可验证任务数据（RL 阶段，不进预训练）

这些用于 SFT 冷启动、DPO 对照和 GRPO 的 prompt 池，接到现有 `envs/` 里。

| 数据集 | 规模 | 许可证 | 验证方式 |
|--------|------|--------|----------|
| `SynthLabsAI/Big-Math-RL-Verified` | 251,122 题 | — | 闭式答案，专为 RL 筛过：去掉多选、判断题、多问题、证明题、8/64 次 rollout 全错的题；已对 MATH-500 / Omni-MATH 做去污染。**每题带 `llama8b_solve_rate`（64 次 rollout 通过率），可以直接按难度筛** |
| `PRIME-RL/Eurus-2-RL-Data` | 数学 + 代码 | — | 数学用 LaTeX boxed 答案，代码用测试用例（源自 APPS / CodeContests / TACO / Codeforces）|
| DeepMath-103K | 103K 题 | — | 高难度、已去污染、可规则验证，每题附 3 个 R1 解 |
| `math-eval/TAL-SCQ5K`（EN + CN） | 各 5K（3K 训 / 2K 测） | **MIT** | **中英双语**数学竞赛题，带 CoT 解析，LaTeX 标准化。许可最干净，且是唯一的中文可验证数学源 |
| GSM8K / MATH | 8K / 12K | MIT / MIT | 评测标准，**必须从训练集里排除** |

**`Big-Math-RL-Verified` 的 `llama8b_solve_rate` 是个宝藏字段。**
上一轮实验的结论是"零奖励是 GRPO 的吸收态"——零方差组没有梯度。有了通过率就能直接按难度做
课程：先喂 solve_rate 在 0.2–0.8 之间的题（保证组内有奖励方差），而不是靠运气。

---

## 2. 别人的配比经验（可引用）

三份公开的、可直接对标的配比：

### SmolLM3（3B，11.2T token，三阶段）

| 阶段 | web | code | math |
|------|-----|------|------|
| 1（0→8T） | 85%（其中 12% 多语种） | 12% | 3% |
| 2（8→10T） | 75%（12% 多语种） | 15% | 10% |
| 3（10→11.1T，衰减期） | 63%（12% 多语种） | 24% | 13% |

多语种走 FineWeb2 / FineWeb2-HQ，中文是其中一个子集（配置里的 `fw2-cmn`），
所以**中文实际占比只有 1–2%**。

### SmolLM2-1.7B（11T token，四阶段）

| 阶段 | 英文 web | code | math | 其他 |
|------|---------|------|------|------|
| 2（6→8T） | 75%（FWE:DCLM = 60:40） | 20% | 5% | |
| 3（8→10T） | —（FWE:DCLM 调成 40:60） | Stack-Edu | ~10% | |
| 4（衰减，10→11T） | 58% | 24% | 14% | Cosmopedia v2 4% |

### CCI3.0-HQ 的混合消融（0.5B 模型，100B token）

**英文 : 代码 : 中文 = 60 : 10 : 30**，英文用 FineWeb-Edu，代码用 StarCoder。
这是唯一一份明确给出中文占比的公开配比，也是我们最该对标的。

### 两条方法论要点

1. **小模型用单阶段。** SmolLM2 的 135M / 360M 没用多阶段，而是"单阶段 + 全程高质量数据"，
   并且 Stack-Edu / InfiMM-WebMath / FineMath **从一开始就加入**。我们 10B 的预算根本撑不起
   多阶段，所以照 135M 的做法。
2. **别重复太多遍。** SmolLM2 明确控制在"大多数数据集 4–5 个 epoch"以内
   （依据 Muennighoff 等人的数据受限 scaling law）。我们每个子集都要算一下重复次数。

---

## 3. 建议的起点配比

单阶段、全程高质量、代码和数学从第一步就加。总预算 ~10B token。

| 领域 | 占比 | token | 数据集 | 池子大小 | 重复次数 |
|------|------|-------|--------|---------|---------|
| 中文 web | 30% | 3.0B | FineWeb2-HQ `cmn_Hani` | 784GB | ≪1 |
| 英文 web | 28% | 2.8B | FineWeb-Edu | 1.3T token | ≪1 |
| 代码 | 20% | 2.0B | starcoderdata（Python + Markdown） | ~30B+ | ≪1 |
| 数学 | 14% | 1.4B | FineMath4+ | 10B token | ≪1 |
| 书籍 | 5% | 0.5B | Common Pile Gutenberg / pre-1929 | 大 | ≪1 |
| 合成教材 | 3% | 0.3B | Cosmopedia v2 | 30B token | ≪1 |

**没有任何子集需要重复采样**，这在 10B 预算下是好事——不用担心记忆化。

选这组数的理由：

- **中文 30%** 直接对标 CCI3.0-HQ 的消融配比，是唯一有公开依据的中文占比。
- **代码 + 数学 = 34%** 接近 SmolLM2 最终阶段的 38%（24% + 14%），而不是它第一阶段的 15%。
  因为我们的目标是可验证任务，而且 SmolLM2-135M 的经验支持小模型全程高质量、不做课程。
- **书籍 5% / 合成 3%** 只求来源多样性，不指望它们贡献能力。

### 与目标能力的一致性检查

RL 阶段要做算术和代码。这组配比里 34% 是代码和数学，且数学源（FineMath4+）本身就是
带推理过程的教育类内容——这正是 GRPO 冷启动需要的分布。上一轮的失败教训是
"冷启动不够，accuracy 没有方差"，这次从预训练就把数学密度提上去。

---

## 4. 消融计划

按信息量排序。每组都是 29M 代理模型 + 0.3–0.5B token，评测用固定 holdout PPL + 少量任务级指标。

| # | 变量 | 组别 | 想回答的问题 |
|---|------|------|-------------|
| 1 | **中文占比** | 0% / 15% / 30% / 45% | 双语在 100M 尺度上的代价有多大？这是本项目最该自己做的消融，因为没人发表过小模型双语配比 |
| 2 | 中文语料源 | FineWeb2-HQ vs CCI3-HQ vs 两者混合 | 许可干净的那个是否够用 |
| 3 | 代码占比 | 10% / 20% / 30% | 代码挤占语言能力的临界点 |
| 4 | 数学占比 | 7% / 14% / 21% | 对下游 GSM8K 式任务的边际收益 |
| 5 | 词表大小 | 16k / 32k / 48k | 压缩率提升 vs 嵌入层参数占用的权衡 |
| 6 | 单阶段 vs 两阶段 | 全程同配比 vs 后 20% 上采样代码数学 | SmolLM2 的多阶段结论在 10B 尺度上还成立吗 |

消融 1 和 5 优先——它们决定其余所有配置。

**评测指标**（不能只看 PPL，PPL 跨配比不可比，因为 token 分布不同）：
- 中英各自的 holdout PPL（同一 tokenizer 下才可比）
- 算术任务准确率（复用 `envs/arithmetic.py` 的评测集）
- 代码：简单函数补全的执行通过率
- 语言混淆率：中文 prompt 下输出英文的比例（双语小模型的典型故障）

---

## 5. 获取与许可的坑（已核实）

1. **Stack-Edu / The Stack v2 只有 SWHID，没有文件内容。** 内容在 Software Heritage 的 S3，
   批量下载需与 SoftwareHeritage 和 INRIA 签协议。官方脚本拉 python-edu 在 16 核 AWS 上约 6 小时。
   → 先用 `starcoderdata`。
2. **`m-a-p/MAP-CC` 是 CC BY-NC-ND 4.0**，非商用且禁止衍生。公开项目建议回避。
3. **`BAAI/CCI3-HQ` 在 HF 上是 gated**，需先同意使用协议（"不得用于伤害人类受试者的实验"）。
4. **`opencsg/chinese-fineweb-edu-v2` 商用需邮件获得许可**（OpenCSG 社区许可 + Apache 2.0）。
5. **FineWeb / FineWeb2 是 ODC-By，且受 CommonCrawl 使用条款约束**，需保留署名。
6. **别下全量。** FineWeb2 `cmn_Hani` 是 1.48TB，我们只要几 GB。用 `streaming=True`
   或只下指定的几个 parquet 分片。
7. **去污染是必做项。** SmolLB2 的做法是对 GSM8K / MATH / MMLU 做 13-gram 匹配 +
   最长公共子序列重叠率 0.6 阈值。仓库里 `datatools.dedup --against` 目前是 prompt 精确匹配，
   需要升级成 n-gram 匹配。

---

## 6. 数据管线（已实现）

配比写在 `configs/mixture_v1.json` 里，`prepare.py` 按它跑完整流程：

```bash
# 先看一眼各源会取多少、被过滤掉多少，不落盘
python3 -m datatools.prepare configs/mixture_v1.json --dry_run

# 正式产出（每源一个 JSONL + train/val/holdout + manifest）
python3 -m datatools.prepare configs/mixture_v1.json --out_dir datasets/prepared
```

流程顺序是 **拉取 → 质量过滤 → 去重 → 去污染 → 划分 → manifest**：

| 阶段 | 模块 | 要点 |
|------|------|------|
| 拉取 | `prepare.py` | HF 流式（`streaming=True`），按 token 预算边数边停，不下全量 |
| 过滤 | `filters.py` | 长度、语种比例、重复度、符号/数字占比、行级重复；每条拒绝都归因到具体规则 |
| 去重 | `dedup.py` | 精确 + MinHash 近重复（源内） |
| 去污染 | `decontaminate.py` | 13-gram 重叠 + 可选 LCS 比例，CJK 按字符切、拉丁按词切 |
| 划分 | `split.py` | 按内容哈希确定性划分，重跑结果一致，同文档不会跨 split |

`manifest.json` 记录每源实际取到的 token / 文档数、各条过滤规则的拒绝计数、去重和去污染的删除量、
随机种子和 HF revision——这是配比消融能对得上号的前提。

## 7. 下一步

1. 用清洗后的语料重训 tokenizer（vocab 由消融 #5 定，起点 32k），顺手修掉
   `train_tokenizer.py` 里硬编码的 `pretrain.jsonl` 路径
2. 跑消融 #1（中文占比）和 #5（词表大小）——它们决定其余所有配置
3. 按定下的配比产出正式数据集，租卡做 ~100M / ~10B token 的正式预训练
4. 重建 SFT / DPO / GRPO 阶段的数据（可验证任务部分见 §1.6）

---

## 引用

- FineWeb / FineWeb-Edu：Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, 2024
- FineWeb2：HuggingFace, 2024（1000+ 语种）
- FineWeb2-HQ：Messmer et al., 2025
- SmolLM2：Allal et al., *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*, arXiv:2502.02737
- SmolLM3：HuggingFace blog, 2025
- CCI3.0-HQ：Wang et al., arXiv:2410.18505
- CCI4.0：arXiv:2506.07463
- OpenCSG Chinese Corpus：arXiv:2501.08197
- Nemotron-CC：NVIDIA, 2024（6.3T token）
- DCLM：Li et al., 2024
- MegaMath：LLM360, COLM 2025
- Common Pile v0.1：Kandpal et al., arXiv:2506.05209
- PG-19：Rae et al., arXiv:1911.05507
- Big-Math：arXiv:2502.17387
- DeepMath-103K：arXiv:2504.11456
- 数据受限 scaling law：Muennighoff et al., *Scaling Data-Constrained Language Models*, 2023
- DoReMi（配比优化）：Xie et al., 2023

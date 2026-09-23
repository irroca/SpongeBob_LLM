# Whetstone

> *A whetstone doesn't add metal — it removes what isn't an edge.*

从零实现的中英双语小语言模型，目标能力是**可验证任务**（算术、代码）。全栈自己写，包括数据管线和
强化学习：**语料构建 → Pretrain → SFT → 知识蒸馏 → DPO → GRPO/RLVR**。

RL 部分不依赖 TRL/veRL：可验证奖励环境、组相对优势、clipped policy loss、k3 KL 全部从零实现，
并把 Dr. GRPO / DAPO 的几个关键改动做成开关而不是分叉代码，方便做消融。

> 本仓库定位是**可深挖的学习/研究项目**，不是生产级大模型平台。无多卡并行、无推理服务、
> 无量化实现、无 PPO critic。规模刻意做小，为的是把算法和训练现象看清楚。

**当前状态**：算法链路和数据管线完成且有单测覆盖；语料方案已定（见
[`docs/corpus-plan.md`](docs/corpus-plan.md)），目标模型 ~100M / ~10B token，正式训练尚未开始。
仓库里的 GRPO 实验结果是 29M 玩具规模的**实现验证**，不是能力声明（见
[`docs/experiments.md`](docs/experiments.md)）。

> 接手开发请先读 [`docs/status.md`](docs/status.md)：项目状态、待办、长期路线，以及
> **正式训练之前必须先解决的三件工程问题**。

## 环境

```bash
# CPU 环境（如无 GPU）建议先装 CPU 版 PyTorch：
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python3 -m pytest tests/ -q     # 全部 CPU、无网络、不需要 checkpoint
```

`tokenizer/zh_6400/` 是**遗留分词器**（BPE，vocab 6400，只在中文语料上训过），保留它是为了让
CPU 单测和 smoke 跑得起来。它在代码上的压缩率只有 2.23 字符/token，双语 + 代码语料需要
重训一个 32k 词表的版本（见下文「分词器」一节）。

## 仓库结构

```text
config.py  model.py  dataset.py  losses.py  train_utils.py  rollout.py   # 核心库
pretrain.py  sft.py  distill.py  dpo.py  grpo.py                         # 五个训练阶段
eval_ppl.py  chat.py  analyze_grpo.py  train_tokenizer.py                # 评估与工具
datatools/   envs/   configs/   tests/   docs/   tokenizer/
```

| 路径 | 说明 |
|------|------|
| `model.py` / `config.py` | 模型与配置（RoPE、RMSNorm、SwiGLU、可选 GQA、权重共享）|
| `dataset.py` | Pretrain / SFT / Preference(DPO) 数据集与 assistant loss mask |
| `losses.py` / `train_utils.py` | CE/KD/DPO/GRPO loss；共享训练工具与 CLI |
| `datatools/` | 语料管线：统计、过滤、去重、去污染、划分、配比编排、评测集拉取 |
| `envs/` | 可验证奖励环境（RLVR）与各阶段数据生成 |
| `rollout.py` | GRPO 在线采样：分组 rollout、completion mask、logprob |
| `pretrain.py` / `sft.py` / `distill.py` / `dpo.py` / `grpo.py` | 五个阶段的训练入口 |
| `eval_ppl.py` / `chat.py` / `analyze_grpo.py` | 困惑度评估、交互式生成、RL 指标分析 |
| `configs/` | 配比 spec（`mixture_v1.json`）|
| `docs/corpus-plan.md` | 语料候选清单、许可证、配比与消融计划 |
| `docs/experiments.md` | 实验协议与已记录的跑批结果 |

## 从零到训练：完整顺序

```bash
# 1. 拉评测集（去污染要用；不做这步去污染就是空转）
python3 -m datatools.fetch_evals --decontamination_only --update_spec configs/mixture_v1.json

# 2. 按配比构建语料（拉取 → 过滤 → 去重 → 去污染 → 划分 → manifest）
python3 -m datatools.prepare configs/mixture_v1.json --out_dir datasets/prepared

# 3. 在清洗后的语料上重训分词器
python3 train_tokenizer.py --data datasets/prepared/train.jsonl --out tokenizer/v1_32k --vocab_size 32768

# 4. 五个阶段（架构只在第一步声明，后续自动继承）
python3 pretrain.py --dim 768 --n_layers 12 --n_heads 12 --n_kv_heads 3 \
  --tokenizer_path tokenizer/v1_32k --data_path datasets/prepared/train.jsonl --save_dir results
python3 sft.py     --pretrained_path results/pretrain_final.pth --data_path datasets/sft.jsonl
python3 dpo.py     --policy_path results/sft_final.pth --data_path datasets/preference.jsonl
python3 grpo.py    --policy_path results/dpo_final.pth --env arithmetic
```

## 模型结构由 CLI 决定

五个训练脚本加上 `eval_ppl.py` / `chat.py` 都通过 `train_utils.add_model_args` 暴露
`--dim` / `--n_layers` / `--n_heads` / `--n_kv_heads` / `--hidden_dim` / `--dropout` /
`--rope_theta` 以及 `--tokenizer_path`，做尺寸消融不需要改源码：

```bash
python3 pretrain.py --dim 256 --n_layers 6 --n_kv_heads 2 --data_path datasets/pretrain.jsonl
python3 sft.py --pretrained_path results/pretrain_final.pth   # 架构自动沿用，不必重复声明
```

`resolve_model_config` 的优先级是 **显式 CLI > checkpoint 记录 > 库默认值**。这一层不只是省参数：
`load_state_dict(strict=False)` 在形状不匹配时会报错，但对**缺失的键是静默容忍**的——把 6 层的
checkpoint 加载进 8 层模型，多出来的两层会保持随机初始化且毫无提示。现在这种情况会告警。

- `*_final.pth` 仍然是纯 `state_dict`，但同时会写一个 `*.config.json` sidecar。原因是
  **`n_heads` 无法从张量形状反推**（`head_dim = dim // n_heads`，所以 `wq` 永远是 `dim × dim`，
  只有 kv/q 的比例可见）。没有 sidecar 时只能假设 `n_heads` 并告警。
- checkpoint 的 `vocab_size` 与 tokenizer 不一致会直接报错。**重训 tokenizer 会让旧权重失效**，
  这一点在换语料时几乎必踩。
- 因此 `distill.py` 现在能做**真正的跨尺寸蒸馏**：teacher 和 student 各自从自己的 checkpoint
  解析架构，只要求共享 vocab。

## 数据工具（`datatools/`）

四种数据 schema（`text` / `conversations` / `prompt+chosen+rejected` / `question+answer`）
全部自动识别，所以没有任何工具需要 `--schema` 参数。

### 一条命令跑完整管线

配比写在 spec 里（见 `configs/mixture_v1.json`），`prepare` 按它执行
**拉取 → 质量过滤 → 精确去重 → 去污染 → 划分 → manifest**：

```bash
python3 -m datatools.prepare configs/mixture_v1.json --dry_run          # 先看各源会取多少
python3 -m datatools.prepare configs/mixture_v1.json --out_dir datasets/prepared
python3 -m datatools.prepare configs/mixture_v1.json --scale 0.001      # 千分之一预算试跑管线
```

整条链路是**流式**的：配比按 token 计，而语料按文档和字节发布，所以只能边 tokenize 边记数、
取满即停。10B token 是约 30GB 文本，任何一步都不能全量进内存。

输出是 `train/val/holdout.jsonl` 加一个 `manifest.json`，后者记录每源实际取到的 token 和文档数、
**每条过滤规则各拒绝了多少**、去重和去污染的删除量、以及种子。没有这些，配比消融之间无法对账。

报告里两个值得盯的信号：

```text
source                 target       tokens   fill      docs      read  kept%  tok/doc
zh_web                 120000        20268   17%       601       653   92%       34
  ! zh_web ran out of data at 17% of its budget
```

- `fill < 100%` 且标了 `ran out of data`：这个源被**悄悄降权**了，实际配比已经不是你写的那个
- `kept%` 是真实的过滤通过率（缓冲区里取进来但没用上的记录会从 `read` 里扣掉，
  否则一个干净的源会看起来像被过滤掉了大半）

### 单独使用

```bash
# 语料统计：schema 合法性、长度分位数、语种混合、重复率、单文档重复度、preference 长度偏置
python3 -m datatools.stats datasets/raw.jsonl --tokenizer ./tokenizer/zh_6400 --json stats.json

# 去重：精确 + MinHash 近重复
python3 -m datatools.dedup datasets/raw.jsonl --out datasets/clean.jsonl --threshold 0.8

# 去污染：13-gram 重叠 + 可选 LCS 比例（SmolLM2 的做法）
python3 -m datatools.decontaminate datasets/train.jsonl --out datasets/clean.jsonl \
  --against datasets/gsm8k.jsonl datasets/math.jsonl --n 13 --lcs_threshold 0.6

# 确定性划分（按内容哈希，重跑一致、语料增长时已有划分不变）
python3 -m datatools.split datasets/clean.jsonl --out_prefix datasets/v1

# tokenizer 压缩率：算语料的 token 量，以及比较多个候选词表
python3 -m datatools.tokenizer_stats --probe datasets/zh.jsonl
python3 -m datatools.tokenizer_stats datasets/zh.jsonl --tokenizer ./tok_16k ./tok_32k
```

### 各模块的要点

`stats` 报几个直接决定能不能训的东西：

- **malformed 行**按行号报出来并跳过，大 dump 里的一行坏数据不该让整个任务挂掉
- **repetition_ratio**（单文档内重复的字符 n-gram 占比）能抓出 boilerplate 循环和退化爬虫结果
- **preference 的长度偏置**：`chosen` 如果系统性更长，DPO 会顺带学到「越长越好」，
  超过 70% 时会直接告警。这件事一旦开训就看不见了

`filters` 的每条规则返回的是**拒绝原因的名字**而不是布尔值。过滤这一步真正有用的输出不是留下的
集合，而是**哪条规则删了多少**——一个阈值静默删掉八成语料是 bug，只有归因才看得见。
阈值故意没有调优：先用 `stats` 看真实分位数，再写进 spec。

`dedup` 的 MinHash 是直接在 numpy 上实现的（不依赖 `datasketch`）：

- shingle 用**字符 n-gram**，因为中文没有空格分词
- 置换系数做了上界约束，`a*h + b < 2^63`，uint64 不会回绕——这是一个诚实的 universal hash
  族，而不是依赖溢出行为
- LSH 只负责挑候选，**每个候选都会用完整签名复核**，所以分带只影响召回和速度，
  不会引入低于阈值的误判。`--bands` / `--rows` 可以手动调这个权衡
- **内存上界**：每条存活文档一个签名（128 个置换约 1KB），大约能撑 100–200 万条文档。
  十亿 token 级别的语料要按源分文件跑，这也是 `prepare` 内联只做流式精确去重的原因

`decontaminate` 是真正的污染检查（`dedup --against` 只做精确 prompt 匹配）：

- 沿用 SmolLM2 的做法：**13-gram 重叠 + 可选 LCS 重叠比例 0.6**。后者用来把巧合撞上的
  13-gram 判回干净
- **CJK 需要自己的切分**：按空格切词的 13-gram 在中文里不存在。`text_units` 把每个汉字当
  一个单元、拉丁/数字连续段当一个词，于是 13-gram 在英文是 13 个词、在中文是 13 个字，
  两者都约等于一个句子片段
- 评测项**逐字段单独建索引**，而不是拼成一条。拼接会插入 `=>`、role 前缀这种自然文本里
  不存在的分隔符，反而让只引用了题干的网页漏过去
- 短于 13 个单元的评测项按**自身长度**建索引——否则一道 10 个词的题永远匹配不上只会产生
  13 单元窗口的长网页
- **已知限制**：极短答案（如 `42`，少于 5 个单元）只能靠精确相等匹配。把任何出现 `42`
  的文档都判为污染会把语料删空，所以保护主要来自题干。这条在测试里被显式断言，
  是已知性质而不是意外

`split` 按**内容哈希**划分而不是按位置或打乱的下标，换来两个对消融很重要的性质：重跑结果一致，
以及语料增长时已有文档不会被重新洗牌（验证曲线跨数据版本仍可比）。`holdout` 是任何阶段都不训的那份。

### 评测集与去污染

```bash
python3 -m datatools.fetch_evals --all --out_dir datasets/eval
python3 -m datatools.fetch_evals --decontamination_only --update_spec configs/mixture_v1.json
```

从 HF 拉取并转成仓库的 `{"question","answer"}` schema，所以同一个文件既能喂
`datatools.prepare` 的去污染，也能直接给 `grpo.py --eval_path` 当评测集：

| set | 规模 | 用途 |
|-----|------|------|
| `gsm8k` | 1319 | 去污染主目标（SmolLM2 用的也是它）+ 评测 |
| `math500` | 500 | 标准 MATH 评测子集 |
| `tal_scq5k_cn` / `tal_scq5k_en` | 各 2000 | MIT 许可的中英竞赛数学，唯一干净的中文可验证源 |
| `mmlu` | 14042 | 只做去污染 |
| `big_math` | 251k | GRPO 的 prompt 池（gated，需 HF token）|

转换不是直接搬字段：GSM8K 的答案要从 `#### N` 里抽出来、CoT 留在 `solution` 字段；
TAL-SCQ5K 的 `answer_value` 只是选项字母（`B`），要解析 `answer_option_list` 换成选项**内容**，
否则不可验证；MMLU 的 `answer` 是下标，要换成选项文本。`solution` 字段会被
`record_parts` 一起索引——**只抄了解答、没抄题目的网页同样是泄漏**。

`big_math` 保留了 `llama8b_solve_rate`（每题 64 次 rollout 的通过率）。这是做难度课程的关键：
零奖励是 GRPO 的吸收态（组内全错 → 无奖励方差 → 无梯度），有了通过率就能按难度带筛题，
而不是靠运气碰方差。

实跑验证：1319 道 GSM8K 索引出 3957 个片段，把三种泄漏形态（原题、只抄解答、题目埋在长网页里）
各埋一条进 300 篇干净文档，三条全部命中，零误杀。

### 分词器

`tokenizer_stats` 解决两个问题：**语料到底有多少 token**（配比是按 token 算的，而语料是按文档/
字节发布的），以及**这个词表配不配这份数据**。遗留的 6400 词表实测：

| 领域 | 字符/token | 单字符 token 占比 |
|------|-----------|------------------|
| 中文 | 1.40 | 63.3% |
| 英文 | 4.00 | 14.7% |
| **代码** | **2.23** | 42.5% |

英文和代码都是 ASCII，代码还更重复，正常词表下代码的压缩率不该差于散文，这里却低了 44%
（`grpo_advantages` → `gr|p|o|_|ad|v|ant|ages`）。**加代码语料必须重训词表。**
注意 `single_char_frac` 只能在同一书写系统内比较——中文单字本身就是有意义的单位，
63% 是正常的，不是碎片化。

重训用 `train_tokenizer.py`，它复用 `datatools.records`，所以分词器看到的文本与训练阶段
完全一致（包括展平后的对话）：

```bash
python3 train_tokenizer.py --data datasets/prepared/train.jsonl --out tokenizer/v1_32k --vocab_size 32768
python3 -m datatools.tokenizer_stats --probe --tokenizer tokenizer/v1_32k tokenizer/zh_6400
```

**词表大小和模型规模必须一起定**：嵌入层是 `vocab_size × dim`，32k 词表在 29M 模型上占
39.5% 的参数，在 ~100M（`dim=768, n_layers=12`）上占 25.3%。这也是本项目把模型定在 100M 的原因。
语料方案（候选清单、许可证坑、配比与消融计划）见 [`docs/corpus-plan.md`](docs/corpus-plan.md)。

## 快速跑通（CPU smoke）

使用仓库内置 fixtures（无需外部数据）：

```bash
python3 pretrain.py --data_path tests/fixtures/pretrain_tiny.jsonl \
  --epochs 1 --batch_size 2 --max_seq_len 128 --save_dir results --device cpu --dtype float32

python3 sft.py --data_path tests/fixtures/sft_tiny.jsonl \
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

printf '磨刀石是用来做什么的？\nquit\n' | python3 chat.py \
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

四个训练入口（`pretrain.py` / `sft.py` / `distill.py` / `dpo.py`）共享同一套 CLI 参数，由
`train_utils.add_common_train_args(parser, **overrides)` 统一添加（`--save_dir` / `--epochs` /
`--batch_size` / `--learning_rate` / `--device` / `--use_wandb` / `--wandb_project` / `--dtype` /
`--num_workers` / `--accumulation_steps` / `--grad_clip` / `--log_step` / `--save_step` /
`--max_seq_len` / `--data_path` / `--resume_from` / `--seed`）；每个脚本通过关键字参数覆盖自己的默认值
（如 `distill.py` 用 `wandb_project="Whetstone-Distill"`），再 `add_argument` 自己的额外参数
（如 `--teacher_path` / `--beta`）。

- `--device` 统一默认 `"cuda" if torch.cuda.is_available() else "cpu"`（四个训练脚本 + `eval_ppl.py` +
  `chat.py` 一致；此前 `pretrain.py`/`sft.py`/`distill.py`/`dpo.py` 默认写的是 `"cuda:0"`）。
- `--use_wandb True --wandb_project ...`：四个训练脚本现在都支持（`distill.py`/`dpo.py` 是本轮新增，
  之前只有 `pretrain.py`/`sft.py` 有）。日志由 `train_utils.init_wandb_if_needed(args, run_name=...)`
  统一处理：`use_wandb=False` 时直接返回 `None`（不 import）；为 `True` 时才 `import swanlab as wandb`
  并 `wandb.init(...)`，随后训练循环里 `if wandb is not None: wandb.log({...})`。`swanlab` 是可选依赖
  （见 `requirements.txt`），未安装时打开 `--use_wandb` 会直接抛 `ModuleNotFoundError`。

## 测试

```bash
python3 -m pytest tests/ -q
```

## 模型配置

`config.LLMConfig` 的默认值是遗留的小配置，留作 CPU 测试和消融代理模型用：

```python
LLMConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=8, vocab_size=6400, max_seq_len=1024)
# ≈ 29M params；设置 n_kv_heads < n_heads 即启用 GQA
```

正式训练的目标配置是 **~99.5M**（词表 32k 下嵌入层占 25.3%，见「分词器」一节）：

```bash
--dim 768 --n_layers 12 --n_heads 12 --n_kv_heads 3 --max_seq_len 2048
```

| 配置 | 总参数 | 嵌入占比 |
|------|--------|---------|
| `dim=512, L=8`, vocab 6400（默认/代理） | 29.0M | 11.3% |
| `dim=512, L=8`, vocab 32k | 42.5M | 39.5% |
| **`dim=768, L=12`, vocab 32k（目标）** | **99.5M** | 25.3% |
| `dim=960, L=16`, vocab 32k | 184.8M | 17.0% |

## 已知行为与限制

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
- **`generate` 支持 batch>1 且逐行独立判断 EOS**：`Whetstone._stream_generate` 维护一个
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
- **`tokenizer/zh_6400` 只在中文上训过**，代码压缩率 2.23 字符/token。双语 + 代码正式训练前
  必须重训词表（见「分词器」）。用旧词表训出来的 checkpoint 与新词表**不兼容**，
  `resolve_model_config` 会在 `vocab_size` 不匹配时直接报错。
- **`datatools.dedup` 的近重复去重有内存上界**（每条文档约 1KB 签名，约 100–200 万条），
  所以 `prepare` 内联只做流式精确去重，近重复要按源分文件单独跑。
- **极短评测答案的去污染保护较弱**（少于 5 个单元时退化为精确匹配）。保护主要来自题干。
- 无分布式训练、无 FlashAttention 绑定、无服务化 API、无 INT8/INT4、无 PPO critic、无 PRM。
- fixtures / CPU smoke **不能**代表语言能力；真实数字见 `docs/experiments.md`。
- 旧版伪「特殊 token 加权蒸馏」已移除，现为真实 KD。

## License

MIT — see [LICENSE](LICENSE).

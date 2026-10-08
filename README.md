# 低正类比例下的中文反讽识别：上下文建模与 Recall 约束迭代训练

Chinese Sarcasm Detection under Low Class Prevalence: Context Modeling and Recall-Constrained Iterative Training

本研究识别中文社交媒体评论与弹幕中的上下文依赖型反讽。评论是判断对象，原帖提供解释其字面立场与真实意图的证据。研究沿三条相互衔接的路线展开：构建并验证大规模标注语料，通过模型、输入形式和长度对照确定上下文建模方案，再在低正类比例下以假阳性挖掘和 Recall 约束优化筛选效果。

在正类占比 1.5% 的评论流中，负例数量约为正例的 66 倍，少量负例误判就会占据候选池。因此，研究以保留至少 80% 反讽的 Recall 为约束，提升候选池的 Precision，使模型输出适合优先人工复核。

```text
任务定义 → 200 万条评论—上下文采集与清洗
        → 3 万条开发样本上的标注 Prompt 优化、人工抽检与冻结
        → 三模型标注，形成 170.8 万条语料
        → 中文 ModernBERT 继续预训练
        → LSTM / BERT / ModernBERT 选型
        → 仅正文 / 上下文拼接 / Prompt+MLM 对照
        → 输入长度诊断，确定 1280 token
        → V1 平衡扩增训练 → V2–V4 难负例混合训练与 Recall 约束选优
        → 冻结 V4 → 两套独立人工测试 → 候选评论筛选
```

## 完整研究路线

### 1. 将反讽定义落实为可标注的任务

每条样本由目标评论或弹幕及对应原帖文本组成。正类需要同时满足五项条件：

1. 能识别评论的字面立场。
2. 能识别评论实际表达的意图立场。
3. 两种立场存在矛盾或反转。
4. 上下文证据支持反讽解释，强于真诚表达的解释。
5. 能识别反讽指向的对象。

这一定义将判断依据落实到评论、原帖及其关系，作为标注 Prompt、人工复核和最终测试的统一标准。

### 2. 从实时舆情流构建评论—上下文语料

从覆盖多个社交媒体来源与话题的实时舆情流采集 **2,000,000** 条评论—上下文对。标注前依次处理：

- 统一空白字符，按标准化后的评论正文去重。
- 去除缺失内容、低信息文本和已知模板水军内容。
- 用字符 4-gram 的新增信息比例压缩逐帧重复 OCR 上下文。
- 将清洗后的 OCR 上下文限制为 2,000 字符。

清洗后得到 **1,714,227** 条待标注样本，保留评论及其原帖证据。

### 3. 开发、验证并冻结标注 Prompt，再扩展到全量语料

先在 **30,000** 条开发样本上进行标注 Prompt 的误差分析与规则修订。每轮从模型预测正类和预测负类中各抽取 200 条，由人工复核并据错误类型修改规则。冻结时，两类抽检确认率分别为 **92%** 和 **94%**。

随后用三批新样本验证冻结 Prompt；每批仍从两个预测类别各抽取 200 条，三批中两类确认率均超过 **90%**。验证后用冻结 Prompt 完成大规模标注。

全量标注分别查询 **Qwen Plus、DeepSeek V4 Flash 和 DeepSeek V4 Pro**，每个模型输出 0（非反讽）、1（反讽）或 2（弃权）。论文的聚合协议为：至少两票 1 时记为正类，全为 2 时剔除，其余记为负类。剔除 **6,235** 条一致弃权样本后，得到：

| 类别 | 条数 | 比例 |
| --- | ---: | ---: |
| 反讽 | 26,250 | 1.5369% |
| 非反讽 | 1,681,742 | 98.4631% |
| 合计 | **1,707,992** | 100% |

这些弱监督标签用于训练、候选负例挖掘和迭代选优；最终评价另用人工标注数据。

### 4. 继续预训练中文长上下文底座

原帖同时包含长文本与 OCR 提取的视频字幕。为表示这些中文社交媒体内容，研究在 ModernBERT 架构上进行中文掩码语言建模继续预训练，得到 `ModernBertHansir-zh-8k-base`：

- 采用 **151,666** 规模的 ByteLevel BPE 词表，支持 **8,192 token**。
- 沿用 ModernBERT 的局部/全局交替注意力与 RoPE：每第三层使用全局注意力，其余层的局部窗口配置为 128 token。
- 在 **21,136,007** 条中文序列上训练 1 epoch，共 **660,501** 步，更新编码器与 MLM 预测头。
- 留出集 perplexity 为 **4.8102**。

该底座为下游同时表示评论和原帖、复用 MLM 头进行分类提供基础。

### 5. 用编码器、输入形式和长度实验确定分类方案

#### 编码器选型

先在历史 **5,250** 条、正负类各占 50% 的测试集上比较仅输入评论正文的 LSTM、BERT 和 ModernBERT。

| 编码器（仅正文） | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| LSTM | 54.55% | 40.00% | 46.16% |
| BERT | 60.00% | 70.00% | 64.62% |
| ModernBERT | **63.64%** | **70.00%** | **66.67%** |

ModernBERT 在三者中取得最高 Precision 和 F1，因此作为后续输入形式实验的固定编码器。

#### 上下文与 Prompt+MLM 对照

固定 ModernBERT 后，比较仅正文、直接拼接原帖上下文，以及 Prompt+MLM：

| 输入形式 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 仅正文 | 63.64% | 70.00% | 66.67% |
| 正文与原帖直接拼接 | 67.14% | 72.31% | 69.63% |
| Prompt+MLM | **82.00%** | **83.08%** | **82.54%** |

加入原帖使 F1 从 66.67% 提升至 69.63%；Prompt+MLM 进一步提升至 82.54%。它将评论与原帖组织为问句，复用预训练 MLM 头预测“否/是”：

```text
结合 {retweeted_content} ，判断{text}是否是反讽表达，[MASK]
```

在 `[MASK]` 位置取两个候选词的 logits，归一化为二类概率。候选词“否”对应标签 0，“是”对应标签 1。

#### 输入长度诊断

再按截断前的 token 长度将样本分为 `[0,50)`、`[50,256)`、`[256,512)`、`[512,∞)` 四组，观察不同随机种子训练中的分组错误率：

| 输入 | 最大长度 | 观察到的错误率 |
| --- | ---: | --- |
| 仅正文 | 384 token | 最短组为 24–26% |
| 正文与上下文 | 384 token | 最长组为 21–26%，为四组最高 |
| 正文与上下文 | 512 token | 各组约 17–21% |
| 正文与上下文 | 1280 token | 各组收窄至 15.2–16.6% |

仅正文时，短评论组的高错误率提示需要原帖证据；加入上下文后，最高错误率出现在截断最集中的长输入组。扩大输入预算后，各长度组的错误率降低并趋于接近，因此 V1–V4 使用 **1280 token**，并采用结构感知截断：优先保留 `[MASK]`、目标评论和固定问句，首先移除超长的原帖上下文。

### 6. 为不同实验阶段设置数据划分

历史编码器与输入形式实验按 **8:1:1** 划分训练、开发和测试数据。后续 V1–V4 使用独立的平衡诊断集及固定真实比例选优集。各集合的职责如下：

| 集合 | 总数 | 正类 | 正类比例 | 用途 |
| --- | ---: | ---: | ---: | --- |
| 历史基线训练集 | 42,000 | 21,000 | 50% | 编码器与输入形式训练 |
| 历史基线开发集 | 5,250 | 2,625 | 50% | 基线开发与配置选择 |
| 历史基线测试集 | 5,250 | 2,625 | 50% | 报告编码器、输入形式对照结果 |
| V1–V4 平衡诊断集 | 10,500 | 5,250 | 50% | 观察迭代后的决策边界 |
| 固定真实比例选优集 | 50,000 | 750 | 1.5% | 每轮 Recall 约束下选择最高 Precision |
| 人工真实比例测试集 | 50,000 | 750 | 1.5% | 冻结 V4 的主要最终评价 |
| 人工平衡测试集 | 4,000 | 2,000 | 50% | 冻结 V4 的辅助区分能力评价 |

平衡数据用于训练和诊断，真实比例数据用于评价低正类比例下的误报成本。模型选优始终采用训练前固定的 **0.5** 阈值。

### 7. 从 V1 初始化转入 Recall 约束的迭代训练

#### V1：保持 1:1 比例，扩大训练覆盖

初始训练集包含 21,000 正例和 21,000 负例。每条正例额外复制一次，同时加入 21,000 条未使用负例，得到 **84,000 行**：42,000 正类行对应 21,000 个唯一评论—上下文对，负类为 42,000 条唯一样本。V1 从中文预训练底座开始训练。

V1 在 10,500 条平衡诊断集上的 Precision 为 **86.10%**，但在正类占比 1.5% 的选优集上只有 **9.22%**，并产生 **6,814** 个假阳性。这个差异推动训练重点转向真实比例下的假阳性控制。

#### V2–V4：固定正例，混合难负例与普通负例

每轮从未使用的弱监督负例中构建候选池，排除初始数据划分、固定选优集和先前轮次已用负例。上一轮模型在阈值 0.5 下将其预测为反讽的样本进入难负例池。

每轮训练集包含：

| 组成 | 数量 | 作用 |
| --- | ---: | --- |
| 固定唯一正例 | 21,000 | 保持正类学习信号 |
| 新挖掘难负例 | 17,850 | 学习上一轮最易误判的负类，降低假阳性 |
| 普通负例 | 3,150 | 保留更广的负类覆盖面 |
| 合计 | **42,000** | 正负类 1:1，负例中难例占 85% |

论文中普通负例从剩余合格负例池抽样，不按上一轮的预测结果筛选。每轮从上一轮胜出权重热启动训练，使用类别加权 focal loss，`gamma=2`。固定优化设置为 AdamW、cosine schedule、batch size 32、weight decay 0.01、warmup ratio 0.10，以及模型/数据 seed=42；候选只改变学习率、epoch 数和正类权重 alpha。

每轮先筛选 **Recall ≥ 80%** 的候选，再选其中 Precision 最高者作为下一轮起点。若没有候选达到 Recall 门槛，停止选择新模型。

### 8. 在固定低正类比例下检验迭代效果

同一份 50,000 条选优集包含 750 正例和 49,250 负例。四轮结果为：

| 版本 | Precision | Recall | F1 | FPR | TP / TN / FP / FN |
| --- | ---: | ---: | ---: | ---: | --- |
| V1 | 9.22% | 92.27% | 16.76% | 13.84% | 692 / 42,436 / 6,814 / 58 |
| V2 | 21.12% | 86.40% | 33.94% | 4.91% | 648 / 46,830 / 2,420 / 102 |
| V3 | 35.03% | 82.40% | 49.16% | 2.33% | 618 / 48,104 / 1,146 / 132 |
| V4 | **50.12%** | **81.07%** | **61.95%** | **1.23%** | 608 / 48,645 / 605 / 142 |

假阳性从 6,814 降至 605，减少 **91.12%**；FPR 从 13.84% 降至 1.23%，四个版本的 Recall 均达到 80% 门槛。完成四轮并达到项目性能目标后，研究固定 V4。

V4 从 V3 继续微调，采用 learning rate 5e-6、3 epochs、batch size 32、max length 1280、focal gamma 2.0 和 positive alpha 0.6。

### 9. 用独立人工标签验证冻结模型及筛选价值

人工测试样本与训练数据在同一采集时期覆盖多个来源和话题，并经过独立去重与水军过滤。三名标注者采用 AB、BC、CA 轮换配对：每条样本由两人独立标注，第三人裁决分歧。标注员培训后的 **300** 条校准样本上，裁决前的 Krippendorff's alpha 为 **0.87**，AB、BC、CA 三组 Cohen's kappa 分别为 **0.83、0.89、0.85**，分歧率为 **5.2%**。

V4、阈值和选优规则冻结后，使用裁决后的人工标签进行最终评价：

| 测试集 | 样本数 | 正类比例 | Precision | Recall | F1 | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 真实比例人工测试 | 50,000 | 1.5% | **49.6386%** | **82.4000%** | **61.9549%** | **1.2731%** |
| 平衡人工测试 | 4,000 | 50% | 93.4600% | 83.6000% | 88.2555% | 5.8500% |

对应混淆矩阵：真实比例测试 TP=618、TN=48,623、FP=627、FN=132；平衡测试 TP=1,672、TN=1,883、FP=117、FN=328。

真实比例人工测试的 95% bootstrap CI 为 Precision [46.8594%, 52.4412%]、Recall [79.6630%, 85.1152%]、F1 [59.4433%, 64.4404%]、FPR [1.1740%, 1.3732%]。这些固定模型的样本抽样区间由混淆矩阵的经验分布计算，使用 seed=42、10,000 次多项式重采样和 percentile 法。

在 50,000 条人工测试样本中，V4 筛出 **1,245** 条候选，占 **2.49%**，覆盖 750 条真实反讽中的 **618** 条。候选集中约一半是反讽，形成用于优先人工复核的较小样本池。

## 仓库结构

```text
.
├── README.md
├── code/
│   ├── pipeline_config.py         # 路径与环境变量配置
│   ├── run_pipeline.py            # 采集与标注调度
│   ├── data_label/                # 采集、清洗、标注与投票
│   └── model_train/
│       ├── train_cls.py           # 仅正文 CLS 基线
│       ├── train_cls_prompt.py    # Prompt+MLM 训练
│       ├── train_cls_prompt_iterative.py # 单轮训练入口
│       ├── run_iterative_pipeline.py    # 多轮训练调度
│       ├── experiment_tracking.py       # 实验记录
│       └── tools/                 # 挖负例、评估、选优与分歧分析
├── data/
│   ├── examples/                  # 人工编写的 JSONL 格式示例
│   ├── data_summary.json          # 原始语料聚合统计
│   └── split_manifest.json        # 历史划分统计与 SHA-256
└── models/
    ├── base/                      # 底座配置与预训练指标
    └── v1/ … v4/                  # 胜出模型配置与指标
```

## 数据格式

`data/examples/` 提供 12 条人工编写的虚构标注示例（6 正 / 6 负），对应 train 8 条、test 4 条，用于展示字段及反讽判定规则。

标注数据使用 `content`（评论正文）、`retweeted_content`（原帖上下文）、`is_sarcasm`（最终 0/1 标签）、三模型各自标签及 `positive_votes`（标签 1 的票数）。标准化训练与评估格式为：

```json
{"text":"真是高效率，等了三个小时连号码都没叫到。","retweeted_content":"服务窗口宣传当天办理无需等待。","label":1,"source_line":1,"is_synthetic":true}
```

`text` 和 `retweeted_content` 为字符串，`label` 为 0/1 整数。示例中的 `is_synthetic: true` 标记虚构文本，`source_line` 为示例编号。`data_summary.json` 和 `split_manifest.json` 记录原始完整语料的统计与文件指纹。

## 运行方式

### 1. 准备环境与模型

历史训练环境为 Python 3.11.9、PyTorch 2.7.1+cu126、Transformers 4.52.4 和单张 NVIDIA L20。创建 Python 环境，安装与本机 CUDA 匹配的 PyTorch，再安装其余依赖：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# 在此环境中安装与 CUDA 匹配的 PyTorch。
python -m pip install "transformers==4.52.4" "accelerate>=1,<2" \
  numpy scikit-learn requests tensorboard safetensors
export BASE_MODEL_PATH="/absolute/path/to/ModernBertHansir-zh-8k-base/base"
```

`BASE_MODEL_PATH` 指向包含权重、完整 tokenizer 和 `config.json` 的底座目录。仓库中的 `models/` 保存配置与指标，运行时另外准备模型权重和 tokenizer。

### 2. 准备数据

将完整标注语料、训练集、测试集和固定选优集放置到以下路径：

```text
code/data/labeled/sarcasm_labeled.jsonl
code/data/splits/train.jsonl
code/data/splits/test.jsonl
code/outputs/real_distribution_eval.jsonl
```

先去重并划分训练与评估数据；难负例候选池排除已进入 train/test、固定选优集及先前轮次的样本。最终人工测试集在模型确定后单独使用。

### 3. 单轮训练

下例使用 V3 权重及 V4 的训练参数：

```bash
python code/model_train/train_cls_prompt_iterative.py \
  --model-path /absolute/path/to/v3/best_model \
  --train-file code/data/splits/train.jsonl \
  --test-file code/data/splits/test.jsonl \
  --output-dir code/outputs/reference_run \
  --max-length 1280 --batch-size 32 \
  --learning-rate 5e-6 --focal-alpha 0.6 --num-epochs 3
```

### 4. 多轮调度与评估

调度脚本的参数网格位于 `PARAM_GRID_BY_ITERATION`。按论文设置运行时，对齐以下具体口径：

| 项目 | 论文协议 | 当前代码 |
| --- | --- | --- |
| 标注正类聚合 | 至少两票 1 | `relabel_votes.py` 使用“0 票数 ≤ 1 且至少一票 1” |
| 普通负例抽样 | 从剩余合格池抽样，不按上一轮预测筛选 | `mine_hard_negatives.py` 从上一轮预测为 0 的候选中抽取 |
| 输入长度 | 1280 token | `train_cls_prompt.py` 的 `MAX_LENGTH` 默认 384 |

两种投票规则在 `[1,2,2]`、`[1,0,2]` 等组合上产生不同标签。按论文协议复跑时，将聚合条件设为 `labels.count(1) >= 2`，对齐普通负例抽样规则，并将 `MAX_LENGTH` 设为 1280；每轮候选参数按目标实验配置设置。

```bash
python code/model_train/run_iterative_pipeline.py \
  --max-iterations 1 --hard-negative-ratio 0.85 \
  --recall-threshold 0.80 --dry-run
```

`--dry-run` 打印各组训练命令，移除后执行训练。实验编号写入输出目录的 `latest.txt`，评估时读取相应实验保存的输入配置和冻结阈值：

```bash
python code/model_train/tools/eval_on_real_distribution_prompt.py \
  --experiment-dir "code/outputs/reference_run/experiments/$(cat code/outputs/reference_run/latest.txt)" \
  --eval-file code/outputs/real_distribution_eval.jsonl
```

### 5. 采集与标注配置

采集需安装 RocketMQ 绑定并配置服务；三模型标注通过环境变量读取凭据。配置项如下：

| 环境变量 | 用途 |
| --- | --- |
| `DASHSCOPE_API_KEY` | Qwen 标注接口凭据 |
| `TENCENT_TOKENHUB_API_KEY` | DeepSeek 标注接口凭据 |
| `ROCKETMQ_NAME_SERVER` | MQ 服务地址 |
| `ROCKETMQ_TOPIC` | 采集 topic |
| `ROCKETMQ_CONSUMER_GROUP` | 消费者组 |
| `SARCASM_EXPERIMENT_DIR` | 分歧分析使用的实验目录 |
| `CUDA_VISIBLE_DEVICES` | 指定 GPU |

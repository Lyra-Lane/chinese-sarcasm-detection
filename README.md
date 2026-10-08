# 中文反讽识别：Prompt+MLM 与迭代难负例训练

Chinese Sarcasm Detection with Context-Aware Prompt+MLM and Iterative Hard Negative Mining

本项目识别中文社交媒体评论中的**严格反话型反讽**，输入为评论正文及被评论原帖的上下文。原始语料中反讽占比约 1.54%；在这一类别分布下，负类误判会显著影响 Precision。因此，项目以降低假阳性为训练重点，通过迭代难负例挖掘提高 Precision，同时保持至少 80% 的 Recall。

## 核心方法

底座 `ModernBertHansir-zh-8k-base` 在中文语料上继续进行掩码语言建模预训练，支持 8192 token 上下文和 151,666 规模词表。下游分类复用 MLM 预测头，将原帖与评论组织为带 `[MASK]` 的问句：

```text
结合 {retweeted_content} ，判断{text}是否是反讽表达，[MASK]
```

在 `[MASK]` 位置取“否”和“是”的 logits，计算二类概率，分别对应非反讽和反讽。分类阈值在训练前固定为 0.5。历史胜出模型使用 `max_length=1280`；输入过长时优先截断原帖上下文，保留评论正文和 `[MASK]`，极端情况下再截断正文。

训练流程如下：

1. **弱监督标注**：清洗、去重并处理 OCR 重复上下文，再由三个模型分别输出 0（非反讽）、1（反讽）或 2（无法判定）。0 的票数不超过 1 且至少有一票 1 时，最终记为 1；否则记为 0。全为 2 的样本丢弃。
2. **构建平衡训练集**：初始 train/test 均为 1:1。V1 将每条正例复制一次，并补充等量全新负例，扩大训练规模和负例覆盖面。
3. **挖掘难负例**：上一轮胜出模型扫描未使用的负例池，收集“弱监督标签为 0、预测为 1”的样本，替换下一轮训练负例。第 2 轮起固定唯一正例，每轮负例由 85% 难负例和 15% 普通负例组成，并从上一轮胜出权重继续微调。
4. **选优与最终测试**：在固定真实比例选优集上筛选 Recall ≥ 0.80 的候选，再选择 Precision 最高者。V4 模型与阈值确定后，用两套纯人工标注测试集评价最终表现。

## 数据与实验成果

弱监督语料共 **1,707,992** 条，其中正类 **26,250** 条、负类 **1,681,742** 条，正类比例约 **1.54%**。初始 train 为 42,000 条（21,000 正 / 21,000 负），test 为 10,500 条（5,250 正 / 5,250 负）。V1 训练集扩增到 84,000 行；第 2 轮起，每轮使用 21,000 条唯一正例和 21,000 条全新负例。

### 固定真实比例选优集

各轮在同一份 50,000 条弱监督数据上比较，包含 750 条正类和 49,250 条负类，正类占比 1.5%。四轮迭代后，Precision 从 9.22% 提高到 50.12%，Recall 为 81.07%。

| 版本 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| V1 | 9.22% | 92.27% | 16.76% |
| V2 | 21.12% | 86.40% | 33.94% |
| V3 | 35.03% | 82.40% | 49.16% |
| V4 | **50.12%** | **81.07%** | **61.95%** |

V4 从 V3 继续微调，使用 learning rate 5e-6、3 epochs、batch size 32、max length 1280、focal gamma 2.0 和 positive alpha 0.6。第 5 轮扫描 1,552,348 条候选负例后得到 4,184 条难负例，少于所需 17,850 条，触发样本不足停止条件，最终模型为 V4。

### 冻结 V4 的人工测试

两套纯人工标注数据在 V4 确定后用于最终测试，分别评价真实低正类比例下的表现和类别平衡时的区分能力。

| 测试集 | 样本数 | 正类占比 | Precision | Recall | F1 | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 真实比例人工测试 | 50,000 | 1.5% | **49.6386%** | **82.4000%** | **61.9549%** | **1.2731%** |
| 平衡人工测试 | 4,000 | 50% | 93.4600% | 83.6000% | 88.2555% | 5.8500% |

对应混淆矩阵：真实比例测试 TP=618、TN=48,623、FP=627、FN=132；平衡测试 TP=1,672、TN=1,883、FP=117、FN=328。

真实比例人工测试的 95% bootstrap CI 为 Precision [46.8594%, 52.4412%]、Recall [79.6630%, 85.1152%]、F1 [59.4433%, 64.4404%]。这些固定模型的样本抽样区间由上述混淆矩阵的经验分布计算，使用 seed=42、10,000 次多项式重采样和 percentile 法。

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

调度脚本的参数网格位于 `PARAM_GRID_BY_ITERATION`，输入长度读取 `train_cls_prompt.py` 的 `MAX_LENGTH`（默认 384）。采用历史 1280 长度时，将该值改为 1280，并按目标实验的训练配置设置各轮参数网格。

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

# 中文反讽识别：Prompt+MLM 与迭代难负例训练

Chinese Sarcasm Detection with Context-Aware Prompt+MLM and Iterative Hard Negative Mining

本仓库记录中文社交媒体**严格反话型反讽**识别研究的核心代码、数据格式示例、模型配置和实验成果。上传内容仅包含 `README.md`、`code/`、`data/` 的示例与聚合统计、`models/` 的配置与指标。输入为评论正文及原帖上下文，目标是在反讽占比约 1.5% 的场景中减少误报，同时保持至少 80% 的召回率。

这是从本地研究材料整理的轻量归档。数据示例全部为人工编写的虚构文本；完整语料、模型权重及逐样本预测不随仓库提供。代码用于理解方法和开展参考实验，当前快照与历史实跑参数存在差异，不承诺直接复现全部历史结果。

## 核心方法

1. **上下文感知的 Prompt+MLM 分类**：使用中文继续预训练的 `ModernBertHansir-zh-8k-base`，将原帖和评论构造成完形填空问句，在 `[MASK]` 位置比较“否/是”的 logits，分别对应非反讽/反讽。
2. **结构感知截断**：优先保留 `[MASK]` 和评论正文，输入过长时首先截断原帖上下文；极端情况下再截断正文。历史胜出模型的 `max_length` 为 1280，底座最大上下文长度为 8192。
3. **迭代难负例挖掘**：用上一轮模型寻找“弱监督标签为 0、预测为 1”的样本，替换下一轮训练负例，并从上一轮胜出权重继续微调。
4. **低正类比例下选优**：在固定弱监督真实比例选优集上筛选 Recall ≥ 0.80 的模型，再选择 Precision 最高者。阈值 0.5 在训练前固定。
5. **冻结后人工测试**：V4 确定后，再使用两套纯人工标注测试集报告最终结果和 bootstrap 置信区间。

```text
原帖上下文 + 评论正文
  → 结合 {retweeted_content} ，判断{text}是否是反讽表达，[MASK]
  → MLM 的“否/是”二类概率
  → 阈值 0.5
  → 非反讽 / 反讽

上一轮模型 → 未使用负例池 → 难负例 + 普通负例 → 下一轮训练
```

三模型弱监督标注使用标签 0（非反讽）、1（反讽）、2（无法判定）。聚合规则为“0 的票数 ≤ 1 且至少有一票 1，则记为 1；其余记为 0”；全为 2 的样本在聚合前丢弃。该规则带弃权机制，并非简单的三票多数表决。

## 数据与实验成果

原始弱监督语料共 **1,707,992** 条，反讽正类 **26,250** 条，占约 **1.54%**。初始平衡划分为 train 42,000 条、test 10,500 条。历史 V1 训练曾扩增到 84,000 行（复制正例并补充等量全新负例）；该独立扩增脚本未包含在原始归档中。第 2 轮起固定唯一正例，重建每轮 1:1 训练集，历史难负例/普通负例比例为 85%/15%。

### 迭代阶段：弱监督真实比例选优集

固定 50,000 条，750 正类、49,250 负类。此集合用于版本选择，不能作为独立最终测试。

| 版本 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| V1 | 9.22% | 92.27% | 16.76% |
| V2 | 21.12% | 86.40% | 33.94% |
| V3 | 35.03% | 82.40% | 49.16% |
| V4 | **50.12%** | **81.07%** | **61.95%** |

### 冻结 V4：纯人工最终测试

| 测试集 | 样本数 | 正类占比 | Precision | Recall | F1 | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 真实比例人工测试（主结果） | 50,000 | 1.5% | **49.6386%** | **82.4000%** | **61.9549%** | **1.2731%** |
| 平衡人工测试（辅助结果） | 4,000 | 50% | 93.4600% | 83.6000% | 88.2555% | 5.8500% |

主结果的 95% bootstrap CI：Precision [46.8594%, 52.4412%]，Recall [79.6630%, 85.1152%]，F1 [59.4433%, 64.4404%]。两套人工测试均未参与训练、版本选择或阈值调整；此结论依据原始实验记录，逐样本预测未被归档，无法在本仓库重新审计样本重叠或重跑模型推理。

人工测试的混淆矩阵为：真实比例测试 TP=618、TN=48,623、FP=627、FN=132；平衡测试 TP=1,672、TN=1,883、FP=117、FN=328。置信区间以这两组计数为输入，使用 seed=42、10,000 次多项式 bootstrap 重采样计算 percentile 95% CI；它只反映固定模型的样本抽样不确定性，不包含重训或标注误差。

历史 V4 胜出配置：learning rate 5e-6、3 epochs、batch size 32、max length 1280、focal gamma 2.0、positive alpha 0.6，并从 V3 继续微调。第 5 轮据原日志扫描 1,552,348 条候选负例后，只找到 4,184 条难负例，少于需要的 17,850 条，因此停止，没有产生 V5。

## 仓库结构

```text
.
├── README.md                      # 项目、数据、成果与运行说明
├── code/
│   ├── pipeline_config.py         # 核心代码必需的路径与环境变量配置
│   ├── run_pipeline.py            # 历史采集/标注入口
│   ├── data_label/                # 采集、清洗、标注与投票
│   └── model_train/
│       ├── train_cls.py           # 仅正文 CLS 基线
│       ├── train_cls_prompt.py    # 核心 Prompt+MLM 训练
│       ├── train_cls_prompt_iterative.py
│       ├── run_iterative_pipeline.py
│       ├── experiment_tracking.py
│       └── tools/                 # 挖负例、评估、选优与分歧分析
├── data/
│   ├── examples/                  # 虚构 labeled/train/test JSONL
│   ├── data_summary.json          # 原始语料聚合统计
│   └── split_manifest.json        # 历史划分统计与原始文件指纹
└── models/
    ├── base/                      # 底座配置与预训练聚合指标
    └── v1/ … v4/                 # 胜出模型配置与指标，无权重
```

## 数据格式与示例

示例文本全部为人工编写的虚构内容，不是原始社交媒体数据的抽样。标注文件共 12 条（6 正 / 6 负），对应训练文件 8 条（4 正 / 4 负）、测试文件 4 条（2 正 / 2 负）。训练和测试的示例编号互不重叠；三模型票数为人为设置，未调用外部服务。

标注数据使用 `content`（评论正文）、`retweeted_content`（原帖上下文）、`is_sarcasm`（最终 0/1 标签）、三模型各自标签和 `positive_votes`（标签 1 的票数）。标准化训练/测试数据使用 `text`、`retweeted_content`、`label`、`source_line`：

```json
{"text":"真是高效率，等了三个小时连号码都没叫到。","retweeted_content":"服务窗口宣传当天办理无需等待。","label":1,"source_line":1,"is_synthetic":true}
```

`is_synthetic: true` 表示虚构示例，`source_line` 是示例编号，不对应真实语料行号。示例不含真实账号、平台 URL、记录 ID 或时间戳，仅展示格式和判定规则，不能用于验证论文性能。`data_summary.json` 和 `split_manifest.json` 对应原始完整语料；其中 SHA-256 是历史原始文件的指纹，不对应虚构示例，原机器路径已用 `<historical-project-root>` 替代。

## 参考训练与评估

### 1. 准备独立环境与底座

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "transformers==4.52.4" "accelerate>=1,<2" \
  numpy scikit-learn requests tensorboard safetensors
export BASE_MODEL_PATH="/absolute/path/to/ModernBertHansir-zh-8k-base/base"
```

另行安装与驱动/CUDA 匹配的 PyTorch（核心代码需要 2.5 或以上版本）。GPU 训练需可用的 NVIDIA GPU。历史训练记录为 Python 3.11.9、PyTorch 2.7.1+cu126、Transformers 4.52.4 和单张 NVIDIA L20。上述依赖安装命令用于参考，未经过全新 GPU 环境安装验证。需要实际的模型权重、完整 tokenizer 和 `config.json`；仓库中的配置文件不足以加载模型。

### 2. 放置有权使用的训练数据

默认布局为 `code/data/labeled/sarcasm_labeled.jsonl`、`code/data/splits/train.jsonl`、`code/data/splits/test.jsonl`，固定选优集放在 `code/outputs/real_distribution_eval.jsonl`。这些运行时数据和输出目录不属于上传内容。

以下仅复制虚构示例以检查数据格式，不能用来验证论文性能：

```bash
mkdir -p code/data/labeled code/data/splits
cp data/examples/labeled.jsonl code/data/labeled/sarcasm_labeled.jsonl
cp data/examples/train.jsonl code/data/splits/train.jsonl
cp data/examples/test.jsonl code/data/splits/test.jsonl
```

正式实验必须换成自己的数据，完成去重和训练/选优/最终测试隔离，并构建足够规模的负例池。数据字段见上文“数据格式与示例”。

### 3. 单轮训练

```bash
python code/model_train/train_cls_prompt_iterative.py \
  --model-path "$BASE_MODEL_PATH" \
  --train-file code/data/splits/train.jsonl \
  --test-file code/data/splits/test.jsonl \
  --output-dir code/outputs/reference_run \
  --max-length 1280 --batch-size 32 \
  --learning-rate 5e-6 --focal-alpha 0.6 --num-epochs 3
```

这是参照 V4 胜出超参数的调用示例。历史 V4 实际从 V3 继续微调，在基础模型上执行此命令不能等价复现 V4。

### 4. 迭代调度与冻结阈值评估

先检查调度命令：

```bash
python code/model_train/run_iterative_pipeline.py \
  --max-iterations 1 --hard-negative-ratio 0.85 \
  --recall-threshold 0.80 --dry-run
```

准备完整数据、底座和固定选优集后，移除 `--dry-run` 执行。当前调度源码使用的最大输入长度默认来自训练脚本（384），与历史胜出模型的 1280 不同；若要对齐历史实验，还需调整训练默认长度和各轮参数网格；当前难负例默认比例为 1.0，历史混合比例应显式设置为 0.85。

```bash
python code/model_train/tools/eval_on_real_distribution_prompt.py \
  --experiment-dir code/outputs/reference_run/experiments/<experiment_id> \
  --eval-file code/outputs/real_distribution_eval.jsonl
```

`experiment_id` 由训练输出产生，可查看 `code/outputs/reference_run/latest.txt`。评估复用该实验保存的输入配置和冻结阈值。

## 标注、权重与归档边界

采集需另行准备 RocketMQ 绑定与服务权限；标注需设置 `DASHSCOPE_API_KEY` 和 `TENCENT_TOKENHUB_API_KEY`。脚本读取导出的环境变量。采集配置由 `ROCKETMQ_NAME_SERVER`、`ROCKETMQ_TOPIC`、`ROCKETMQ_CONSUMER_GROUP` 提供；分歧分析可用 `SARCASM_EXPERIMENT_DIR` 指定实验目录，GPU 可通过 `CUDA_VISIBLE_DEVICES` 指定。历史接口名称和模型标识被保留，当前服务是否可用未验证，执行标注会产生服务费用。

完整语料、原始平台链接、用户标识、模型权重、私有 MQ 地址和密钥未上传。原始划分 SHA-256 对应本地完整文件，**不对应虚构示例**。部分历史脚本未进入源归档，本发布版补齐了文档明确规定的投票函数。上传副本的必要整理包括：移除真实凭据及私有 MQ 配置、将底座路径改为 `BASE_MODEL_PATH` 环境变量、让迭代调度使用当前 Python、修正过时的 1:70 自检预期为当前 1:1，并保留调用方设置的 GPU。新增的 `relabel_votes.py` 只按现存调用脚本明确写出的规则恢复 `final_label`，不是找回的历史完整重标程序。

当前调度参数网格与历史实跑不完全相同；V1 扩增脚本、旧上下文基线和部分专项脚本没有包含在源归档中。模型目录只提供配置与聚合结果，加载实际模型还需权重及完整 tokenizer。整理时已通过数据格式、投票与若干 CPU 逻辑检查，没有重跑 GPU 训练或模型推理。

仓库未新增开源许可证；公开发布、再分发数据或模型、添加许可证之前，需要由权利人确认授权。研究指标是本地归档结果，此次整理没有重新训练或执行完整模型推理。

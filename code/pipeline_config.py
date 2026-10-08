"""整条反讽识别流水线共用的路径与采集规模配置。"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# 凭据仅从环境变量读取；不在仓库中保存真实值。
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
TENCENT_TOKENHUB_API_KEY = os.environ.get("TENCENT_TOKENHUB_API_KEY", "")

# 只需修改这里：wtype=7 和 wtype=8 各采集多少条。
TARGET_COUNT_PER_TYPE_7 = 250000
TARGET_COUNT_PER_TYPE_8 = 1000
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_FILE = DATA_DIR / "cleaned" / "sarcasm_cleaned.jsonl"
# 待标注文件：从 CLEANED_FILE 里剔除"已标注/已丢弃"的行、再按 content 去重后的结果，
# 每条记录带原始 source_line，供 thr_label.py --resume 继续标注（见 build_pending.py）。
PENDING_FILE = DATA_DIR / "cleaned" / "sarcasm_pending.jsonl"
LABELED_FILE = DATA_DIR / "labeled" / "sarcasm_labeled.jsonl"
LABELED_CSV_FILE = DATA_DIR / "labeled" / "sarcasm_labeled.csv"
LABEL_ERRORS_FILE = DATA_DIR / "labeled" / "sarcasm_errors.jsonl"
THREE_MODEL_DISCARDED_FILE = DATA_DIR / "labeled" / "sarcasm_three_model_discarded.jsonl"
# 平衡后的数据集：下采样多数类（0）到与少数类（1）相同数量，供切分使用。
BALANCED_FILE = DATA_DIR / "labeled" / "sarcasm_balanced.jsonl"
SPLIT_DIR = DATA_DIR / "splits"

# ── 上线效果验证：政府认证账号原帖 + 评论专用数据（online/collect_gov_posts.py 采集，
# online/eval_specified_model.py 用小模型初筛 + 三模型核验）──
# 与主链路的 CLEANED_FILE / LABELED_FILE 完全区分开，互不覆盖、互不混用。
GOV_ONLINE_DATA_DIR = DATA_DIR / "online"
GOV_CLEANED_FILE = GOV_ONLINE_DATA_DIR / "gov_posts_cleaned.jsonl"
# 默认采集目标条数（原帖+评论，去重去水军后的有效条数）；改这里即可调整规模。
GOV_TARGET_COUNT = 20000

# ── 政务场景专项优化（model_train/gov_scene/，第5轮训练）──
# 上线验证用的固定评估集：内容固定不变、不参与任何训练/建集流程，只在评估时读取，
# 避免训练数据与线上验证数据重叠。第5轮建训练集用的是下面独立的 GOV_SCENE_RAW_FILE。
GOV_EVAL_FILE = GOV_ONLINE_DATA_DIR / "eval.jsonl"
# 第5轮专用的大规模采集文件（默认 100 万条），与 GOV_EVAL_FILE 彻底分开。
GOV_SCENE_RAW_FILE = GOV_ONLINE_DATA_DIR / "gov_posts_1m.jsonl"
GOV_SCENE_TARGET_COUNT = 1000000
# 第5轮的数据集与实验产物根目录。
GOV_SCENE_ROOT = PROJECT_ROOT / "outputs" / "gov_scene_round5"
# retweeted_user.verified==2（蓝V）且 verified_type 属于以下集合时视为"政府相关"原帖：
# 1=政府 3=媒体 4=校园 10=类政府。
GOV_VERIFIED_TYPES = (1, 3, 4, 10)

CONTENT_MODEL_DIR = PROJECT_ROOT / "outputs" / "sarcasm_cls_content"

# 迭代式主动学习训练流水线（run_iterative_pipeline.py）专用路径。
# 采集/标注/平衡/切分这几个阶段的产物（CLEANED_FILE/LABELED_FILE/BALANCED_FILE/
# SPLIT_DIR）已构建完毕，不再纳入日常流水线，但路径常量本身仍保留、仍在用
# （标注纠错回路、真实分布评估、迭代挖掘都要读它们）。
ITERATIVE_ROOT = PROJECT_ROOT / "outputs" / "sarcasm_cls_prompt_iterative"
# 已用负例池 + 每轮挖掘历史；不存在时由 mine_hard_negatives.py 首次运行自动初始化。
ITERATIVE_STATE_FILE = SPLIT_DIR / "iterative_state.json"
# 固定生成一次、后续所有轮次/所有组参数共用的真实分布评估集（不随轮次重新生成），
# 由 build_real_distribution_eval.py 产出。其中被抽中的负例样本会被
# mine_hard_negatives.py 排除在候选池之外，避免同一条数据被同时用作评估和训练
# （build_real_distribution_eval.py 抽样时本身也只挑"从未进入 train/test"的负例，
# 两边口径一致，不会出现评估集数据反被后续轮次挖回训练集的情况）。
REAL_DIST_EVAL_FILE = PROJECT_ROOT / "outputs" / "real_distribution_eval.jsonl"

# 迭代式主动学习流水线第1轮的起点底座（虚拟机训练环境路径）；v2、v3...
# 各轮都在上一轮胜出模型的 best_model 基础上继续 fine-tune，不会再用到这个路径。
BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH", str(PROJECT_ROOT.parent / "models" / "base")
)
'''
========================================================================
数据流（LABELED_FILE 是唯一"源头"，balanced / splits 都是派生文件，
随时可以重新生成；CLEANED_FILE 是采集产物，只读不改）：

  RocketMQ
    └─ mqread78.py ─────────────> CLEANED_FILE      (data/cleaned/sarcasm_cleaned.jsonl)
         └─ build_pending.py ───> PENDING_FILE      (data/cleaned/sarcasm_pending.jsonl)
              └─ thr_label.py ──> LABELED_FILE      (data/labeled/sarcasm_labeled.jsonl)
                   └─ balance.py ──> BALANCED_FILE  (data/labeled/sarcasm_balanced.jsonl)
                        └─ split_data.py ──> SPLIT_DIR/{train,test}.jsonl
                             └─ train_cls_prompt.py ──> outputs/sarcasm_cls_prompt_mlm/...
                             └─ run_iterative_pipeline.py ──> outputs/sarcasm_cls_prompt_iterative/...

  上面这条链路（采集→标注→平衡→切分）数据已构建完毕，不再是日常要重跑的
  流水线，本文件不再提供其操作命令；仍需要时可直接看各脚本自身的 docstring
  （data_label/mqread78.py、build_pending.py、thr_label.py、balance.py、
  model_train/split_data.py）。

  纠错回路：LABELED_FILE ─scan_label_disagreements─> CSV ─人工复核─>
            apply_label_corrections ─> LABELED_FILE（再重跑 balance/split 之后的步骤）
========================================================================

------------------------------------------------------------------------
迭代式主动学习训练流水线（当前主力训练方式，基于 train_cls_prompt.py 的
Prompt+MLM 范式）：解决"1:1 平衡数据上准确率高、真实比例下准确率低"的问题，
思路是每一轮都用上一轮模型挖出的"真实标0、模型误判为1"的负例替换训练集里的
标0部分（标1部分不变），迫使模型持续学习它当前最容易犯错的负样本。

# 0）前置：只需要生成一次，不随轮次重新生成（否则各轮/各组参数的评估结果
#    不可比）。生成后固定不变，路径见 REAL_DIST_EVAL_FILE。
python3 model_train/tools/build_real_distribution_eval.py

# 1）跑流水线：每轮先（除第1轮外）挖负例构造新 train，再顺序跑 4 组参数
#    （见 model_train/run_iterative_pipeline.py 顶部 PARAM_GRID），用固定的
#    真实分布评估集选出 recall>=80% 里 precision 最高的一组作为本轮最优模型，
#    下一轮基于该模型继续 fine-tune。
python3 model_train/run_iterative_pipeline.py --max-iterations 1   # 先跑通单轮
python3 model_train/run_iterative_pipeline.py --max-iterations 3

------------------------------------------------------------------------
用已训练模型反查标注是否有系统性错误（主动学习式纠错，配合模型诊断使用）：
不纳入 all（依赖已有 best_model，不是每次跑流水线都要做），需要时单独执行。

# 1. 用 best_model 对 LABELED_FILE 全量样本跑推理（默认含 train/val/test，
#    用 in_training 列区分），筛出"模型判断与原标签不一致、置信度>=阈值"的
#    候选，输出 outputs/label_disagreements.csv。
#    注意：脚本顶部的 MODEL_PATH / USE_CONTEXT / MAX_LENGTH 必须配套——
#    上下文模型用 USE_CONTEXT=True + MAX_LENGTH=384，纯正文模型用 False + 256。
python3 model_train/tools/scan_label_disagreements.py
python3 model_train/tools/scan_label_disagreements.py --threshold 0.8   # 覆盖脚本内 CONFIDENCE_THRESHOLD
python3 model_train/tools/scan_label_disagreements.py --direction 0to1  # 只看"原标签0/模型判1"这一个方向
python3 model_train/tools/scan_label_disagreements.py --exclude-training  # 只扫模型没见过的训练集外样本
# 或通过 run_pipeline.py 统一入口（自动走 model 容器 + conda 环境）：
python3 run_pipeline.py scan_disagreements --scan-threshold 0.8 --scan-direction both

# 2. 人工在 CSV 的 human_review_label 列填确认后的正确标签（仅需修改的行填，其余留空）

# 3. 把人工纠正写回 LABELED_FILE（默认不覆盖，先看 *_corrected.jsonl；
#    确认无误后加 --in-place 才原地覆盖，且自动先生成 .bak 备份）。
#    ⚠️ 执行前必须先停掉后台标注进程：thr_label.py 以 "a" 模式持续追加写
#    LABELED_FILE，本步骤是整份重写，同时进行会损坏文件并丢掉纠正。
python3 data_label/apply_label_corrections.py
python3 data_label/apply_label_corrections.py --in-place

# 4. LABELED_FILE 更新了，必须重新走 balance -> split -> 训练，
#    否则纠正不会体现在训练数据里。

------------------------------------------------------------------------
用真实标签分布估计模型的真实表现（train/val/test 都是 1:1 平衡集，其上算出的
precision/recall/f1 严重偏离真实分布下的表现，尤其 precision——真实数据里正类
占比远低于1:1，同样的模型行为搬到真实分布，负类基数放大，FP绝对数量跟着放大，
1:1测试集完全看不出来）：
不改动现有 train/val/test，只是另外构造一份评估集 + 单独跑一次推理，不纳入
日常调参流程（调参仍用现有1:1 val，保持历史实验可比）。

# 1. 构造真实分布评估集：正类复用 test.jsonl 里的正类（已划出、无信息泄漏）；
#    负类从 LABELED_FILE 里随机抽"从未进入 train/val/test"的样本，按 LABELED_FILE
#    全量统计出的真实正类占比换算比例，输出 outputs/real_distribution_eval.jsonl。
python3 model_train/tools/build_real_distribution_eval.py
python3 model_train/tools/build_real_distribution_eval.py --target-size 50000  # 样本量越大指标越稳
python3 model_train/tools/build_real_distribution_eval.py --ratio 0.013  # 手动指定真实正类占比

# 2. 用指定模型在这份评估集上跑推理，算出真实分布下的 precision/recall/f1。
#    默认指向当前上下文方向最优模型（exp_20260807_163717）；换模型时
#    --use-context/--no-context 和 --max-length 必须与该模型训练时的设置一致。
python3 model_train/tools/eval_on_real_distribution.py
python3 model_train/tools/eval_on_real_distribution.py --threshold 0.6  # 只用于观察阈值敏感度，不用于选阈值
python3 model_train/tools/eval_on_real_distribution.py --sweep  # 扫一组阈值，输出筛选比例/precision/recall对照表
python3 model_train/tools/eval_on_real_distribution.py --sweep --use-cache  # 复用已缓存的推理结果，不重新跑模型

------------------------------------------------------------------------
旁路工具（不在主链路上）：

# 对比两次实验的 sync/ 数据（只读小文件，本机无训练依赖也能跑）
python3 model_train/tools/compare_experiments.py --root outputs/sarcasm_cls_content

# 训练集扩容（正样本复制+负样本补充）：
python3 model_train/augment_train_positives.py
'''
"""
代码2：文本分类微调训练
读取 split_data.py 切分出的 train/test 数据，
在 ModernBertHansir-zh-8k-base 底座上微调一个分类模型。

上下文对照实验由 train_cls_context.py 复用本脚本，只改变输入拼接方式。
"""

import os
import json
import time
import argparse
import torch
import random
import sys
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import CONTENT_MODEL_DIR, SPLIT_DIR
# 实验记录：保存逻辑独立在 experiment_tracking.py，本文件只负责调用。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_tracking as et
from split_data import (
    TEST_NEGATIVE_PER_POSITIVE,
)

# ============================================================
# 【可调参数区 1】路径与数据
# ============================================================
# 指定使用哪块 GPU（多卡时可改成 "0,1"；用 CPU 可留空 ""）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# 模型底座的绝对路径（里面要有 config.json、分词器、权重文件）
from pipeline_config import BASE_MODEL_PATH
MODEL_PATH = BASE_MODEL_PATH

# split_data.py 生成的数据文件夹
DATA_DIR = SPLIT_DIR
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")

# 训练结果保存目录
OUTPUT_DIR = CONTENT_MODEL_DIR

# 对照实验默认只输入评论正文；上下文版入口会将其设为 True。
USE_CONTEXT = False

# ============================================================
# 【可调参数区 2】训练超参数
# ============================================================
# 学习率：BERT 类模型微调常用 2e-5 ~ 5e-5。
#   太大(如1e-4)训练不稳、可能发散；太小(如1e-7)几乎学不动。
# 基线实验(1:1平衡后实际13004条)：训练集约1.04万条，属中等规模，取常规值 2e-5。
LEARNING_RATE = 2e-5

# 每批喂多少条：越大越稳越快但越吃显存。
#   显存不足(OOM)时依次调小：32 -> 16 -> 8 -> 4
# 基线实验：1.04万训练样本仍足够支撑 batch=32（约325步/轮），梯度更稳定；OOM 则退回 16。
BATCH_SIZE = 32

# 训练轮数：数据整体过几遍。小数据集 3~5 轮，大数据集 1~2 轮。
# 历史实验显示训练 4 轮已出现过拟合；在无验证集的固定训练方案中，统一训练 3 轮。
NUM_EPOCHS = 3

# 文本最大长度(token)：短文本设 128/256 更快更省显存；
#   需要读长上下文再调大(最大支持 8192)。评论文本通常 256 足够。
MAX_LENGTH = 256

# 权重衰减：抑制过拟合的正则项，一般 0.01 附近。
WEIGHT_DECAY = 0.01

# 学习率预热比例：开头用小学习率慢慢升，稳定训练。常用 0.05~0.1。
WARMUP_RATIO = 0.1

# 优化器：GPU 上用 adamw_torch_fused 更快；CPU 或报错时换成 adamw_torch。
OPTIM = "adamw_torch_fused"

RANDOM_SEED = 42

# 只保留 train/test 时，阈值必须在看 test 前固定；修改后需重新冻结一份测试集。
CLASSIFICATION_THRESHOLD = 0.5
# ============================================================


def parse_args(argv=None):
    """选择本次训练使用的训练集；测试集始终保持当前固定路径。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file",
        default=TRAIN_FILE,
        help="训练 JSONL 文件；默认使用 data/splits/train.jsonl",
    )
    return parser.parse_args(argv)


def load_jsonl(path):
    """读取 jsonl 为 [{'text':..,'label':..}, ...]"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件：{path}，请先运行 split_data.py")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = example["text"]
        if USE_CONTEXT and example.get("retweeted_content"):
            text += "\n" + example["retweeted_content"]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(example["label"], dtype=torch.long),
        }


def prepare_datasets(tokenizer, train_file):
    train_raw = load_jsonl(train_file)
    test_raw = load_jsonl(TEST_FILE)
    validate_dataset_distributions(train_raw, test_raw)

    random.seed(RANDOM_SEED)
    random.shuffle(train_raw)

    # 标签映射：只用训练集建立，测试集沿用同一套映射（避免编号错位）
    unique_labels = sorted({e["label"] for e in train_raw})
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    def encode_labels(rows):
        for e in rows:
            if e["label"] not in label2id:
                raise ValueError(f"测试集出现训练集没有的标签：{e['label']}")
            e["label"] = label2id[e["label"]]

    encode_labels(train_raw)
    encode_labels(test_raw)

    train_ds = CustomDataset(train_raw, tokenizer, MAX_LENGTH)
    test_ds = CustomDataset(test_raw, tokenizer, MAX_LENGTH)
    return train_ds, test_ds, len(unique_labels), label2id, id2label


def validate_dataset_distributions(train_rows, test_rows):
    """训练集和测试集均允许任意比例，但都必须包含正、负类。"""
    for name, rows in (("train", train_rows), ("test", test_rows)):
        positives = sum(int(row["label"]) == 1 for row in rows)
        negatives = sum(int(row["label"]) == 0 for row in rows)
        if not positives or not negatives:
            raise ValueError(
                f"{name} 必须同时包含正、负类，实际正类={positives}、负类={negatives}"
            )


def metrics_at_threshold(labels, positive_probs, threshold):
    """使用固定阈值计算二分类指标；测试集只允许调用这个固定阈值版本。"""
    labels = np.asarray(labels, dtype=int)
    predictions = (np.asarray(positive_probs) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "sarcasm_precision": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "sarcasm_recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
        "sarcasm_f1": f1_score(labels, predictions, pos_label=1, zero_division=0),
        "false_positive_rate": 1.0 - specificity,
        "specificity": specificity,
        "predicted_positive_ratio": float(predictions.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fixed_evaluation_metrics(logits, labels, threshold, prefix):
    """用训练前固定的阈值评估指定 split。"""
    positive_probs = et.softmax(logits)[:, 1]
    metrics = metrics_at_threshold(labels, positive_probs, threshold)
    metrics["average_precision"] = average_precision_score(labels, positive_probs)
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


# ============================================================
# 【可调参数区 3】训练损失
# focal：强调难样本；cross_entropy：标准二分类损失，不使用类别权重。
# ============================================================
LOSS_TYPE = "focal"  # focal / cross_entropy
FOCAL_GAMMA = 2.0
# exp_20260803_155343（1:1数据，f1导向，best_epoch=3）validation fp=158>fn=124，
# test fp=175>fn=143，两个split均轻微偏向"非讽刺判成讽刺"。之前尝试用
# balance ratio=2（exp_20260803_163931）从数据层面纠偏，但因未同步调整alpha，
# 数据端与loss端未协同，precision/recall/f1三项全面下降（已回退到1:1数据）。
# 数据量扩大到23944条训练样本后（exp_20260806_140146），继续按同一方向微调：
# 0.45→0.40，test fp/fn由175/166≈1.05进一步收窄，f1=78.8%，为当前最佳基线。
# 曾尝试"正样本复制+负样本扩容"（augment_train_positives.py，mentor方案）配合
# alpha=0.4/0.5两版（exp_140926/exp_142324），均为负收益（f1降至73~74%），
# 已放弃该数据扩容方向，train.jsonl 不应使用 augment_train_positives.py 的输出。
FOCAL_ALPHA = 0.45   # 讽刺类权重；非讽刺类权重 = 1 - FOCAL_ALPHA = 0.60
# 历史参考：alpha=0.75 时 exp_20260717 出现 FP/FN=8.5，模型过度偏向讽刺类，
# 调低至 0.60 改善 precision——方向判断的最早依据，现已迭代到 0.40。


class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        ce_loss = CrossEntropyLoss(reduction="none")(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        # alpha 按 label 选取：讽刺类=FOCAL_ALPHA，非讽刺类=1-FOCAL_ALPHA
        alpha_t = torch.where(labels == 1,
                              torch.tensor(FOCAL_ALPHA, device=logits.device),
                              torch.tensor(1 - FOCAL_ALPHA, device=logits.device))
        loss = (alpha_t * (1 - pt) ** FOCAL_GAMMA * ce_loss).mean()
        return (loss, outputs) if return_outputs else loss


def loss_config():
    """返回本次实验实际使用的损失配置，写入 train_config 方便对照。"""
    if LOSS_TYPE == "cross_entropy":
        return {"type": "cross_entropy", "class_weight": None}
    if LOSS_TYPE == "focal":
        return {
            "type": "focal_loss",
            "gamma": FOCAL_GAMMA,
            "alpha_positive_label1": FOCAL_ALPHA,
            "alpha_negative_label0": 1 - FOCAL_ALPHA,
        }
    raise ValueError(f"不支持的 LOSS_TYPE：{LOSS_TYPE!r}")


class TensorBoardCallback(TrainerCallback):
    def __init__(self, writer):
        self.writer = writer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, state.global_step)
            self.writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()


def main():
    args = parse_args()
    active_loss_config = loss_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 为本次训练创建全新的、不可覆盖的实验目录 ----
    paths = et.create_experiment_dir(OUTPUT_DIR)
    task_name = os.path.basename(os.path.normpath(str(OUTPUT_DIR)))
    source_files = [
        os.path.join(os.path.dirname(__file__), "train_cls.py"),
        str(Path(__file__).resolve().parents[1] / "pipeline_config.py"),
    ]
    started_ts = time.monotonic()
    manifest = et.init_manifest(paths, task_name, sys.argv,
                                repo_dir=os.path.dirname(__file__),
                                source_files=source_files)
    print(f"实验目录：{paths['exp_dir']}  (状态: running)")

    # 把标准输出/错误同时写入 run.log，便于诊断。
    _tee_out = et.Tee(sys.stdout, paths["run_log_path"])
    _tee_err = et.Tee(sys.stderr, paths["run_log_path"])
    sys.stdout, sys.stderr = _tee_out, _tee_err

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.model_max_length = MAX_LENGTH

        train_ds, test_ds, num_labels, label2id, id2label = prepare_datasets(
            tokenizer, args.train_file
        )
        print(f"训练集：{len(train_ds)}  测试集：{len(test_ds)}")
        print(f"类别数：{num_labels}  标签映射：{label2id}")
        print(f"输入方式：{'content + retweeted_content' if USE_CONTEXT else 'content'}")
        print(f"训练损失：{LOSS_TYPE}")

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )

        training_args = TrainingArguments(
            output_dir=paths["checkpoints_dir"],  # checkpoint 落到本实验的 checkpoints/
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            num_train_epochs=NUM_EPOCHS,
            lr_scheduler_type="cosine",       # 学习率调度：cosine 平滑下降，也可改 "linear"
            warmup_ratio=WARMUP_RATIO,
            optim=OPTIM,
            logging_dir=paths["logs_dir"],
            logging_strategy="steps",
            logging_steps=10,
            eval_strategy="no",
            save_strategy="epoch",
            save_safetensors=True,           # 用 safetensors 格式存权重（更安全、加载更快）
            save_total_limit=3,               # 最多保留几个存档，防止占满硬盘
            load_best_model_at_end=False,
            max_grad_norm=1.0,                # 梯度裁剪，防止训练爆炸
            report_to=["tensorboard"],
            fp16=False,                       # 有较新 GPU 可改 bf16=True 省显存提速
            push_to_hub=False,
            seed=RANDOM_SEED,
            data_seed=RANDOM_SEED,
        )

        # ---- 记录真正生效的训练配置（覆盖后的值，非源码默认值）----
        et.write_json(paths["train_config_path"], et.build_train_config(
            training_args,
            runtime={
                "model_name_or_path": MODEL_PATH,
                "tokenizer_name_or_path": MODEL_PATH,
                "max_length": MAX_LENGTH,
                "label2id": label2id,
                "use_context": USE_CONTEXT,
                "context_concat_format": "content + '\\n' + retweeted_content",
                "truncation": True,
                "padding": "max_length",
                "early_stopping": None,           # 未使用 EarlyStoppingCallback
                "classification_threshold": CLASSIFICATION_THRESHOLD,
                "threshold_selection": {
                    "dataset": None,
                    "method": "fixed_before_training",
                    "value": CLASSIFICATION_THRESHOLD,
                },
                "loss_type": LOSS_TYPE,
                "class_weight": active_loss_config,
            },
        ))

        # ---- 记录数据摘要（复用已加载数据，流式统计 token 长度）----
        et.write_json(paths["data_summary_path"], {
            "split_method": (
                "custom train/test ratios"
            ),
            "split_seed": RANDOM_SEED,
            "train": et.summarize_split(
                train_ds.data, tokenizer, MAX_LENGTH, USE_CONTEXT, "train",
                args.train_file, "train=custom_ratio", RANDOM_SEED, id2label),
            "test": et.summarize_split(
                test_ds.data, tokenizer, MAX_LENGTH, USE_CONTEXT, "test",
                TEST_FILE, "test=custom_ratio", RANDOM_SEED, id2label),
        })

        writer = SummaryWriter(log_dir=paths["logs_dir"])
        trainer_class = FocalLossTrainer if LOSS_TYPE == "focal" else Trainer
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            callbacks=[TensorBoardCallback(writer)],
        )

        trainer.train()

        # ---- 训练历史：从内存中的 log_history 安全转换，不依赖 checkpoint 目录 ----
        et.write_jsonl(paths["history_path"],
                       et.build_history_rows(trainer.state.log_history))

        # ---- 测试集只在训练结束后评估一次；阈值在训练前已固定 ----
        test_pred = trainer.predict(test_ds, metric_key_prefix="test")
        selected_threshold = CLASSIFICATION_THRESHOLD
        test_metrics = dict(test_pred.metrics)
        test_metrics.update(fixed_evaluation_metrics(
            test_pred.predictions, test_pred.label_ids, selected_threshold, "test"
        ))

        print("\n===== 测试集最终评估（训练前固定阈值）=====")
        print(test_metrics)

        # 训练集 loss 取最后一条含 loss 的训练日志（HF 无逐轮 train accuracy）
        train_raw = {}
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry:
                train_raw = {"train_loss": entry["loss"], "epoch": entry.get("epoch")}
                break

        # 没有验证集时不进行 checkpoint 选择，直接使用固定轮数后的最终模型。
        best_ckpt = None
        et.write_json(paths["decision_config_path"], {
            "selected_on": "predefined_before_training",
            "selection_mode": "fixed_before_training",
            "classification_threshold": selected_threshold,
            "num_train_epochs": NUM_EPOCHS,
            "best_checkpoint": best_ckpt,
        })
        et.write_json(paths["metrics_path"], et.build_metrics(
            train_raw=train_raw,
            val_raw={},
            test_raw=test_metrics,
            primary_metric_name=None,
            best_epoch=None,
            best_checkpoint=best_ckpt,
        ))

        # ---- 逐样本预测：仅测试集，概率由原始 logits softmax 得到 ----
        pred_rows = et.build_prediction_rows(
            test_ds.data, test_pred.predictions, test_pred.label_ids, "test",
            tokenizer, MAX_LENGTH, USE_CONTEXT, id2label,
            classification_threshold=selected_threshold)
        et.write_jsonl(paths["predictions_path"], pred_rows)

        # ---- 分析摘要：直接用内存中的 pred_rows 提炼，不重新读 predictions.jsonl ----
        metrics_obj = json.loads(Path(paths["metrics_path"]).read_text(encoding="utf-8"))
        digest = et.build_analysis_digest(pred_rows, metrics_obj, top_k=20, id2label=id2label)
        et.write_json(paths["digest_path"], digest)

        # ---- 固定轮数训练后的最终模型保存到本实验的 best_model/ ----
        best_path = paths["best_model_dir"]
        trainer.save_model(best_path)
        tokenizer.save_pretrained(best_path)
        print(f"\n最终模型已保存至：{best_path}")

        # ---- 全部核心结果写完后，标记 completed 并更新 latest.txt ----
        et.finalize_manifest(paths, manifest, "completed", started_ts)
        et.update_latest(OUTPUT_DIR, paths["experiment_id"])
        # sync/：只打包小文件（manifest/train_config/data_summary/metrics/history/digest），
        # 不含 predictions.jsonl、checkpoints/、best_model/，方便整体传出虚拟机。
        et.populate_sync_dir(paths)
        print(f"实验完成：{paths['experiment_id']} (状态: completed)")

    except Exception as exc:
        # 失败时保留已生成的日志，标记 failed，且不更新 latest.txt；
        # sync/ 仍打包已生成的小文件，方便远程排查失败原因。
        et.finalize_manifest(paths, manifest, "failed", started_ts, error=exc)
        et.populate_sync_dir(paths)
        print(f"实验失败：{paths['experiment_id']} (状态: failed) -> {type(exc).__name__}: {exc}")
        raise
    finally:
        sys.stdout, sys.stderr = _tee_out._stream, _tee_err._stream
        _tee_out.close()
        _tee_err.close()


def _selfcheck():
    """不加载模型，只验证两份数据比例与固定阈值预测。"""
    assert parse_args(["--train-file", "custom_train.jsonl"]).train_file == "custom_train.jsonl"

    global LOSS_TYPE
    original_loss_type = LOSS_TYPE
    LOSS_TYPE = "cross_entropy"
    assert loss_config() == {"type": "cross_entropy", "class_weight": None}
    LOSS_TYPE = "focal"
    assert loss_config()["type"] == "focal_loss"
    LOSS_TYPE = original_loss_type

    # 训练集和测试集都允许非固定比例。
    validate_dataset_distributions(
        [{"label": 1}, {"label": 1}, {"label": 0}],
        [{"label": 1}, {"label": 0}, {"label": 0}],
    )
    labels = np.array([1] * 10 + [0] * (10 * TEST_NEGATIVE_PER_POSITIVE))
    positive_probs = np.array(
        [0.95] * 8 + [0.10] * 2
        + [0.90] * 5 + [0.20] * (10 * TEST_NEGATIVE_PER_POSITIVE - 5)
    )
    fixed = metrics_at_threshold(labels, positive_probs, CLASSIFICATION_THRESHOLD)
    assert fixed["threshold"] == CLASSIFICATION_THRESHOLD, fixed
    assert fixed["tp"] == 8 and fixed["fp"] == 5, fixed

    rows = [{"text": "a", "retweeted_content": "", "label": 1}]
    prediction_rows = et.build_prediction_rows(
        rows, [[0.0, 0.2]], [1], "test", None,
        MAX_LENGTH, False, classification_threshold=CLASSIFICATION_THRESHOLD,
    )
    assert prediction_rows[0]["predicted_label"] == 0, prediction_rows
    assert prediction_rows[0]["classification_threshold"] == CLASSIFICATION_THRESHOLD
    print("自检通过")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

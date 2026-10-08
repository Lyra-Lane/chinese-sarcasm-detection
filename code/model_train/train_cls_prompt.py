"""
Prompt + 掩码预测（MLM）对照实验。

不同于 train_cls.py / train_cls_context.py 的序列分类范式，本实验把每条样本
改写成话题提示的问答形式：

    "结合 {retweeted_content} ，判断{text}是否是反讽表达，[MASK]"

其中 retweeted_content（原帖内容）直接充当话题（topic），不做单独抽取。
模型在 [MASK] 位置预测"是"/"否"两个候选词，用它们的 logits 做二分类
（"是"=讽刺=label 1，"否"=非讽刺=label 0），而不是走 CLS 分类头。

截断规则（区别于普通 truncation=True 整体截断）：
只有当 max_length 不足以容纳完整 prompt 时，才截断 retweeted_content
（话题部分）；评论正文（text）和 prompt 固定结构（"结合"/"，判断"/
"是否是反讽表达，"/[MASK]）始终保留完整，确保 [MASK] 位置和问句结构不被破坏。

不接入 run_pipeline.py：这是独立的对照实验脚本，需要时直接运行本文件。
数据仍复用 split_data.py 产出的 train.jsonl / test.jsonl（数据准备与训练范式
无关，不需要重新切分）。
"""

import math
import os
import json
import time
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import PROJECT_ROOT, SPLIT_DIR
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_tracking as et
# 复用 train_cls.py 已验证过的数据加载/校验/固定阈值评估逻辑，
# 不重新实现一套评估口径，保证与现有基线可直接对比。
import train_cls


# ============================================================
# 【可调参数区 1】路径与 Prompt 结构
# ============================================================
# setdefault 而不是直接覆盖：共享多卡机器上 GPU0 有时会被其它任务占满，
# 允许调用方提前 export CUDA_VISIBLE_DEVICES=1 之类的方式切换到别的卡；
# 不显式设置时仍默认用卡0，行为与之前一致。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from pipeline_config import BASE_MODEL_PATH
MODEL_PATH = BASE_MODEL_PATH

DATA_DIR = SPLIT_DIR
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")

# 单独存放本实验产出，不与 train_cls.py / train_cls_context.py 的
# CONTENT_MODEL_DIR / CONTEXT_MODEL_DIR 混在一起。
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sarcasm_cls_prompt_mlm"

# Prompt 固定结构："结合 {topic} ，判断{text}是否是反讽表达，[MASK]"
PROMPT_PREFIX = "结合 "
PROMPT_CONNECTOR = " ，判断"    # 连接 topic 与 text
PROMPT_SUFFIX = "是否是反讽表达，"  # 紧跟在 text 之后，其后拼接 [MASK]

# 掩码预测的候选词（verbalizer）："是"=讽刺(label 1)，"否"=非讽刺(label 0)。
# 顺序固定为 [负类词, 正类词]，与 label 0/1 对齐。
VERBALIZER_NEGATIVE = "否"
VERBALIZER_POSITIVE = "是"

# 拼接后长度与 train_cls_context.py 的上下文实验量级相近（topic 同样来自
# retweeted_content），沿用同一档 384，覆盖长尾同时不必跳到 512。
MAX_LENGTH = 384

# ============================================================
# 【可调参数区 2】训练超参数 —— 与 train_cls.py 当前最佳基线保持一致，
# 只让"Prompt+MLM"这一个变量单独生效，结果才能和现有基线直接比较。
# ============================================================
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
OPTIM = "adamw_torch_fused"
RANDOM_SEED = 42
CLASSIFICATION_THRESHOLD = 0.5

LOSS_TYPE = "focal"  # focal / cross_entropy
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.40    # 与 train_cls.py 当前最佳基线（exp_20260806_140146）一致
# ============================================================


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file",
        default=TRAIN_FILE,
        help="训练 JSONL 文件；默认使用 data/splits/train.jsonl",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Prompt 编码：唯一的截断实现，Dataset 和数据统计都调用这一份，不重复实现。
# ---------------------------------------------------------------------------
def _encode_single_char(tokenizer, char, role_label):
    """用 encode() 而非 convert_tokens_to_ids() 解析候选词。

    ModernBERT 等 BPE/字节级分词器的词表里不存原始 UTF-8 字符串，
    convert_tokens_to_ids 对这类候选词永远查不到（返回 None），
    必须走真正的编码入口 encode()，这与 build_prompt_encoding 里
    对 text/topic 的编码方式一致。要求编码结果正好是 1 个 token，
    否则说明这个候选词在当前词表里会被切成多段，不适合做 verbalizer。
    """
    token_ids = tokenizer.encode(char, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"verbalizer {role_label} 候选词 {char!r} 在当前词表里被编码为 "
            f"{len(token_ids)} 个 token（{token_ids}），不是单一 token，"
            f"无法作为 [MASK] 位置的候选词，请换一个词。"
        )
    return token_ids[0]


def resolve_verbalizer_ids(tokenizer):
    """把"是"/"否"映射为词表 id，顺序固定为 [neg_id, pos_id]。"""
    neg_id = _encode_single_char(tokenizer, VERBALIZER_NEGATIVE, "负类(否)")
    pos_id = _encode_single_char(tokenizer, VERBALIZER_POSITIVE, "正类(是)")
    if neg_id == pos_id:
        raise ValueError("verbalizer 的正负类候选词映射到了同一个 id，请换一组候选词。")
    return [neg_id, pos_id]


def build_prompt_encoding(tokenizer, text, topic, max_length,
                          prefix=PROMPT_PREFIX, connector=PROMPT_CONNECTOR,
                          suffix=PROMPT_SUFFIX):
    """构造一条 prompt 的完整 token 序列，只在必要时截断 topic。

    Prompt 结构（五段顺序拼接）：
        固定前缀"结合 " + 可截断的 topic + 固定连接词" ，判断"
        + 必须完整保留的 text + 固定后缀"是否是反讽表达，" + [MASK]

    优先级（由高到低）：[MASK] > text（评论正文，唯一判断对象）
    > 固定结构（prefix/connector/suffix）> topic（可截断的话题）。

    返回 dict：input_ids、attention_mask、mask_position、
    topic_truncated（话题被截断）、text_truncated（连正文都装不下的极端兜底）。
    """
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError(
            f"{tokenizer.__class__.__name__} 没有 mask_token，无法使用 Prompt+MLM 训练方式。"
        )

    prefix_ids_full = tokenizer.encode(prefix, add_special_tokens=False)
    connector_ids = tokenizer.encode(connector, add_special_tokens=False)
    text_ids_full = tokenizer.encode(str(text or ""), add_special_tokens=False)
    tail_ids = tokenizer.encode(suffix, add_special_tokens=False) + [mask_token_id]
    topic_text = str(topic or "")
    topic_ids_full = tokenizer.encode(topic_text, add_special_tokens=False) if topic_text else []

    num_special = tokenizer.num_special_tokens_to_add(pair=False)
    available = max_length - num_special

    # 固定不可截断部分：connector + text + tail（含 [MASK]）。
    # prefix 是唯一允许被牺牲的固定结构（topic 都装不下时才轮到它），
    # 因为它只是引导词，去掉后 prompt 仍可读；text 是判断对象，不可去掉。
    fixed_min_ids = connector_ids + text_ids_full + tail_ids

    text_truncated = False
    if len(fixed_min_ids) > available:
        # ponytail: 最坏情况兜底——连"connector + 完整正文 + 固定后缀"都装不下。
        # [MASK] 的优先级最高，text 次之（评论正文是唯一判断对象，比 topic
        # 更该保留）；此时舍弃 prefix 和 topic，并从 text 尾部截断到刚好能
        # 塞进 connector + text + tail。正常语料下评论远超这个长度，基本
        # 不会触发；若大量触发说明 max_length 设得太小，应调大 max_length。
        budget_for_text = available - len(connector_ids) - len(tail_ids)
        if budget_for_text < 0:
            # 连 connector+tail 都装不下：只保留 tail 末尾（含 [MASK]）。
            tail_ids = tail_ids[-available:] if available > 0 else [mask_token_id]
            if tail_ids[-1] != mask_token_id:
                tail_ids = tail_ids[1:] + [mask_token_id]
            prefix_ids = []
            connector_ids = []
            text_ids = []
            topic_ids = []
        else:
            text_ids = text_ids_full[:budget_for_text]
            prefix_ids = []
            topic_ids = []
        text_truncated = True
    else:
        text_ids = text_ids_full
        budget_for_prefix_and_topic = available - len(fixed_min_ids)
        if len(prefix_ids_full) > budget_for_prefix_and_topic:
            # 次级兜底：prefix 都装不下，直接舍弃 prefix 和 topic。
            prefix_ids = []
            topic_ids = []
        else:
            budget_for_topic = budget_for_prefix_and_topic - len(prefix_ids_full)
            prefix_ids = prefix_ids_full
            topic_ids = topic_ids_full[:budget_for_topic]
    topic_truncated = len(topic_ids) < len(topic_ids_full)

    content_ids = tokenizer.build_inputs_with_special_tokens(
        prefix_ids + topic_ids + connector_ids + text_ids + tail_ids
    )
    if len(content_ids) > max_length:
        raise RuntimeError(
            f"内部错误：截断预算计算有误，构造出的序列长度 {len(content_ids)} "
            f"超过 max_length={max_length}，请检查 build_prompt_encoding 的预算计算。"
        )

    attention_mask = [1] * len(content_ids)
    pad_len = max_length - len(content_ids)
    if pad_len > 0:
        pad_id = tokenizer.pad_token_id
        content_ids = content_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    mask_positions = [i for i, token_id in enumerate(content_ids) if token_id == mask_token_id]
    if not mask_positions:
        raise RuntimeError("截断后丢失了 [MASK]，请检查 max_length 设置或截断逻辑。")

    return {
        "input_ids": content_ids,
        "attention_mask": attention_mask,
        "mask_position": mask_positions[-1],
        "topic_truncated": topic_truncated,
        "text_truncated": text_truncated,
    }


class PromptDataset(Dataset):
    """Prompt 格式："结合 {retweeted_content} ，判断{text}是否是反讽表达，[MASK]"。

    只截断 retweeted_content；text 与 prompt 固定结构始终保留完整。
    """

    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        encoded = build_prompt_encoding(
            self.tokenizer, example["text"], example.get("retweeted_content", ""),
            self.max_length,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "mask_position": torch.tensor(encoded["mask_position"], dtype=torch.long),
            "labels": torch.tensor(example["label"], dtype=torch.long),
        }


def prepare_datasets(tokenizer, train_file):
    """复用 train_cls.py 的加载与校验逻辑，只替换 Dataset 实现。"""
    train_raw = train_cls.load_jsonl(train_file)
    test_raw = train_cls.load_jsonl(TEST_FILE)
    train_cls.validate_dataset_distributions(train_raw, test_raw)

    random.seed(RANDOM_SEED)
    random.shuffle(train_raw)

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

    train_ds = PromptDataset(train_raw, tokenizer, MAX_LENGTH)
    test_ds = PromptDataset(test_raw, tokenizer, MAX_LENGTH)
    return train_ds, test_ds, len(unique_labels), label2id, id2label


# ---------------------------------------------------------------------------
# 数据统计与逐样本预测：input 结构和 train_cls.py 不一样，单独实现，
# 不强行套用 experiment_tracking.py 里为 content(+context) 拼接设计的口径。
# ---------------------------------------------------------------------------
def summarize_prompt_split(rows, tokenizer, max_length, split_name, file_path, seed):
    n = len(rows)
    label_counts = {}
    empty_topic = 0
    topic_truncated_count = 0
    text_truncated_count = 0
    seen = set()
    duplicates = 0

    for row in rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
        topic = row.get("retweeted_content", "")
        if not str(topic or "").strip():
            empty_topic += 1
        dedup_key = (row.get("text", ""), topic)
        if dedup_key in seen:
            duplicates += 1
        else:
            seen.add(dedup_key)
        encoded = build_prompt_encoding(tokenizer, row["text"], topic, max_length)
        topic_truncated_count += int(encoded["topic_truncated"])
        text_truncated_count += int(encoded["text_truncated"])

    return {
        "split": split_name,
        "file": os.fspath(file_path) if file_path else None,
        "file_sha256": et.file_sha256(file_path) if file_path else None,
        "num_samples": n,
        "label_counts": {str(k): v for k, v in label_counts.items()},
        "split_seed": seed,
        "empty_topic_count": empty_topic,
        "duplicate_count": duplicates,
        "max_length": max_length,
        "topic_truncated_count": topic_truncated_count,
        "topic_truncated_ratio": (topic_truncated_count / n) if n else None,
        "text_truncated_count": text_truncated_count,
        "prompt_format": "结合 {retweeted_content} ，判断{text}是否是反讽表达，[MASK]",
    }


def build_prompt_prediction_rows(rows, verbalizer_logits, label_ids, split,
                                 tokenizer, max_length, id2label,
                                 classification_threshold=None):
    """逐样本预测记录：字段与 experiment_tracking.build_analysis_digest 期望的
    结构保持一致（split/is_correct/error_type/probabilities/... ），
    这样可以直接复用现成的 build_analysis_digest，不需要另写一套摘要逻辑。
    """
    probs = et.softmax(verbalizer_logits)
    probs = np.asarray(probs)
    if classification_threshold is not None:
        preds = (probs[:, 1] >= classification_threshold).astype(int)
    else:
        preds = probs.argmax(axis=-1)

    out = []
    for i, row in enumerate(rows):
        pred = int(preds[i])
        true = int(label_ids[i]) if label_ids is not None else None
        topic = row.get("retweeted_content", "")
        encoded = build_prompt_encoding(tokenizer, row["text"], topic, max_length)
        prob_row = probs[i]
        prob_map = {
            str(id2label.get(c, c)): float(prob_row[c]) for c in range(len(prob_row))
        }
        out.append({
            "id": i,
            "split": split,
            "content": row.get("text", ""),
            "context": topic or None,
            "true_label": true,
            "predicted_label": pred,
            "classification_threshold": classification_threshold,
            "probabilities": prob_map,
            "is_correct": (true == pred) if true is not None else None,
            "error_type": et.error_type(true, pred) if true is not None else None,
            "content_token_length": None,
            "context_token_length": None,
            "total_token_length": len(encoded["input_ids"]),
            "was_truncated": bool(encoded["topic_truncated"] or encoded["text_truncated"]),
        })
    return out


# ============================================================
# 训练损失：只对 [MASK] 位置的"是"/"否" 两个候选词 logits 算损失，
# 不是整句 MLM 损失。cross_entropy / focal 两种与 train_cls.py 保持同名同义。
# ============================================================
class BasePromptMLMTrainer(Trainer):
    def __init__(self, *args, verbalizer_ids, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbalizer_ids = verbalizer_ids  # [neg_id, pos_id]

    def _verbalizer_logits(self, model, inputs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        mask_position = inputs["mask_position"]
        batch_indices = torch.arange(outputs.logits.size(0), device=outputs.logits.device)
        mask_logits = outputs.logits[batch_indices, mask_position]
        return mask_logits[:, self.verbalizer_ids]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        verbalizer_logits = self._verbalizer_logits(model, inputs)
        loss = self._loss_from_verbalizer_logits(verbalizer_logits, labels)
        return (loss, {"logits": verbalizer_logits}) if return_outputs else loss

    def _loss_from_verbalizer_logits(self, verbalizer_logits, labels):
        raise NotImplementedError


class PromptCrossEntropyTrainer(BasePromptMLMTrainer):
    def _loss_from_verbalizer_logits(self, verbalizer_logits, labels):
        return CrossEntropyLoss()(verbalizer_logits, labels)


class PromptFocalTrainer(BasePromptMLMTrainer):
    def _loss_from_verbalizer_logits(self, verbalizer_logits, labels):
        ce_loss = CrossEntropyLoss(reduction="none")(verbalizer_logits, labels)
        probs = torch.softmax(verbalizer_logits, dim=-1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        alpha_t = torch.where(
            labels == 1,
            torch.tensor(FOCAL_ALPHA, device=verbalizer_logits.device),
            torch.tensor(1 - FOCAL_ALPHA, device=verbalizer_logits.device),
        )
        return (alpha_t * (1 - pt) ** FOCAL_GAMMA * ce_loss).mean()


def loss_config():
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

    paths = et.create_experiment_dir(OUTPUT_DIR)
    task_name = os.path.basename(os.path.normpath(str(OUTPUT_DIR)))
    source_files = [
        os.path.join(os.path.dirname(__file__), "train_cls_prompt.py"),
        str(Path(__file__).resolve().parents[1] / "pipeline_config.py"),
    ]
    started_ts = time.monotonic()
    manifest = et.init_manifest(paths, task_name, sys.argv,
                                repo_dir=os.path.dirname(__file__),
                                source_files=source_files)
    print(f"实验目录：{paths['exp_dir']}  (状态: running)")

    _tee_out = et.Tee(sys.stdout, paths["run_log_path"])
    _tee_err = et.Tee(sys.stderr, paths["run_log_path"])
    sys.stdout, sys.stderr = _tee_out, _tee_err

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.model_max_length = MAX_LENGTH
        verbalizer_ids = resolve_verbalizer_ids(tokenizer)

        train_ds, test_ds, num_labels, label2id, id2label = prepare_datasets(
            tokenizer, args.train_file
        )
        print(f"训练集：{len(train_ds)}  测试集：{len(test_ds)}")
        print(f"类别数：{num_labels}  标签映射：{label2id}")
        print(f"Prompt 格式：{PROMPT_PREFIX}{{retweeted_content}}{PROMPT_CONNECTOR}"
              f"{{text}}{PROMPT_SUFFIX}[MASK]")
        print(f"Verbalizer：负={VERBALIZER_NEGATIVE!r}(id={verbalizer_ids[0]})，"
              f"正={VERBALIZER_POSITIVE!r}(id={verbalizer_ids[1]})")
        print(f"训练损失：{LOSS_TYPE}")

        model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)

        # 按 0.5 个 epoch 打印一次日志，而不是固定步数：固定的 logging_steps=10
        # 在样本量变化（挖负例后每轮 train.jsonl 大小不同）时，打印频率相对 epoch
        # 的比例会跟着漂移，数据一大就变成没几步刷一行、疯狂刷屏。
        steps_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)
        logging_steps = max(1, round(steps_per_epoch * 0.5))

        training_args = TrainingArguments(
            output_dir=paths["checkpoints_dir"],
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            num_train_epochs=NUM_EPOCHS,
            lr_scheduler_type="cosine",
            warmup_ratio=WARMUP_RATIO,
            optim=OPTIM,
            logging_dir=paths["logs_dir"],
            logging_strategy="steps",
            logging_steps=logging_steps,
            disable_tqdm=True,  # 关掉逐 step 刷新的进度条，只留 logging_steps 打印的日志行
            eval_strategy="no",
            save_strategy="epoch",
            save_safetensors=True,
            save_total_limit=3,
            load_best_model_at_end=False,
            max_grad_norm=1.0,
            report_to=["tensorboard"],
            fp16=False,
            push_to_hub=False,
            seed=RANDOM_SEED,
            data_seed=RANDOM_SEED,
            # AutoModelForMaskedLM.forward() 不认识 mask_position 这个自定义字段，
            # Trainer 默认 remove_unused_columns=True 会在喂给模型前把它删掉，
            # 导致 compute_loss 里 inputs["mask_position"] 报 KeyError。
            # 关掉自动清理，让 mask_position 完整保留到 compute_loss。
            remove_unused_columns=False,
        )
        print(f"每轮(epoch) {steps_per_epoch} 步，按 0.5 epoch 打印一次日志"
              f"（logging_steps={logging_steps}）")

        et.write_json(paths["train_config_path"], et.build_train_config(
            training_args,
            runtime={
                "model_name_or_path": MODEL_PATH,
                "tokenizer_name_or_path": MODEL_PATH,
                "max_length": MAX_LENGTH,
                "label2id": label2id,
                "use_context": True,
                "context_concat_format": (
                    "prompt_mlm: '结合 {retweeted_content} ，判断{text}是否是反讽表达，[MASK]'"
                    "（只截断 retweeted_content，text 与固定结构始终保留完整）"
                ),
                "truncation": True,
                "padding": "max_length",
                "early_stopping": None,
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

        et.write_json(paths["data_summary_path"], {
            "split_method": "custom train/test ratios",
            "split_seed": RANDOM_SEED,
            "train": summarize_prompt_split(
                train_ds.data, tokenizer, MAX_LENGTH, "train", args.train_file, RANDOM_SEED),
            "test": summarize_prompt_split(
                test_ds.data, tokenizer, MAX_LENGTH, "test", TEST_FILE, RANDOM_SEED),
        })

        writer = SummaryWriter(log_dir=paths["logs_dir"])
        trainer_class = PromptFocalTrainer if LOSS_TYPE == "focal" else PromptCrossEntropyTrainer
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            callbacks=[TensorBoardCallback(writer)],
            verbalizer_ids=verbalizer_ids,
        )

        trainer.train()

        et.write_jsonl(paths["history_path"],
                       et.build_history_rows(trainer.state.log_history))

        test_pred = trainer.predict(test_ds, metric_key_prefix="test")
        selected_threshold = CLASSIFICATION_THRESHOLD
        # verbalizer_logits 形状与序列分类的 (N, 2) logits 一致（列0=负类，列1=正类），
        # 直接复用 train_cls.py 的固定阈值评估口径。
        test_metrics = dict(test_pred.metrics)
        test_metrics.update(train_cls.fixed_evaluation_metrics(
            test_pred.predictions, test_pred.label_ids, selected_threshold, "test"
        ))

        print("\n===== 测试集最终评估（训练前固定阈值）=====")
        print(test_metrics)

        train_raw = {}
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry:
                train_raw = {"train_loss": entry["loss"], "epoch": entry.get("epoch")}
                break

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

        pred_rows = build_prompt_prediction_rows(
            test_ds.data, test_pred.predictions, test_pred.label_ids, "test",
            tokenizer, MAX_LENGTH, id2label,
            classification_threshold=selected_threshold)
        et.write_jsonl(paths["predictions_path"], pred_rows)

        metrics_obj = json.loads(Path(paths["metrics_path"]).read_text(encoding="utf-8"))
        digest = et.build_analysis_digest(pred_rows, metrics_obj, top_k=20, id2label=id2label)
        et.write_json(paths["digest_path"], digest)

        best_path = paths["best_model_dir"]
        trainer.save_model(best_path)
        tokenizer.save_pretrained(best_path)
        print(f"\n最终模型已保存至：{best_path}")

        et.finalize_manifest(paths, manifest, "completed", started_ts)
        et.update_latest(OUTPUT_DIR, paths["experiment_id"])
        et.populate_sync_dir(paths)
        print(f"实验完成：{paths['experiment_id']} (状态: completed)")

    except Exception as exc:
        et.finalize_manifest(paths, manifest, "failed", started_ts, error=exc)
        et.populate_sync_dir(paths)
        print(f"实验失败：{paths['experiment_id']} (状态: failed) -> {type(exc).__name__}: {exc}")
        raise
    finally:
        sys.stdout, sys.stderr = _tee_out._stream, _tee_err._stream
        _tee_out.close()
        _tee_err.close()


# ---------------------------------------------------------------------------
# 自检：只验证截断算法本身，用最小可行的假 tokenizer（按字符切分），
# 不需要加载真实模型/词表，可在没有 GPU 的机器上运行。
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    """最小可行的假分词器：每个字符是一个 token，专门用来验证截断算法。"""

    def __init__(self):
        self.mask_token_id = 1
        self.pad_token_id = 0
        self.unk_token_id = 2
        self.cls_id = 3
        self.sep_id = 4
        self._vocab = {"[MASK]": 1, "[PAD]": 0, "[UNK]": 2, "是": 5, "否": 6}

    def encode(self, text, add_special_tokens=False):
        # "是"/"否" 映射为固定 id（模拟词表里存在的单字 token）；
        # 其余字符映射为 (字符码点 % 90) + 10，避免和特殊 id 冲突，纯粹用于测试。
        return [self._vocab.get(ch, (ord(ch) % 90) + 10) for ch in text]

    def num_special_tokens_to_add(self, pair=False):
        return 2  # 模拟 CLS + SEP

    def build_inputs_with_special_tokens(self, ids):
        return [self.cls_id] + list(ids) + [self.sep_id]

def _selfcheck():
    tokenizer = _FakeTokenizer()

    # 1) topic 远超预算时，只截断 topic，text/前后缀结构完整保留，[MASK] 仍存在。
    long_topic = "话" * 500
    encoded = build_prompt_encoding(tokenizer, "这条评论很短", long_topic, max_length=64)
    assert len(encoded["input_ids"]) == 64
    assert encoded["topic_truncated"] is True
    assert encoded["text_truncated"] is False
    assert encoded["input_ids"][encoded["mask_position"]] == tokenizer.mask_token_id

    # 2) topic 为空、内容很短时不应触发任何截断，且仍能正确定位 [MASK]。
    encoded_short = build_prompt_encoding(tokenizer, "短评论", "", max_length=64)
    assert encoded_short["topic_truncated"] is False
    assert encoded_short["text_truncated"] is False
    assert encoded_short["input_ids"][encoded_short["mask_position"]] == tokenizer.mask_token_id

    # 3) 极端兜底：max_length 小到连"connector+完整正文+固定后缀"都装不下时，
    #    应先舍弃 prefix/topic，再从正文尾部截断，text_truncated 应为 True，
    #    但 [MASK] 必须始终存在（[MASK] 优先级最高）。
    encoded_extreme = build_prompt_encoding(tokenizer, "这是一段比较长的正文测试文字", "话题",
                                            max_length=6)
    assert encoded_extreme["text_truncated"] is True
    assert len(encoded_extreme["input_ids"]) == 6
    assert encoded_extreme["input_ids"][encoded_extreme["mask_position"]] == tokenizer.mask_token_id

    # 3b) 更极端：连固定连接词+固定后缀（不含正文）都装不下时，
    #     只保留固定后缀末尾（含 [MASK]），prefix/topic/connector/text 全部舍弃。
    encoded_most_extreme = build_prompt_encoding(tokenizer, "正文", "话题", max_length=3)
    assert encoded_most_extreme["text_truncated"] is True
    assert len(encoded_most_extreme["input_ids"]) == 3
    assert (encoded_most_extreme["input_ids"][encoded_most_extreme["mask_position"]]
            == tokenizer.mask_token_id)

    # 3c) 中间情形：固定结构都能放下，但正文本身太长，只截断正文尾部，
    #     此时 prefix/topic 也会被舍弃（预算优先满足 connector+text+tail）。
    long_text = "这是一段用来测试正文截断逻辑的比较长的评论文字内容"  # 24 字
    encoded_partial_text = build_prompt_encoding(tokenizer, long_text, "话题", max_length=25)
    assert encoded_partial_text["text_truncated"] is True
    assert len(encoded_partial_text["input_ids"]) == 25
    assert (encoded_partial_text["input_ids"][encoded_partial_text["mask_position"]]
            == tokenizer.mask_token_id)

    # 4) verbalizer 校验：正常应通过；候选词在词表里被拆成多个 token 时应拒绝
    #    （这正是 ModernBERT 等 BPE/字节级分词器最常见的失败场景）。
    assert resolve_verbalizer_ids(tokenizer) == [6, 5]

    class _BpeSplitTokenizer(_FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            # 模拟"是"/"否"被切成多个子词 token，其余字符行为不变。
            if text in (VERBALIZER_NEGATIVE, VERBALIZER_POSITIVE):
                return [20, 21]
            return super().encode(text, add_special_tokens)

    try:
        resolve_verbalizer_ids(_BpeSplitTokenizer())
        raise AssertionError("verbalizer 候选词被拆成多个 token 时应抛出 ValueError")
    except ValueError:
        pass

    # 5) loss_config 与 train_cls.py 同名同义，覆盖 focal/cross_entropy 两种取值。
    global LOSS_TYPE
    original_loss_type = LOSS_TYPE
    LOSS_TYPE = "cross_entropy"
    assert loss_config() == {"type": "cross_entropy", "class_weight": None}
    LOSS_TYPE = "focal"
    assert loss_config()["type"] == "focal_loss"
    LOSS_TYPE = original_loss_type

    print("自检通过")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

"""用训练好的分类模型扫描"模型判断与原始标签不一致"的候选纠错样本。

背景：多轮实验里同一批负样本（label=0）反复被模型高置信度误判为讽刺（label=1），
人工复核后确认这批样本标签本身没问题，怀疑模型存在系统性偏见，而非单纯超参数
问题（详见 工作日志.md 2026-08-03 记录）。本脚本不直接改标签、不重新训练，只是
把模型"最有信心判错"的样本挑出来，交给人工做最终判断。

扫描范围：标注数据 LABELED_FILE（本身已不含重复记录和 OCR 重复——新数据由
build_pending.py 在标注前处理，历史数据由 clean_labeled.py 补齐）。
默认扫全体数据，包括已进入 train/val/test 的样本，用 in_training 列区分——
训练集内样本的高置信度分歧可能来自模型对训练数据的记忆，人工复核时可信度
要打折扣，但仍值得看（复核目的是查标签本身，不是查模型泛化能力）。
如果只想看模型完全没见过的样本（置信度不受记忆效应影响），加
--exclude-training。

本次默认扫描全量已标注数据，只导出"原标签1、模型判0"（1to0）的样本；不设
置信度门槛，供人工复核时自行判断是否为标注错误。

重要：默认使用上下文实验 exp_20260813_152927，自动从其 train_config.json
读取 use_context 和 max_length，确保推理输入与训练保持一致。

用法：
    python3 model_train/tools/scan_label_disagreements.py
    python3 model_train/tools/scan_label_disagreements.py --threshold 0.90
    python3 model_train/tools/scan_label_disagreements.py --direction 0to1
    python3 model_train/tools/scan_label_disagreements.py --exclude-training
    python3 model_train/tools/scan_label_disagreements.py --selfcheck   # 不加载模型的纯逻辑自检
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import LABELED_FILE, SPLIT_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tracking import softmax

# ============================================================
# 【可调参数区】——按需修改，改完直接重跑即可
# ============================================================
# 默认使用上下文模型实验目录。通过实验目录读取模型、阈值和输入配置，避免
# 手动设置 use_context / max_length 与训练配置不一致。
EXPERIMENT_DIR = os.environ.get(
    "SARCASM_EXPERIMENT_DIR",
    str(Path(__file__).resolve().parents[3] / "models" / "v4"),
)
MODEL_PATH = f"{EXPERIMENT_DIR}/best_model"

# 是否拼接上下文（retweeted_content）作为输入。
# 必须与所用模型训练时的设置一致，否则输入分布不匹配、输出概率不可信：
#   - 拼接上下文的模型 -> True
#   - 纯正文模型（train_cls.py 产出）        -> False
USE_CONTEXT = True

# 0 表示导出所有 1to0 分歧，不因置信度过滤；CSV 会按置信度降序，便于优先复核。
CONFIDENCE_THRESHOLD = 0.0

# 旧模型未保存 decision_config 时的兼容默认值；新实验应通过 --experiment-dir
# 读取验证集选出的业务阈值。
CLASSIFICATION_THRESHOLD = 0.5

# 只筛原标签1、模型预测0的样本。
TARGET_DIRECTION = "1to0"

# 是否排除训练集内样本（已进入 train/val/test 的记录）。
# False（默认）：全体数据一起扫，包括训练集，结果里用 in_training 列区分——
#   训练集内样本的高置信度分歧可能来自模型记忆，可信度打折扣，但仍可复核。
# True：只扫模型从未见过的样本，置信度不受记忆效应影响，筛出的分歧最可信。
EXCLUDE_TRAINING = False

# 手动指定模型时的回退值；默认实验目录会覆盖为 train_config.json 中的值（384）。
MAX_LENGTH = 384

# 推理 batch size：显存不足时调小。
BATCH_SIZE = 64

# 指定 GPU；CPU 推理则留空字符串 ""。
CUDA_VISIBLE_DEVICES = "0"

# 输出候选清单路径（CSV，utf-8-sig，Excel 直接打开）。
OUTPUT_CSV = Path(__file__).resolve().parents[2] / "outputs" / "context_label_1to0_review.csv"

# 调试用：只扫描排除池的前 N 条（None = 不限制，扫全部）。
LIMIT_EXCLUDED_POOL = None
# ============================================================

# setdefault：允许调用方提前 export CUDA_VISIBLE_DEVICES=1 切换到别的卡
# （共享多卡机器上卡0偶尔会被其它任务占满导致 OOM），不显式设置时仍默认用卡0。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", CUDA_VISIBLE_DEVICES)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_text(value):
    return " ".join(str(value or "").split())


def load_training_source_lines(data_path=None):
    """读取正式 splits 中参与 train/val/test 的 source_line。

    传入旧 balanced 文件时仍按单文件读取，供历史旁路脚本兼容使用。
    """
    data_path = Path(data_path or SPLIT_DIR)
    paths = (
        [data_path / f"{name}.jsonl" for name in ("train", "val", "test")]
        if data_path.is_dir() else [data_path]
    )
    rows = []
    for path in paths:
        if path.exists():
            rows.extend(read_jsonl(path))
    return {row["source_line"] for row in rows if "source_line" in row}


def build_candidate_pool(limit=None, exclude_training=EXCLUDE_TRAINING):
    """候选池：读标注数据 LABELED_FILE，过滤空文本。

    exclude_training=False（默认）时全量扫描（含训练集内样本），用
    in_training 列区分；True 时只保留从未进入 train/val/test 的样本
    （模型没见过，置信度不受记忆效应影响）。
    """
    training_lines = load_training_source_lines()
    rows = []
    for r in read_jsonl(LABELED_FILE):
        text = normalize_text(r.get("content", ""))
        if not text:
            continue
        in_training = r.get("source_line") in training_lines
        if exclude_training and in_training:
            continue
        rows.append({
            "source": "deduped",
            "source_line": r.get("source_line"),
            "text": text,
            "retweeted_content": normalize_text(r.get("retweeted_content", "")),
            "label": int(r["is_sarcasm"]),
            "in_training": in_training,
        })
        if limit is not None and len(rows) >= limit:
            break
    return rows


def build_input_text(row, use_context):
    """复刻 train_cls.CustomDataset.__getitem__ 的输入拼接逻辑。

    必须与训练时完全一致，否则模型看到的输入分布和训练时不同，
    输出概率不可信（早期版本漏了这一步，只喂正文去跑上下文模型）。
    """
    text = row.get("text", "") or ""
    if use_context and row.get("retweeted_content"):
        return text + "\n" + row["retweeted_content"]
    return text


def predict_batches(rows, model_path, max_length, batch_size,
                    use_context=USE_CONTEXT,
                    classification_threshold=CLASSIFICATION_THRESHOLD):
    """批量推理，返回每条样本的 (predicted_label, confidence)。

    输入拼接方式由 use_context 决定，必须与所用模型训练时一致
    （见 build_input_text）。

    torch/transformers 延迟导入：--selfcheck 不需要加载真实模型，
    本机（无训练依赖的环境）也能跑通纯逻辑自检。
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    preds, confidences = [], []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            texts = [build_input_text(r, use_context) for r in batch]
            enc = tokenizer(
                texts, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.cpu().numpy()
            probs = softmax(logits)
            for p in probs:
                pred = int(p[1] >= classification_threshold)
                preds.append(pred)
                confidences.append(float(p[pred]))
    return preds, confidences


def resolve_inference_config(experiment_dir=None, model_path=None,
                             classification_threshold=None,
                             use_context=None, max_length=None):
    """优先从 experiment_dir 读取模型输入配置和已冻结的业务阈值。"""
    if experiment_dir:
        experiment_dir = Path(experiment_dir)
        train_config = json.loads(
            (experiment_dir / "train_config.json").read_text(encoding="utf-8")
        )
        decision = json.loads(
            (experiment_dir / "decision_config.json").read_text(encoding="utf-8")
        )
        if decision.get("selected_on") not in {"validation", "predefined_before_training"}:
            raise ValueError(
                "decision_config 的 selected_on 必须为 validation 或 "
                "predefined_before_training"
            )
        default_model_path = experiment_dir / "best_model"
        default_threshold = decision["classification_threshold"]
        default_use_context = train_config["use_context"]
        default_max_length = train_config["max_length"]
    else:
        default_model_path = MODEL_PATH
        default_threshold = CLASSIFICATION_THRESHOLD
        default_use_context = USE_CONTEXT
        default_max_length = MAX_LENGTH

    resolved = {
        "model_path": str(model_path or default_model_path),
        "classification_threshold": float(
            default_threshold if classification_threshold is None else classification_threshold
        ),
        "use_context": (
            bool(default_use_context) if use_context is None else bool(use_context)
        ),
        "max_length": int(default_max_length if max_length is None else max_length),
    }
    if not 0 <= resolved["classification_threshold"] <= 1:
        raise ValueError("classification_threshold 必须在 [0, 1] 范围内")
    return resolved


def filter_disagreements(rows, preds, confidences, threshold, direction):
    """筛出指定方向、置信度达标的分歧样本。

    direction: '0to1'（原标签0/模型判1）、'1to0'（原标签1/模型判0）、
    'both'（只要原标签与模型判断不一致就收，不限方向——"标注不一致"复筛场景）。
    """
    out = []
    for r, pred, conf in zip(rows, preds, confidences):
        if r["label"] == pred or conf < threshold:
            continue
        actual_direction = f"{r['label']}to{pred}"
        if direction != "both" and actual_direction != direction:
            continue
        out.append({**r, "predicted_label": pred, "confidence": conf, "direction": actual_direction})
    return out


def write_candidates_csv(candidates, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates_sorted = sorted(candidates, key=lambda r: r["confidence"], reverse=True)
    fieldnames = [
        "content", "retweeted_content", "original_label",
        "predicted_label", "direction", "confidence", "in_training", "human_review_label",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in candidates_sorted:
            writer.writerow({
                "content": r["text"],
                "retweeted_content": r["retweeted_content"],
                "original_label": r["label"],
                "predicted_label": r["predicted_label"],
                "direction": r["direction"],
                "confidence": round(r["confidence"], 4),
                "in_training": r["in_training"],
                "human_review_label": "",  # 留空，人工复核时填写
            })


def main():
    parser = argparse.ArgumentParser(description="扫描模型与原始标签的分歧样本")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="模型预测类别的最低置信度；0 表示不按置信度过滤")
    parser.add_argument("--direction", choices=["0to1", "1to0", "both"], default=TARGET_DIRECTION,
                        help="覆盖脚本顶部的 TARGET_DIRECTION："
                             "0to1=原标签0/模型判1；1to0=原标签1/模型判0；"
                             "both=只要标注不一致就收")
    parser.add_argument("--exclude-training", action="store_true",
                        help="只扫训练集外的数据（默认全量扫描，含训练集内样本）")
    parser.add_argument("--experiment-dir", default=EXPERIMENT_DIR,
                        help="新实验目录；自动读取 best_model、train_config 和 decision_config")
    parser.add_argument("--model-path", help="手动覆盖模型目录")
    parser.add_argument("--classification-threshold", type=float,
                        help="手动覆盖分类阈值；默认读取 decision_config")
    parser.add_argument("--use-context", action="store_true", default=None)
    parser.add_argument("--no-context", dest="use_context", action="store_false")
    parser.add_argument("--max-length", type=int)
    args = parser.parse_args()

    inference = resolve_inference_config(
        args.experiment_dir, args.model_path, args.classification_threshold,
        args.use_context, args.max_length,
    )

    exclude_training = EXCLUDE_TRAINING or args.exclude_training
    scope = "仅训练集外样本" if exclude_training else "全量（含训练集内）"
    print(f"候选池构建中（LABELED_FILE，范围：{scope}，limit={LIMIT_EXCLUDED_POOL}）...")
    rows = build_candidate_pool(LIMIT_EXCLUDED_POOL, exclude_training=exclude_training)
    n_in_training = sum(1 for r in rows if r["in_training"])
    print(f"候选池总数：{len(rows)}"
          f"（参与过训练/验证/测试：{n_in_training}，"
          f"从未参与：{len(rows) - n_in_training}）")

    print(f"加载模型：{inference['model_path']}")
    print(f"输入方式：{'content + retweeted_content' if inference['use_context'] else 'content'}"
          f"，MAX_LENGTH={inference['max_length']}")
    print(f"分类阈值：{inference['classification_threshold']:.8f}")
    preds, confidences = predict_batches(
        rows,
        inference["model_path"],
        inference["max_length"],
        BATCH_SIZE,
        inference["use_context"],
        inference["classification_threshold"],
    )

    candidates = filter_disagreements(rows, preds, confidences, args.threshold, args.direction)

    print(f"\n方向：{args.direction}，置信度阈值：{args.threshold}")
    print(f"符合条件的候选纠错样本数：{len(candidates)} / {len(rows)}"
          f"（占比 {len(candidates) / len(rows):.2%}）")

    write_candidates_csv(candidates, OUTPUT_CSV)
    print(f"已写入：{OUTPUT_CSV}")


def _selfcheck():
    """不加载模型，只验证输入拼接与方向/阈值筛选逻辑。"""
    import tempfile

    # 训练前固定阈值的上下文实验必须能自动读取模型输入配置。
    with tempfile.TemporaryDirectory() as tmp:
        experiment_dir = Path(tmp)
        (experiment_dir / "best_model").mkdir()
        (experiment_dir / "train_config.json").write_text(
            json.dumps({"max_length": 384, "use_context": True}), encoding="utf-8"
        )
        (experiment_dir / "decision_config.json").write_text(
            json.dumps({
                "selected_on": "predefined_before_training",
                "classification_threshold": 0.5,
            }),
            encoding="utf-8",
        )
        resolved = resolve_inference_config(experiment_dir=experiment_dir)
        assert resolved["model_path"] == str(experiment_dir / "best_model"), resolved
        assert resolved["classification_threshold"] == 0.5, resolved
        assert resolved["use_context"] is True and resolved["max_length"] == 384, resolved

    inference = resolve_inference_config(
        model_path="model", classification_threshold=0.8,
        use_context=True, max_length=384,
    )
    assert inference == {
        "model_path": "model",
        "classification_threshold": 0.8,
        "use_context": True,
        "max_length": 384,
    }
    rows = [
        {"text": "a", "retweeted_content": "", "label": 0, "in_training": True},
        {"text": "b", "retweeted_content": "", "label": 0, "in_training": False},
        {"text": "c", "retweeted_content": "", "label": 1, "in_training": False},
        {"text": "d", "retweeted_content": "", "label": 0, "in_training": True},
    ]
    preds = [1, 1, 0, 0]
    confidences = [0.95, 0.80, 0.99, 0.99]

    out_90 = filter_disagreements(rows, preds, confidences, 0.90, "0to1")
    assert [r["text"] for r in out_90] == ["a"], out_90

    out_70 = filter_disagreements(rows, preds, confidences, 0.70, "0to1")
    assert [r["text"] for r in out_70] == ["a", "b"], out_70

    out_1to0 = filter_disagreements(rows, preds, confidences, 0.90, "1to0")
    assert [r["text"] for r in out_1to0] == ["c"], out_1to0

    # both：不限方向，只要标注不一致且置信度达标就收（d 的 label/pred 相同，
    # 始终不算分歧，用来验证 both 不会把"判断正确"的样本也收进来）。
    out_both = filter_disagreements(rows, preds, confidences, 0.90, "both")
    assert [r["text"] for r in out_both] == ["a", "c"], out_both
    assert [r["direction"] for r in out_both] == ["0to1", "1to0"], out_both

    # 输入拼接必须与 train_cls.CustomDataset 一致：
    # use_context=True 且有上下文时才拼，否则只用正文。
    with_ctx = {"text": "评论", "retweeted_content": "原帖"}
    assert build_input_text(with_ctx, True) == "评论\n原帖"
    assert build_input_text(with_ctx, False) == "评论"
    no_ctx = {"text": "评论", "retweeted_content": ""}
    assert build_input_text(no_ctx, True) == "评论"

    print("自检通过")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

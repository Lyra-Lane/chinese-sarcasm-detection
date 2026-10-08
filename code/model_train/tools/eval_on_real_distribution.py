"""使用实验中已冻结的阈值，在任意正负比例的评估集上做最终评估。"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import SPLIT_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tracking import softmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_label_disagreements import build_input_text, read_jsonl


EVAL_FILE = SPLIT_DIR / "test.jsonl"
BATCH_SIZE = 64
CUDA_VISIBLE_DEVICES = "0"

# setdefault：允许调用方提前 export CUDA_VISIBLE_DEVICES=1 切换到别的卡
# （共享多卡机器上卡0偶尔会被其它任务占满导致 OOM），不显式设置时仍默认用卡0。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", CUDA_VISIBLE_DEVICES)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_experiment(experiment_dir):
    """从同一次实验读取模型、训练输入配置和已冻结的分类阈值。"""
    experiment_dir = Path(experiment_dir)
    model_path = experiment_dir / "best_model"
    train_config_path = experiment_dir / "train_config.json"
    decision_path = experiment_dir / "decision_config.json"
    for path in (model_path, train_config_path, decision_path):
        if not path.exists():
            raise FileNotFoundError(f"实验文件不存在：{path}")

    train_config = read_json(train_config_path)
    decision = read_json(decision_path)
    selected_on = decision.get("selected_on")
    if selected_on not in {"validation", "predefined_before_training"}:
        raise ValueError(
            "decision_config 的 selected_on 必须为 validation 或 "
            "predefined_before_training"
        )

    threshold = decision.get("classification_threshold")
    if threshold is None or not 0 <= float(threshold) <= 1:
        raise ValueError("decision_config 缺少有效 classification_threshold")
    return {
        "model_path": model_path,
        "threshold": float(threshold),
        "use_context": bool(train_config.get("use_context", False)),
        "max_length": int(train_config["max_length"]),
        "decision": decision,
        "threshold_source": f"{selected_on} decision_config.json",
    }


def predict_probs(rows, model_path, max_length, batch_size, use_context):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    positive_probs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            texts = [build_input_text(row, use_context) for row in batch]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits.cpu().numpy()
            positive_probs.extend(float(row[1]) for row in softmax(logits))
    return positive_probs


def _fingerprint(eval_file, model_path, max_length, use_context):
    digest = hashlib.sha256()
    digest.update(str(model_path).encode("utf-8"))
    digest.update(str(max_length).encode("utf-8"))
    digest.update(str(bool(use_context)).encode("utf-8"))
    with Path(eval_file).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cached_probs(cache_path, eval_file, model_path, max_length, use_context):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        data = read_json(cache_path)
    except (json.JSONDecodeError, OSError):
        return None
    expected = _fingerprint(eval_file, model_path, max_length, use_context)
    return data.get("positive_probs") if data.get("fingerprint") == expected else None


def save_probs_cache(cache_path, eval_file, model_path, max_length, use_context,
                     positive_probs):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump({
            "fingerprint": _fingerprint(
                eval_file, model_path, max_length, use_context
            ),
            "positive_probs": positive_probs,
        }, f)


def compute_metrics(labels, positive_probs, threshold):
    if len(labels) != len(positive_probs):
        raise ValueError(
            f"标签数 {len(labels)} 与预测概率数 {len(positive_probs)} 不一致"
        )
    tp = tn = fp = fn = 0
    for label, probability in zip(labels, positive_probs):
        prediction = int(probability >= threshold)
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 0:
            tn += 1
        elif label == 0 and prediction == 1:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "total": total,
        "positive_count": tp + fn,
        "negative_count": tn + fp,
        "positive_ratio": (tp + fn) / total if total else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "sarcasm_precision": precision,
        "sarcasm_recall": recall,
        "sarcasm_f1": f1,
        "false_positive_rate": 1.0 - specificity,
        "specificity": specificity,
        "predicted_positive_ratio": (tp + fp) / total if total else 0.0,
        "target_reached": precision >= 0.60 and recall >= 0.80,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True,
                        help="训练输出的 experiments/exp_xxx 目录")
    parser.add_argument("--eval-file", default=str(EVAL_FILE),
                        help="待评估 JSONL；可为真实数据比例评估集")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output", help="默认写入实验目录 test_deployment_metrics.json")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--probs-cache", help="默认写入实验目录 test_probs_cache.json")
    args = parser.parse_args()

    resolved = resolve_experiment(args.experiment_dir)
    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        raise FileNotFoundError(f"正式测试集不存在：{eval_file}，请先运行 split")

    rows = read_jsonl(eval_file)
    labels = [int(row["label"]) for row in rows]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise ValueError(
            f"评估集必须同时包含正、负类，实际正类={positive_count}、负类={negative_count}"
        )
    output_path = Path(args.output or Path(args.experiment_dir) / "test_deployment_metrics.json")
    cache_path = Path(args.probs_cache or Path(args.experiment_dir) / "test_probs_cache.json")

    positive_probs = None
    if args.use_cache:
        positive_probs = load_cached_probs(
            cache_path, eval_file, resolved["model_path"],
            resolved["max_length"], resolved["use_context"],
        )
        if positive_probs is not None and len(positive_probs) != len(rows):
            positive_probs = None
    if positive_probs is None:
        positive_probs = predict_probs(
            rows, resolved["model_path"], resolved["max_length"],
            args.batch_size, resolved["use_context"],
        )
        save_probs_cache(
            cache_path, eval_file, resolved["model_path"],
            resolved["max_length"], resolved["use_context"], positive_probs,
        )

    metrics = compute_metrics(labels, positive_probs, resolved["threshold"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "experiment_dir": str(args.experiment_dir),
            "model_path": str(resolved["model_path"]),
            "eval_file": str(eval_file),
            "threshold_source": resolved["threshold_source"],
            "use_context": resolved["use_context"],
            "max_length": resolved["max_length"],
            **metrics,
        }, f, ensure_ascii=False, indent=2)

    print(f"评估集：{len(rows)} 条，正类占比 {positive_count / len(labels):.4%}")
    print(f"固定阈值：{resolved['threshold']:.8f}（来自 {resolved['threshold_source']}）")
    print(f"precision={metrics['sarcasm_precision']:.4%}  "
          f"recall={metrics['sarcasm_recall']:.4%}  f1={metrics['sarcasm_f1']:.4%}")
    print(f"tp={metrics['tp']} tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']}")
    print(f"目标是否达成：{metrics['target_reached']}")
    print(f"已写入：{output_path}")


def _selfcheck():
    import tempfile

    # 当前 train_cls.py 的训练前固定阈值实验应可被正确解析。
    with tempfile.TemporaryDirectory() as tmp:
        experiment_dir = Path(tmp)
        (experiment_dir / "best_model").mkdir()
        (experiment_dir / "train_config.json").write_text(
            json.dumps({"max_length": 256, "use_context": False}), encoding="utf-8"
        )
        (experiment_dir / "decision_config.json").write_text(
            json.dumps({
                "selected_on": "predefined_before_training",
                "classification_threshold": 0.5,
            }),
            encoding="utf-8",
        )
        resolved = resolve_experiment(experiment_dir)
        assert resolved["threshold"] == 0.5, resolved
        assert resolved["threshold_source"] == "predefined_before_training decision_config.json", resolved

    # 5% 正类的评估集同样应能计算指标，不依赖固定 1:N 比例。
    labels = [1, 1] + [0] * 38
    probabilities = [0.9, 0.1] + [0.9] * 2 + [0.05] * 36
    metrics = compute_metrics(labels, probabilities, 0.85)
    assert metrics["positive_ratio"] == 0.05, metrics
    assert metrics["tp"] == 1 and metrics["fn"] == 1, metrics
    assert metrics["fp"] == 2 and metrics["tn"] == 36, metrics
    print("自检通过")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

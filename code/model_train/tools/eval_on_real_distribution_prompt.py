"""Prompt+MLM 模型专用：使用实验中已冻结的阈值，在任意正负比例的评估集上做最终评估。

背景：train_cls_prompt.py（"结合 {retweeted_content} ，判断{text}是否是反讽表达，[MASK]"，
AutoModelForMaskedLM + 是/否 verbalizer）产出的模型不能直接复用
eval_on_real_distribution.py——原脚本用 AutoModelForSequenceClassification 加载
模型、对整句 CLS 输出做 softmax，两者模型结构和概率提取方式都不同，直接套用会
报错或算出完全错误的指标。

本文件不硬编码 prompt 文案：推理时直接调用 train_cls_prompt.build_prompt_encoding
构造输入，prompt 格式随 train_cls_prompt.py 的定义自动生效，两边不会不一致。

本脚本与 eval_on_real_distribution.py 是同一套流程骨架（阈值解析、指标计算、
概率缓存、CLI 参数），只替换了两处随训练范式而变的部分：
  - resolve_experiment：额外读取 use_context 时要求必须为 True（prompt 格式
    本身就固定拼接 retweeted_content 作为话题，与"是否使用上下文"这个开关
    等价，兼容 train_cls_prompt.py 写入 train_config.json 时的记法）。
  - predict_probs：加载 AutoModelForMaskedLM，用 train_cls_prompt.py 里已验证
    过的 build_prompt_encoding / resolve_verbalizer_ids 构造输入、定位 [MASK]
    位置、取"是"/"否"两个候选词的 logits 做 softmax，而不是对整句 CLS 输出
    做 softmax。共享多卡机器上显存会随其它租户任务瞬时波动，遇到 CUDA OOM
    时会自动对半拆分当前 batch 重试（_predict_batch_with_oom_retry），不会
    因为一次显存紧张就整体崩溃退出（被 mine_hard_negatives.py 挖负例阶段
    复用，是该问题最早出现、也最需要这个自愈能力的调用方）。

用法：
    python3 model_train/tools/eval_on_real_distribution_prompt.py --experiment-dir <exp_xxx 目录>
    python3 model_train/tools/eval_on_real_distribution_prompt.py --experiment-dir <exp_xxx 目录> --eval-file outputs/real_distribution_eval.jsonl
    python3 model_train/tools/eval_on_real_distribution_prompt.py --experiment-dir <exp_xxx 目录> --use-cache
    python3 model_train/tools/eval_on_real_distribution_prompt.py --selfcheck   # 不加载模型的纯逻辑自检
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import SPLIT_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tracking import softmax
# train_cls_prompt.py 里已验证过的截断 / verbalizer 解析逻辑，直接复用，
# 保证推理时的输入构造方式与训练时完全一致，不重新实现一遍。
from train_cls_prompt import build_prompt_encoding, resolve_verbalizer_ids

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 复用现有脚本已验证过的通用工具：与训练范式无关，两边行为必须一致。
from eval_on_real_distribution import (
    compute_metrics,
    load_cached_probs,
    read_json,
    save_probs_cache,
)
from scan_label_disagreements import read_jsonl


EVAL_FILE = SPLIT_DIR / "test.jsonl"
BATCH_SIZE = 64
CUDA_VISIBLE_DEVICES = "0"

# setdefault：允许调用方提前 export CUDA_VISIBLE_DEVICES=1 切换到别的卡
# （共享多卡机器上卡0偶尔会被其它任务占满导致 OOM），不显式设置时仍默认用卡0。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", CUDA_VISIBLE_DEVICES)


def resolve_experiment(experiment_dir):
    """从同一次实验读取模型、训练输入配置和已冻结的分类阈值。

    与 eval_on_real_distribution.resolve_experiment 的唯一区别：额外校验
    use_context 必须为 True——prompt 格式本身固定拼接 retweeted_content
    作为话题，如果 train_config.json 里 use_context=False，说明这不是一次
    正常的 train_cls_prompt.py 实验（或数据准备环节出了问题），直接拒绝，
    避免用错误假设的输入格式去推理。
    """
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

    if not train_config.get("use_context", False):
        raise ValueError(
            f"{train_config_path} 中 use_context 不为 True，"
            "这不像是 train_cls_prompt.py 产出的实验目录，请检查 --experiment-dir。"
        )

    return {
        "model_path": model_path,
        "threshold": float(threshold),
        "max_length": int(train_config["max_length"]),
        "decision": decision,
        "threshold_source": f"{selected_on} decision_config.json",
    }


def _run_forward_batch(model, tokenizer, verbalizer_ids, batch, max_length, device):
    """对一个 batch 跑一次前向，返回该 batch 每条样本的正类概率列表。

    不含重试逻辑，被 _predict_batch_with_oom_retry 调用。
    """
    import torch

    encoded_list = [
        build_prompt_encoding(
            tokenizer, row.get("text", ""), row.get("retweeted_content", ""), max_length,
        )
        for row in batch
    ]
    input_ids = torch.tensor(
        [e["input_ids"] for e in encoded_list], dtype=torch.long
    ).to(device)
    attention_mask = torch.tensor(
        [e["attention_mask"] for e in encoded_list], dtype=torch.long
    ).to(device)
    mask_positions = torch.tensor(
        [e["mask_position"] for e in encoded_list], dtype=torch.long
    ).to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    batch_indices = torch.arange(logits.size(0), device=device)
    mask_logits = logits[batch_indices, mask_positions]
    verbalizer_logits = mask_logits[:, verbalizer_ids].cpu().numpy()
    return [float(row[1]) for row in softmax(verbalizer_logits)]


def _predict_batch_with_oom_retry(model, tokenizer, verbalizer_ids, batch, max_length, device,
                                  min_batch_size=1):
    """遇到 CUDA OOM 时清空显存缓存、把当前 batch 对半拆开重试，直到成功或
    拆到 min_batch_size 仍失败才把原始异常抛出去。

    背景：共享多卡机器上显存占用会随其它租户任务瞬时波动（同一张卡几分钟内
    可用显存可能相差几十GB，见工作日志记录的多次 OOM），固定 batch_size 靠人工
    猜不是长久办法，遇到 OOM 就自动降级重试，能顺利跑完的概率更高。
    """
    import torch

    try:
        return _run_forward_batch(model, tokenizer, verbalizer_ids, batch, max_length, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) <= min_batch_size:
            raise
        mid = len(batch) // 2
        print(f"CUDA OOM，将 batch_size={len(batch)} 拆成两半（{mid}+{len(batch) - mid}）重试...",
              flush=True)
        first = _predict_batch_with_oom_retry(
            model, tokenizer, verbalizer_ids, batch[:mid], max_length, device, min_batch_size
        )
        second = _predict_batch_with_oom_retry(
            model, tokenizer, verbalizer_ids, batch[mid:], max_length, device, min_batch_size
        )
        return first + second


def predict_probs(rows, model_path, max_length, batch_size):
    """加载 Prompt+MLM 模型，对每条样本取 [MASK] 位置"是"（正类）的概率。

    与 eval_on_real_distribution.predict_probs 的区别：
    - 模型类是 AutoModelForMaskedLM，不是 AutoModelForSequenceClassification；
    - 输入不是简单拼接 content+context，而是完整 prompt（含 [MASK]），
      用 build_prompt_encoding 逐条构造，只截断话题部分，与训练时一致；
    - 取的不是整句 CLS 输出的 softmax，而是 [MASK] 位置上"是"/"否"两个
      候选词 logits 的 softmax（verbalizer），这是 Prompt+MLM 范式本身
      定义预测概率的方式。
    - 遇到 CUDA OOM 会自动对半拆分当前 batch 重试（见 _predict_batch_with_oom_retry），
      不会因为共享GPU显存瞬时紧张就直接崩溃退出。
    """
    import time

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.model_max_length = max_length
    verbalizer_ids = resolve_verbalizer_ids(tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
    model.eval()

    total = len(rows)
    progress_step = 10000
    next_progress = progress_step
    start_time = time.time()

    positive_probs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            positive_probs.extend(_predict_batch_with_oom_retry(
                model, tokenizer, verbalizer_ids, batch, max_length, device
            ))

            # ponytail: 每凑够 progress_step(1万) 条才打印一次，而不是每个 batch 都打，
            # 避免 batch_size 较小（如 8）时刷屏；用 len(positive_probs) 而不是单独计数器，
            # 天然保证跨 batch 边界也能及时触发。
            if len(positive_probs) >= next_progress or len(positive_probs) == total:
                elapsed = time.time() - start_time
                speed = len(positive_probs) / elapsed if elapsed > 0 else 0.0
                remaining = (total - len(positive_probs)) / speed if speed > 0 else float("inf")
                print(
                    f"进度: {len(positive_probs)}/{total} "
                    f"({len(positive_probs) / total:.1%})  "
                    f"耗时 {elapsed:.0f}s  预计剩余 {remaining:.0f}s",
                    flush=True,
                )
                next_progress += progress_step
    return positive_probs


def predict_probs_stream(row_iterable, model_path, max_length, batch_size):
    """流式版本的 predict_probs：接受可迭代对象（如逐行读文件的生成器），
    逐 batch 推理，通过生成器逐条 yield (row, probability)。

    与 predict_probs 的区别：不要求提前知道总条数、不整份物化输入列表；更重要的是，
    调用方可以在满足条件后用 for...break 提前终止迭代——row_iterable 若本身是惰性
    生成器（如 mine_hard_negatives.stream_candidate_pool 逐行读 LABELED_FILE），
    尚未被消费的部分永远不会被读取/推理。用于"边标边停"场景（见
    mine_hard_negatives.scan_and_collect_negatives）：候选池可能有上百万条，
    但往往只需扫描一小部分就能凑够所需的难负例数量，没必要标注全量数据。

    共享 predict_probs 同一套模型加载 / OOM 重试逻辑（_predict_batch_with_oom_retry）。
    """
    import time

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.model_max_length = max_length
    verbalizer_ids = resolve_verbalizer_ids(tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
    model.eval()

    progress_step = 10000
    next_progress = progress_step
    start_time = time.time()
    scanned = 0

    buffer = []
    with torch.no_grad():
        for row in row_iterable:
            buffer.append(row)
            if len(buffer) < batch_size:
                continue
            probs = _predict_batch_with_oom_retry(
                model, tokenizer, verbalizer_ids, buffer, max_length, device
            )
            for r, p in zip(buffer, probs):
                yield r, p
            scanned += len(buffer)
            buffer = []
            if scanned >= next_progress:
                elapsed = time.time() - start_time
                speed = scanned / elapsed if elapsed > 0 else 0.0
                print(f"已扫描 {scanned} 条，耗时 {elapsed:.0f}s，速度 {speed:.1f} 条/s", flush=True)
                next_progress += progress_step
        if buffer:
            probs = _predict_batch_with_oom_retry(
                model, tokenizer, verbalizer_ids, buffer, max_length, device
            )
            for r, p in zip(buffer, probs):
                yield r, p


def _fingerprint(eval_file, model_path, max_length):
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(model_path).encode("utf-8"))
    digest.update(str(max_length).encode("utf-8"))
    digest.update(b"prompt_mlm")  # 与序列分类模型的缓存指纹区分，避免混用缓存
    with Path(eval_file).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_review_csv(rows, labels, positive_probs, threshold, path):
    """输出人工审核用 CSV：评论、上下文、原始标记、模型标记（含置信度）。

    覆盖所有样本（不止分歧样本），方便人工按需筛选，utf-8-sig 便于 Excel 直接打开。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["评论", "上下文", "原始标记", "模型标记", "模型置信度", "是否分歧"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, label, prob in zip(rows, labels, positive_probs):
            predicted_label = int(prob >= threshold)
            confidence = prob if predicted_label == 1 else 1 - prob
            writer.writerow({
                "评论": row.get("text", ""),
                "上下文": row.get("retweeted_content", ""),
                "原始标记": label,
                "模型标记": predicted_label,
                "模型置信度": round(float(confidence), 4),
                "是否分歧": "是" if predicted_label != label else "",
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True,
                        help="train_cls_prompt.py 训练输出的 experiments/exp_xxx 目录")
    parser.add_argument("--eval-file", default=str(EVAL_FILE),
                        help="待评估 JSONL；可为真实数据比例评估集")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output", help="默认写入实验目录 test_deployment_metrics.json")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--probs-cache", help="默认写入实验目录 test_probs_cache.json")
    parser.add_argument("--review-csv",
                        help="人工审核 CSV 输出路径；默认写入实验目录 test_review.csv")
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
            cache_path, eval_file, resolved["model_path"], resolved["max_length"],
            use_context=True,  # prompt 格式固定使用 retweeted_content，指纹口径与训练侧一致
        )
        if positive_probs is not None and len(positive_probs) != len(rows):
            positive_probs = None
    if positive_probs is None:
        positive_probs = predict_probs(
            rows, resolved["model_path"], resolved["max_length"], args.batch_size,
        )
        save_probs_cache(
            cache_path, eval_file, resolved["model_path"], resolved["max_length"],
            use_context=True, positive_probs=positive_probs,
        )

    metrics = compute_metrics(labels, positive_probs, resolved["threshold"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "experiment_dir": str(args.experiment_dir),
            "model_path": str(resolved["model_path"]),
            "eval_file": str(eval_file),
            "threshold_source": resolved["threshold_source"],
            "max_length": resolved["max_length"],
            "training_paradigm": "prompt_mlm",
            **metrics,
        }, f, ensure_ascii=False, indent=2)

    print(f"评估集：{len(rows)} 条，正类占比 {positive_count / len(labels):.4%}")
    print(f"固定阈值：{resolved['threshold']:.8f}（来自 {resolved['threshold_source']}）")
    print(f"precision={metrics['sarcasm_precision']:.4%}  "
          f"recall={metrics['sarcasm_recall']:.4%}  f1={metrics['sarcasm_f1']:.4%}")
    print(f"tp={metrics['tp']} tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']}")
    print(f"目标是否达成：{metrics['target_reached']}")
    print(f"已写入：{output_path}")

    review_csv_path = Path(args.review_csv or Path(args.experiment_dir) / "test_review.csv")
    write_review_csv(rows, labels, positive_probs, resolved["threshold"], review_csv_path)
    print(f"人工审核 CSV 已写入：{review_csv_path}")


def _selfcheck():
    import tempfile

    # resolve_experiment 应正确解析 prompt 实验目录，且拒绝 use_context=False。
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
        resolved = resolve_experiment(experiment_dir)
        assert resolved["threshold"] == 0.5, resolved
        assert resolved["max_length"] == 384, resolved
        assert resolved["threshold_source"] == "predefined_before_training decision_config.json", resolved

    with tempfile.TemporaryDirectory() as tmp:
        experiment_dir = Path(tmp)
        (experiment_dir / "best_model").mkdir()
        (experiment_dir / "train_config.json").write_text(
            json.dumps({"max_length": 384, "use_context": False}), encoding="utf-8"
        )
        (experiment_dir / "decision_config.json").write_text(
            json.dumps({
                "selected_on": "predefined_before_training",
                "classification_threshold": 0.5,
            }),
            encoding="utf-8",
        )
        try:
            resolve_experiment(experiment_dir)
            raise AssertionError("use_context=False 时应拒绝，因为这不是 prompt 实验")
        except ValueError:
            pass

    # _predict_batch_with_oom_retry：遇到 CUDA OOM 时应对半拆分 batch 重试，
    # 直到成功或拆到 min_batch_size 仍失败才把异常抛出去；不用真实模型/GPU，
    # 用一个"batch 超过阈值就抛 OOM，否则返回每条样本=1.0"的假前向函数验证
    # 拆分调度逻辑本身（是否真的拆到能成功的粒度、结果条数和顺序是否正确）。
    import torch

    def _fake_run_forward_batch(model, tokenizer, verbalizer_ids, batch, max_length, device):
        if len(batch) > model["max_ok_batch_size"]:
            model["oom_calls"] += 1
            raise torch.cuda.OutOfMemoryError("fake oom")
        model["success_calls"] += 1
        return [1.0] * len(batch)

    original_run_forward_batch = globals()["_run_forward_batch"]
    globals()["_run_forward_batch"] = _fake_run_forward_batch
    try:
        fake_model = {"max_ok_batch_size": 2, "oom_calls": 0, "success_calls": 0}
        batch = list(range(8))  # batch_size=8，超过 max_ok_batch_size=2，会不断对半拆
        result = _predict_batch_with_oom_retry(
            fake_model, None, None, batch, max_length=384, device="cuda"
        )
        assert result == [1.0] * 8, result
        assert fake_model["oom_calls"] > 0, "应该真的触发过 OOM 拆分，而不是一次就成功"
        assert fake_model["success_calls"] == 4, fake_model  # 8条最终拆成4组各2条成功

        # 拆到 min_batch_size 仍失败时应把异常抛出去，不吞掉。
        fake_model2 = {"max_ok_batch_size": 0, "oom_calls": 0, "success_calls": 0}
        try:
            _predict_batch_with_oom_retry(
                fake_model2, None, None, [1, 2], max_length=384, device="cuda"
            )
            raise AssertionError("永远 OOM 时应抛出 CUDA OutOfMemoryError")
        except torch.cuda.OutOfMemoryError:
            pass
    finally:
        globals()["_run_forward_batch"] = original_run_forward_batch

    # compute_metrics 直接复用 eval_on_real_distribution 的实现，这里只验证
    # 本文件的组装逻辑（导入、参数传递）没有写错，不重复测试其内部算法。
    labels = [1, 1] + [0] * 38
    probabilities = [0.9, 0.1] + [0.9] * 2 + [0.05] * 36
    metrics = compute_metrics(labels, probabilities, 0.85)
    assert metrics["positive_ratio"] == 0.05, metrics
    assert metrics["tp"] == 1 and metrics["fn"] == 1, metrics
    assert metrics["fp"] == 2 and metrics["tn"] == 36, metrics

    # write_review_csv：验证列内容、置信度换算（预测为0时取1-prob作为置信度）
    # 以及分歧标记是否正确。
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "review.csv"
        rows = [
            {"text": "评论1", "retweeted_content": "上下文1"},
            {"text": "评论2", "retweeted_content": ""},
        ]
        labels = [1, 0]
        positive_probs = [0.9, 0.7]  # 第2条预测为1(0.7>=0.5)但原标签0，构成分歧
        write_review_csv(rows, labels, positive_probs, 0.5, csv_path)

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            read_rows = list(csv.DictReader(f))
        assert len(read_rows) == 2, read_rows
        assert read_rows[0]["评论"] == "评论1", read_rows[0]
        assert read_rows[0]["上下文"] == "上下文1", read_rows[0]
        assert read_rows[0]["原始标记"] == "1", read_rows[0]
        assert read_rows[0]["模型标记"] == "1", read_rows[0]
        assert read_rows[0]["模型置信度"] == "0.9", read_rows[0]
        assert read_rows[0]["是否分歧"] == "", read_rows[0]
        assert read_rows[1]["原始标记"] == "0", read_rows[1]
        assert read_rows[1]["模型标记"] == "1", read_rows[1]
        assert read_rows[1]["模型置信度"] == "0.7", read_rows[1]
        assert read_rows[1]["是否分歧"] == "是", read_rows[1]

    print("自检通过")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

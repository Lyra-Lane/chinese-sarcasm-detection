"""构造反映真实标签分布的评估集，用于估计模型部署后的真实表现。

背景：train/val/test 目前都是 1:1 平衡集（balance.py 下采样负类后再切分），这是
为了防止训练时模型走"全判0"的捷径，是正确做法；但也导致 val/test 上算出的
precision/recall/f1 严重偏离真实分布下的表现——真实数据里正类（讽刺）占比远低于
1:1，同样的模型行为搬到真实分布上，precision 会大幅下降（负类基数放大，FP 绝对
数量跟着放大，但 1:1 测试集完全看不出这一点）。

本脚本不改动现有 train/val/test，另外构造一份"真实分布评估集"：
- 正类：直接复用 test.jsonl 里的正类样本（已被 split_data.py 划出、从未参与训练，
  拿来做评估没有信息泄漏问题）。之所以不能从 LABELED_FILE 里另找"训练集外"的
  正类样本，是因为 balance.py 会保留 LABELED_FILE 里全部的正类样本，全部正类
  最终都进了 train/val/test，不存在"训练集外"的正类可用。
- 负类：从 LABELED_FILE 里随机抽取"从未进入 train/val/test"的负类样本（用
  BALANCED_FILE 的 source_line 判断是否用过，与
  model_train/tools/scan_label_disagreements.py 的判断口径一致，直接复用其
  load_training_source_lines 函数），抽样数量按真实正类占比换算，让整份评估集
  的类别比例逼近真实分布。负类池通常有上百万条，用 Algorithm R 做流式等概率
  抽样，不需要把全量负类先载入内存。

真实正类占比默认从 LABELED_FILE 全量统计得到（--ratio 可手动覆盖，比如你对
真实线上分布有更准的先验估计，与当前已标注数据的比例不同）。

用法：
    python3 model_train/tools/build_real_distribution_eval.py
    python3 model_train/tools/build_real_distribution_eval.py --target-size 50000
    python3 model_train/tools/build_real_distribution_eval.py --ratio 0.013
    python3 model_train/tools/build_real_distribution_eval.py --selfcheck
"""

import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import LABELED_FILE, BALANCED_FILE, SPLIT_DIR

# 同目录复用 scan_label_disagreements.py 里"判断某条数据是否已参与
# 训练/验证/测试"的既有实现，两处口径必须一致，不重复造轮子。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_label_disagreements import load_training_source_lines

# ============================================================
# 【可调参数区】
# ============================================================
# 评估集目标总样本数（正类+负类）。真实正类占比通常很低，target_size 太小时
# 正类样本数可能只有个位数，指标噪声会很大；建议不低于 1万。
TARGET_SIZE = 20000

# 真实正类占比：None 表示从 LABELED_FILE 全量统计得到；也可手动指定（比如
# 已标注数据的比例和真实线上分布有偏差时，用更准的先验估计覆盖）。
REAL_RATIO = None

RANDOM_SEED = 42

OUTPUT_FILE = Path(__file__).resolve().parents[2] / "outputs" / "real_distribution_eval.jsonl"
SUMMARY_FILE = Path(__file__).resolve().parents[2] / "outputs" / "real_distribution_eval.summary.json"
# ============================================================


def normalize_text(value):
    """与 split_data.py / scan_label_disagreements.py 的 normalize_text 保持一致。"""
    return " ".join(str(value or "").split())


def iter_jsonl(path):
    """流式逐行读取 JSONL，返回 (line_no, row)，line_no 从 1 开始。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                continue


def read_jsonl_full(path):
    return [row for _, row in iter_jsonl(path)]


def compute_real_ratio(labeled_path):
    """统计 LABELED_FILE 全量的正类占比。返回 (ratio, pos_count, total_count)。"""
    pos = total = 0
    for _, row in iter_jsonl(labeled_path):
        label = row.get("is_sarcasm")
        if label not in (0, 1):
            continue
        total += 1
        if label == 1:
            pos += 1
    if total == 0:
        raise ValueError(f"{labeled_path} 里没有找到任何有效标签（is_sarcasm 为 0/1）的记录")
    return pos / total, pos, total


def reservoir_sample(candidates, k, rng):
    """Algorithm R：对可能很长的 candidates 迭代器做等概率抽样，返回最多 k 个元素。

    O(k) 内存，不需要把整个候选集先载入列表。若候选数少于 k，返回全部候选
    （此时抽样数达不到目标，由调用方决定是否警告）。
    """
    reservoir = []
    for i, item in enumerate(candidates):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def iter_eligible_negatives(labeled_path, used_source_lines):
    """流式产出 LABELED_FILE 中"标签为0、从未进入 train/val/test"的样本。"""
    for line_no, row in iter_jsonl(labeled_path):
        if row.get("is_sarcasm") != 0:
            continue
        source_line = row.get("source_line", line_no)
        if source_line in used_source_lines:
            continue
        text = normalize_text(row.get("content", ""))
        if not text:
            continue
        yield {
            "text": text,
            "retweeted_content": normalize_text(row.get("retweeted_content", "")),
            "label": 0,
            "source_line": source_line,
            "origin": "labeled_unused_negative",
        }


def build_eval_set(labeled_path, balanced_path, test_path, target_size, ratio, seed):
    """构造真实分布评估集，返回 (rows, summary)。

    balanced_path 用于判断样本是否已进入 train/val/test（负类池要排除掉这部分）。
    """
    if ratio is None:
        ratio, real_pos, real_total = compute_real_ratio(labeled_path)
    else:
        real_pos = real_total = None

    test_rows = read_jsonl_full(test_path)
    test_positives = [r for r in test_rows if r.get("label") == 1]
    if not test_positives:
        raise ValueError(f"{test_path} 里没有正类样本，无法构造评估集正类部分")

    pos_target = round(target_size * ratio)
    pos_target = max(1, min(pos_target, len(test_positives)))
    neg_target = target_size - pos_target

    rng = random.Random(seed)
    pos_sample = rng.sample(test_positives, pos_target)
    pos_sample = [
        {
            "text": normalize_text(r.get("text", "")),
            "retweeted_content": normalize_text(r.get("retweeted_content", "")),
            "label": 1,
            "source_line": None,  # test.jsonl 已丢失 source_line，不影响后续评估
            "origin": "test_positive",
        }
        for r in pos_sample
    ]

    used_source_lines = load_training_source_lines(balanced_path)
    neg_sample = reservoir_sample(
        iter_eligible_negatives(labeled_path, used_source_lines), neg_target, rng
    )

    rows = pos_sample + neg_sample
    rng.shuffle(rows)

    actual_total = len(rows)
    actual_pos = sum(1 for r in rows if r["label"] == 1)
    summary = {
        "target_size": target_size,
        "requested_ratio": ratio,
        "ratio_source": "manual" if real_pos is None else "computed_from_labeled_file",
        "labeled_file_stats": (
            {"positive_count": real_pos, "total_count": real_total}
            if real_pos is not None else None
        ),
        "test_positive_pool_size": len(test_positives),
        "pos_target": pos_target,
        "neg_target": neg_target,
        "neg_sampled": len(neg_sample),
        "neg_pool_insufficient": len(neg_sample) < neg_target,
        "actual_total": actual_total,
        "actual_positive_count": actual_pos,
        "actual_ratio": actual_pos / actual_total if actual_total else None,
        "seed": seed,
    }
    return rows, summary


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", default=str(LABELED_FILE), help="标注数据全量 JSONL")
    parser.add_argument("--balanced", default=str(BALANCED_FILE),
                        help="用于判断哪些样本已进入 train/val/test 的平衡数据 JSONL")
    parser.add_argument("--test-file", default=str(SPLIT_DIR / "test.jsonl"),
                        help="正类来源：split_data.py 产出的 test.jsonl")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE, help="评估集目标总样本数")
    parser.add_argument("--ratio", type=float, default=REAL_RATIO,
                        help="真实正类占比，默认从 --labeled 全量统计得到")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--summary-output", default=str(SUMMARY_FILE))
    args = parser.parse_args()

    rows, summary = build_eval_set(
        args.labeled, args.balanced, args.test_file, args.target_size, args.ratio, args.seed
    )

    if summary["neg_pool_insufficient"]:
        print(f"警告：可用负类样本只有 {summary['neg_sampled']} 条，"
              f"少于目标 {summary['neg_target']} 条，实际比例会偏高于要求值。")

    print(f"真实正类占比：{summary['requested_ratio']:.4%}（来源：{summary['ratio_source']}）")
    print(f"目标正类 {summary['pos_target']} 条（来自 test.jsonl 正类池 {summary['test_positive_pool_size']} 条）")
    print(f"目标负类 {summary['neg_target']} 条，实际抽到 {summary['neg_sampled']} 条")
    print(f"评估集总数：{summary['actual_total']}，实际正类占比：{summary['actual_ratio']:.4%}")

    write_jsonl(rows, Path(args.output))
    print(f"已写入：{args.output}")
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"摘要已写入：{args.summary_output}")


def _selfcheck():
    """不依赖真实数据文件，只验证抽样比例、去重排除、reservoir 抽样的正确性。"""
    rng = random.Random(0)

    # reservoir_sample：候选数远大于 k 时，抽样数应正好等于 k；候选数小于 k 时
    # 应返回全部候选（不应报错或抽出重复项）。
    big_pool = list(range(10000))
    sample_k = reservoir_sample(iter(big_pool), 100, random.Random(1))
    assert len(sample_k) == 100
    assert len(set(sample_k)) == 100  # 无重复
    assert all(x in big_pool for x in sample_k)

    small_pool = list(range(5))
    sample_small = reservoir_sample(iter(small_pool), 100, random.Random(1))
    assert sorted(sample_small) == small_pool

    # build_eval_set 的核心比例逻辑：伪造 labeled/test 数据验证 pos_target 换算正确。
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp:
        labeled_path = os.path.join(tmp, "labeled.jsonl")
        balanced_path = os.path.join(tmp, "balanced.jsonl")
        test_path = os.path.join(tmp, "test.jsonl")

        # 100 条数据：10 条正类（source_line 1-10，全部"已用于训练"），
        # 90 条负类，其中只有 source_line 20-99 从未用过（80 条可用负类池）。
        with open(labeled_path, "w", encoding="utf-8") as f:
            for i in range(1, 11):
                f.write(json.dumps({"content": f"pos{i}", "retweeted_content": "",
                                    "is_sarcasm": 1, "source_line": i}) + "\n")
            for i in range(11, 101):
                f.write(json.dumps({"content": f"neg{i}", "retweeted_content": "",
                                    "is_sarcasm": 0, "source_line": i}) + "\n")

        with open(balanced_path, "w", encoding="utf-8") as f:
            for i in range(1, 11):
                f.write(json.dumps({"source_line": i}) + "\n")
            for i in range(11, 20):  # 前 9 条负类已用于训练，20-100 未用
                f.write(json.dumps({"source_line": i}) + "\n")

        with open(test_path, "w", encoding="utf-8") as f:
            for i in range(1, 6):  # test.jsonl 里 5 条正类可供抽样
                f.write(json.dumps({"text": f"pos{i}", "retweeted_content": "", "label": 1}) + "\n")
            for i in range(1, 6):
                f.write(json.dumps({"text": f"testneg{i}", "retweeted_content": "", "label": 0}) + "\n")

        rows, summary = build_eval_set(
            labeled_path, balanced_path, test_path, target_size=50, ratio=0.1, seed=0
        )
        # pos_target = round(50*0.1) = 5，正好等于 test 正类池大小
        assert summary["pos_target"] == 5, summary
        assert summary["neg_target"] == 45, summary
        # 可用负类池是 source_line 20-100 共 81 条，够抽 45 条
        assert summary["neg_sampled"] == 45, summary
        assert summary["actual_total"] == 50, summary
        # 负类样本不应包含 source_line < 20（已用于训练的部分）
        neg_lines = {r["source_line"] for r in rows if r["label"] == 0}
        assert all(sl is None or sl >= 20 for sl in neg_lines), neg_lines
        assert len([r for r in rows if r["label"] == 1]) == 5

    print("自检通过")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

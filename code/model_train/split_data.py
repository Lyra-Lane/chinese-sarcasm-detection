"""从完整标注数据生成 train=1:1、test=1:1 的互斥数据集。"""

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import LABELED_FILE, SPLIT_DIR


INPUT_FILE = LABELED_FILE
OUTPUT_DIR = SPLIT_DIR
TRAIN_RATIO, TEST_RATIO = 8, 2
TRAIN_NEGATIVE_PER_POSITIVE = 1
TEST_NEGATIVE_PER_POSITIVE = 1
RANDOM_SEED = 42


def normalize_text(value):
    return " ".join(str(value or "").split())


def iter_data(path):
    """流式校验并标准化完整标注数据，避免一次载入百万级负样本。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "content" not in obj or "is_sarcasm" not in obj:
                raise KeyError(f"第 {line_no} 行缺少 content 或 is_sarcasm")
            text = normalize_text(obj["content"])
            context = normalize_text(obj.get("retweeted_content", ""))
            label = obj["is_sarcasm"]
            if not text:
                continue
            if label not in (0, 1):
                raise ValueError(f"第 {line_no} 行标签必须是 0/1，实际为 {label!r}")
            label = int(label)
            yield {
                "text": text,
                "retweeted_content": context,
                "label": label,
                "source_line": obj.get("source_line", line_no),
            }


def required_negative_count(positive_count):
    ratio_sum = TRAIN_RATIO + TEST_RATIO
    train_positive = int(positive_count * TRAIN_RATIO / ratio_sum)
    test_positive = positive_count - train_positive
    return (
        train_positive * TRAIN_NEGATIVE_PER_POSITIVE
        + test_positive * TEST_NEGATIVE_PER_POSITIVE
    )


def reservoir_sample(rows, sample_size, rng):
    """Algorithm R 等概率流式抽样，只保留实际需要的负样本。"""
    sample = []
    for index, row in enumerate(rows):
        if index < sample_size:
            sample.append(row)
            continue
        replacement = rng.randint(0, index)
        if replacement < sample_size:
            sample[replacement] = row
    return sample


def load_split_candidates(path):
    """两遍读取：保留全部正例，再从全量负例中等概率抽取所需数量。"""
    positives = []
    negative_count = 0
    for row in iter_data(path):
        if row["label"] == 1:
            positives.append(row)
        else:
            negative_count += 1

    needed = required_negative_count(len(positives))
    if negative_count < needed:
        raise ValueError(
            f"负样本不足：按 train=1:1、test=1:1 需要 {needed} 条，"
            f"实际只有 {negative_count} 条"
        )
    negatives = reservoir_sample(
        (row for row in iter_data(path) if row["label"] == 0),
        needed,
        random.Random(RANDOM_SEED + 1),
    )
    return positives + negatives, {
        "positive": len(positives),
        "negative": negative_count,
        "valid_total": len(positives) + negative_count,
        "sampled_negative": len(negatives),
    }


def split_data(rows):
    """正类按 8:2 切分，再为两份数据分配互不重叠的目标数量负类。"""
    rng = random.Random(RANDOM_SEED)
    positives = [row for row in rows if row["label"] == 1]
    negatives = [row for row in rows if row["label"] == 0]
    if not positives or not negatives:
        raise ValueError("完整标注数据必须同时包含标签 0 和标签 1")

    rng.shuffle(positives)
    rng.shuffle(negatives)
    ratio_sum = TRAIN_RATIO + TEST_RATIO
    n_train_pos = int(len(positives) * TRAIN_RATIO / ratio_sum)

    train_pos = positives[:n_train_pos]
    test_pos = positives[n_train_pos:]

    negative_counts = {
        "train": len(train_pos) * TRAIN_NEGATIVE_PER_POSITIVE,
        "test": len(test_pos) * TEST_NEGATIVE_PER_POSITIVE,
    }
    required_negatives = sum(negative_counts.values())
    if len(negatives) < required_negatives:
        raise ValueError(
            f"负样本不足：按 train=1:1、test=1:1 需要 {required_negatives} 条，"
            f"实际只有 {len(negatives)} 条"
        )

    cursor = 0
    train_neg = negatives[cursor:cursor + negative_counts["train"]]
    cursor += negative_counts["train"]
    test_neg = negatives[cursor:cursor + negative_counts["test"]]

    train = train_pos + train_neg
    test = test_pos + test_neg

    for part in (train, test):
        rng.shuffle(part)
    validate_splits(train, test)
    return train, test


def _sample_key(row):
    return row["text"], row.get("retweeted_content", "")


def validate_splits(train, test):
    """拒绝比例错误或样本交叉的切分结果。"""
    expected = {
        "train": TRAIN_NEGATIVE_PER_POSITIVE,
        "test": TEST_NEGATIVE_PER_POSITIVE,
    }
    parts = {"train": train, "test": test}
    for name, part in parts.items():
        counts = Counter(row["label"] for row in part)
        if not counts[1] or counts[0] != counts[1] * expected[name]:
            raise ValueError(
                f"{name} 标签比例错误：期望 1:{expected[name]}，实际 {dict(counts)}"
            )
        part_keys = [_sample_key(row) for row in part]
        if len(set(part_keys)) != len(part_keys):
            raise ValueError(f"{name} 内存在重复的评论+上下文，请先清理标注数据")

    keys = {name: {_sample_key(row) for row in part} for name, part in parts.items()}
    if keys["train"] & keys["test"]:
        raise ValueError("train/test 存在重复的评论+上下文，拒绝写出")


def write_jsonl(rows, path):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_summary(rows, path):
    counts = Counter(row["label"] for row in rows)
    return {
        "file": str(path),
        "sha256": file_sha256(path),
        "samples": len(rows),
        "positive": counts[1],
        "negative": counts[0],
        "negative_per_positive": counts[0] / counts[1],
    }


def main():
    rows, source_counts = load_split_candidates(INPUT_FILE)
    train, test = split_data(rows)
    if not train or not test:
        raise ValueError("数据太少，无法产生非空的训练/测试集")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for name, part in (("train", train), ("test", test)):
        output_path = OUTPUT_DIR / f"{name}.jsonl"
        write_jsonl(part, output_path)
        output_paths[name] = output_path
        print(f"{name}: {len(part)}，标签分布 {dict(Counter(r['label'] for r in part))}")
    stale_val_path = OUTPUT_DIR / "val.jsonl"
    if stale_val_path.exists():
        stale_val_path.unlink()
        print(f"已移除旧验证集：{stale_val_path}")

    manifest = {
        "input_file": str(INPUT_FILE),
        "input_sha256": file_sha256(INPUT_FILE),
        "source_counts": source_counts,
        "seed": RANDOM_SEED,
        "positive_split_ratio": f"{TRAIN_RATIO}:{TEST_RATIO}",
        "splits": {
            "train": split_summary(train, output_paths["train"]),
            "test": split_summary(test, output_paths["test"]),
        },
    }
    manifest_path = OUTPUT_DIR / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"完整标注数据有效样本数：{source_counts['valid_total']}")
    print(f"参与本次划分的样本数：{len(rows)}")
    print(f"划分清单：{manifest_path}")


def _selfcheck():
    import tempfile

    sampled = reservoir_sample(iter(range(1000)), 100, random.Random(1))
    assert len(sampled) == 100 and len(set(sampled)) == 100
    assert required_negative_count(100) == 100
    fake = (
        [
            {"text": f"p{i}", "retweeted_content": "", "label": 1,
             "source_line": i + 1}
            for i in range(100)
        ]
        + [
            {"text": f"n{i}", "retweeted_content": "", "label": 0,
             "source_line": i + 1001}
            for i in range(2500)
        ]
    )
    train, test = split_data(fake)
    assert Counter(r["label"] for r in train) == {0: 80, 1: 80}
    assert Counter(r["label"] for r in test) == {0: 20, 1: 20}
    train_lines = {r["source_line"] for r in train}
    test_lines = {r["source_line"] for r in test}
    assert train_lines.isdisjoint(test_lines)

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "labeled.jsonl"
        with input_path.open("w", encoding="utf-8") as f:
            for row in fake:
                f.write(json.dumps({
                    "content": row["text"],
                    "retweeted_content": row["retweeted_content"],
                    "is_sarcasm": row["label"],
                    "source_line": row["source_line"],
                }, ensure_ascii=False) + "\n")
        candidates, counts = load_split_candidates(input_path)
        assert counts == {
            "positive": 100,
            "negative": 2500,
            "valid_total": 2600,
            "sampled_negative": 100,
        }
        assert Counter(row["label"] for row in candidates) == {0: 100, 1: 100}
    print("自检通过")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

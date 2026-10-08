"""从同一轮的多组参数结果里，按业务规则选出本轮最优模型。

规则（用户确定）：先按 sarcasm_recall >= 0.80 过滤，再在满足条件的候选里选
sarcasm_precision 最高的一组；全部候选都不满足 recall 门槛时直接报错停止，
不做静默降级（比如自动放宽阈值选一个"凑合"的）。

指标来源固定为 eval_on_real_distribution_prompt.py 产出的
test_deployment_metrics.json（真实比例评估集上的指标，而不是 1:1 test.jsonl
上的指标——1:1 上的 precision 不能反映真实部署效果，这正是本条流水线要解决
的问题）。

用法：
    python3 model_train/tools/select_best_of_group.py \\
        --experiment-dir outputs/.../exp_a --experiment-dir outputs/.../exp_b \\
        --experiment-dir outputs/.../exp_c --experiment-dir outputs/.../exp_d

    python3 model_train/tools/select_best_of_group.py --selfcheck
"""

import argparse
import json
import sys
from pathlib import Path


RECALL_THRESHOLD = 0.80
METRICS_FILENAME = "test_deployment_metrics.json"


def load_candidate(experiment_dir, metrics_filename=METRICS_FILENAME):
    path = Path(experiment_dir) / metrics_filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在，请先对 {experiment_dir} 运行 "
            "eval_on_real_distribution_prompt.py 产出真实比例评估指标"
        )
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    for key in ("sarcasm_recall", "sarcasm_precision"):
        if key not in metrics:
            raise KeyError(f"{path} 缺少字段 {key!r}")
    return {
        "experiment_dir": str(experiment_dir),
        "sarcasm_recall": float(metrics["sarcasm_recall"]),
        "sarcasm_precision": float(metrics["sarcasm_precision"]),
        "sarcasm_f1": metrics.get("sarcasm_f1"),
        "metrics_path": str(path),
    }


def select_best(candidates, recall_threshold=RECALL_THRESHOLD):
    """candidates: [{"experiment_dir", "sarcasm_recall", "sarcasm_precision", ...}, ...]。

    过滤 recall>=threshold，按 precision 降序取第一个；空候选或全不达标时报错。
    """
    if not candidates:
        raise ValueError("候选列表为空")
    eligible = [c for c in candidates if c["sarcasm_recall"] >= recall_threshold]
    if not eligible:
        details = "; ".join(
            f"{c['experiment_dir']}: recall={c['sarcasm_recall']:.4%}" for c in candidates
        )
        raise RuntimeError(
            f"本轮 {len(candidates)} 组参数里没有任何一组 sarcasm_recall >= "
            f"{recall_threshold:.0%}，无法选出最优模型。各组结果：{details}"
        )
    return max(eligible, key=lambda c: c["sarcasm_precision"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", dest="experiment_dirs", action="append",
                        required=True, help="可重复传入，本轮各组参数的实验目录")
    parser.add_argument("--recall-threshold", type=float, default=RECALL_THRESHOLD)
    parser.add_argument("--metrics-filename", default=METRICS_FILENAME)
    parser.add_argument("--output", help="选中结果写入的 JSON 路径；不传则只打印")
    args = parser.parse_args()

    candidates = [
        load_candidate(d, args.metrics_filename) for d in args.experiment_dirs
    ]
    for c in candidates:
        print(f"{c['experiment_dir']}: recall={c['sarcasm_recall']:.4%} "
              f"precision={c['sarcasm_precision']:.4%} f1={c['sarcasm_f1']}")

    best = select_best(candidates, args.recall_threshold)
    print(f"\n本轮最优：{best['experiment_dir']} "
          f"(recall={best['sarcasm_recall']:.4%}, precision={best['sarcasm_precision']:.4%})")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"candidates": candidates, "best": best}, f, ensure_ascii=False, indent=2)
        print(f"已写入：{output_path}")


def _selfcheck():
    candidates = [
        {"experiment_dir": "a", "sarcasm_recall": 0.75, "sarcasm_precision": 0.90},
        {"experiment_dir": "b", "sarcasm_recall": 0.82, "sarcasm_precision": 0.70},
        {"experiment_dir": "c", "sarcasm_recall": 0.85, "sarcasm_precision": 0.75},
        {"experiment_dir": "d", "sarcasm_recall": 0.60, "sarcasm_precision": 0.95},
    ]
    # a、d 因 recall 不达标被排除；b、c 里 precision 更高的是 c。
    best = select_best(candidates)
    assert best["experiment_dir"] == "c", best

    # 全部不达标时应报错，而不是放宽阈值兜底。
    try:
        select_best([{"experiment_dir": "x", "sarcasm_recall": 0.5, "sarcasm_precision": 0.99}])
        raise AssertionError("全部候选都不满足 recall 门槛时应抛出 RuntimeError")
    except RuntimeError:
        pass

    try:
        select_best([])
        raise AssertionError("空候选列表应抛出 ValueError")
    except ValueError:
        pass

    # load_candidate：正常读取、缺字段报错、文件不存在报错。
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = Path(tmp) / "exp_a"
        exp_dir.mkdir()
        (exp_dir / METRICS_FILENAME).write_text(
            json.dumps({"sarcasm_recall": 0.83, "sarcasm_precision": 0.77, "sarcasm_f1": 0.80}),
            encoding="utf-8",
        )
        loaded = load_candidate(exp_dir)
        assert loaded["sarcasm_recall"] == 0.83, loaded
        assert loaded["sarcasm_precision"] == 0.77, loaded

        missing_dir = Path(tmp) / "exp_missing"
        missing_dir.mkdir()
        try:
            load_candidate(missing_dir)
            raise AssertionError("缺少 metrics 文件时应抛出 FileNotFoundError")
        except FileNotFoundError:
            pass

        bad_dir = Path(tmp) / "exp_bad"
        bad_dir.mkdir()
        (bad_dir / METRICS_FILENAME).write_text(
            json.dumps({"sarcasm_recall": 0.83}), encoding="utf-8",
        )
        try:
            load_candidate(bad_dir)
            raise AssertionError("缺少 sarcasm_precision 字段时应抛出 KeyError")
        except KeyError:
            pass

    print("自检通过")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

#!/usr/bin/env python3
"""迭代式主动学习训练总调度：挖负例 -> 4组参数顺序训练 -> 按真实分布指标选最优 -> 下一轮。

单轮流程：
  1. 第1轮：train_file = 原始 SPLIT_DIR/train.jsonl，起点模型 = BASE_MODEL_PATH。
     第2轮起：先用上一轮胜出模型调 mine_hard_negatives.py 挖出新一批标0负例，
     拼出本轮 train_iterN.jsonl（正例不变，负例整体替换）。
  2. 固定好本轮 train_file 后，顺序跑该轮对应的 4 组参数（同一张 GPU，不并行，
     避免显存互相挤占），每组独立一个实验目录。参数网格按轮次配置，见
     PARAM_GRID_BY_ITERATION（iter1 / iter2 各一套，iter3+ 默认沿用 iter2 的）。
  3. 每组训练完，用固定不变的 REAL_DIST_EVAL_FILE（只需生成一次，不随轮次
     重新生成）跑 eval_on_real_distribution_prompt.py，得到真实比例下的
     precision/recall。
  4. select_best_of_group.py 在 4 组里选出 recall>=80% 中 precision 最高的一组
     作为本轮最优模型；4 组都不达标则报错终止整条流水线（不静默降级）。
  5. 把本轮结果（各组指标 + 胜出者）记入 ITERATIVE_ROOT/pipeline_history.json，
     进入下一轮。

本文件只负责拼子进程命令调用已有脚本，不重新实现训练/挖掘/选优逻辑，写法与
项目里现有的 run_pipeline.py 风格一致。

用法：
    python3 model_train/run_iterative_pipeline.py --max-iterations 1   # 先跑通单轮
    python3 model_train/run_iterative_pipeline.py --max-iterations 3
    python3 model_train/run_iterative_pipeline.py --dry-run            # 只打印将执行的命令
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from pipeline_config import (
    BASE_MODEL_PATH,
    ITERATIVE_ROOT,
    ITERATIVE_STATE_FILE,
    LABELED_FILE,
    REAL_DIST_EVAL_FILE,
    SPLIT_DIR,
)

ORIGINAL_TRAIN_FILE = SPLIT_DIR / "train.jsonl"
ORIGINAL_TEST_FILE = SPLIT_DIR / "test.jsonl"
HISTORY_FILE = ITERATIVE_ROOT / "pipeline_history.json"

# 每轮各跑 4 组参数，按轮次分别配置（用户确定）：
#   iter1 在原始 train.jsonl 上从 base 底座起训，epoch 少一些（1.5）、alpha 偏中性；
#   iter2 起在上一轮权重上继续 fine-tune、训练集负例换成挖出来的难负例混合，
#   需要更高 alpha 去补偿难负例把决策边界推向"少判讽刺"的效应（iter2 早期
#   尝试低 alpha + 全难负例时 recall 从 84% 掉到 53~56%），epoch 也放宽到 2.5。
# 未显式配置的轮次（iter3 及以后）沿用 DEFAULT_PARAM_GRID。
PARAM_GRID_BY_ITERATION = {
    1: [
        {"learning_rate": 1e-5, "focal_alpha": 0.50, "num_epochs": 1.5},
        {"learning_rate": 1e-5, "focal_alpha": 0.40, "num_epochs": 1.5},
        {"learning_rate": 5e-6, "focal_alpha": 0.45, "num_epochs": 1.5},
        {"learning_rate": 1.5e-5, "focal_alpha": 0.45, "num_epochs": 1.5},
    ],
    2: [
        {"learning_rate": 1e-5, "focal_alpha": 0.5, "num_epochs": 3},
        {"learning_rate": 5e-6, "focal_alpha": 0.6, "num_epochs": 3},
        {"learning_rate": 5e-6, "focal_alpha": 0.55, "num_epochs": 3},
        {"learning_rate": 5e-6, "focal_alpha": 0.50, "num_epochs": 3},
        {"learning_rate": 5e-6, "focal_alpha": 0.45, "num_epochs": 3},
    ],
}
# iter3 及以后与 iter2 同属"在上一轮权重上继续 fine-tune + 难负例训练集"的场景，
# 所以默认沿用 iter2 的网格；需要针对更后面的轮次单独调参时，直接往
# PARAM_GRID_BY_ITERATION 里加对应轮次的条目即可。
DEFAULT_PARAM_GRID = PARAM_GRID_BY_ITERATION[2]


def param_grid_for_iteration(iteration):
    return PARAM_GRID_BY_ITERATION.get(iteration, DEFAULT_PARAM_GRID)

# GPU 训练步骤需要 conda(hbert) 环境（与 run_pipeline.py 的约定一致）；
# 挖掘/评估步骤也要加载模型，同样需要这个环境。
DEFAULT_MODEL_PYTHON_BIN = sys.executable


def run(command, dry_run, env=None):
    print(f"$ {' '.join(str(c) for c in command)}", flush=True)
    if not dry_run:
        run_env = {**os.environ, **env} if env else None
        subprocess.run(command, cwd=ROOT.parent, check=True, env=run_env)


def read_history(history_file):
    history_file = Path(history_file)
    if history_file.exists():
        with history_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"iterations": []}


def write_history(history_file, history):
    history_file = Path(history_file)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with history_file.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def find_latest_experiment_dir(output_dir):
    """train_cls_prompt.py 训练成功后会更新 output_dir/latest.txt。"""
    latest_path = Path(output_dir) / "latest.txt"
    if not latest_path.exists():
        raise RuntimeError(f"{latest_path} 不存在，说明本组训练未成功产出实验目录")
    experiment_id = latest_path.read_text(encoding="utf-8").strip()
    exp_dir = Path(output_dir) / "experiments" / experiment_id
    if not exp_dir.exists():
        raise RuntimeError(f"latest.txt 指向的实验目录不存在：{exp_dir}")
    return exp_dir


def group_output_dir(iteration, group_index):
    return ITERATIVE_ROOT / f"iter{iteration}_group{group_index}"


def mine_negatives_for_iteration(python_bin, iteration, previous_experiment_dir, dry_run,
                                 batch_size=None, env=None, no_dedupe_positives=False,
                                 hard_negative_ratio=None, no_early_stop=False,
                                 buffer_multiplier=None):
    """previous_experiment_dir 必须是实验目录本身（含 train_config.json/
    decision_config.json/best_model/），不是 best_model 权重目录——
    mine_hard_negatives.py 靠 resolve_experiment 读前两者确定推理输入方式和阈值。

    默认走"边标边停"（mine_hard_negatives.py 的 early_stop 默认 True），不用
    对候选池（可能上百万条）做全量推理，难负例凑够缓冲池就停止扫描。
    """
    output_train_file = ITERATIVE_ROOT / f"train_iter{iteration}.jsonl"
    command = [
        python_bin, str(ROOT / "tools" / "mine_hard_negatives.py"),
        "--experiment-dir", str(previous_experiment_dir),
        "--output-train-file", str(output_train_file),
        "--iteration", str(iteration),
        "--train-file", str(ORIGINAL_TRAIN_FILE),
        "--test-file", str(ORIGINAL_TEST_FILE),
        "--labeled-file", str(LABELED_FILE),
        "--real-dist-eval-file", str(REAL_DIST_EVAL_FILE),
        "--state-file", str(ITERATIVE_STATE_FILE),
    ]
    if batch_size is not None:
        command += ["--batch-size", str(batch_size)]
    if no_dedupe_positives:
        command += ["--no-dedupe-positives"]
    if hard_negative_ratio is not None:
        command += ["--hard-negative-ratio", str(hard_negative_ratio)]
    if no_early_stop:
        command += ["--no-early-stop"]
    if buffer_multiplier is not None:
        command += ["--buffer-multiplier", str(buffer_multiplier)]
    run(command, dry_run, env=env)
    return output_train_file


def train_one_group(python_bin, iteration, group_index, params, model_path, train_file, dry_run,
                    env=None):
    output_dir = group_output_dir(iteration, group_index)
    command = [
        python_bin, str(ROOT / "train_cls_prompt_iterative.py"),
        "--model-path", str(model_path),
        "--train-file", str(train_file),
        "--output-dir", str(output_dir),
        "--learning-rate", str(params["learning_rate"]),
        "--focal-alpha", str(params["focal_alpha"]),
        "--num-epochs", str(params["num_epochs"]),
    ]
    run(command, dry_run, env=env)
    if dry_run:
        return output_dir, None
    return output_dir, find_latest_experiment_dir(output_dir)


def evaluate_group(python_bin, experiment_dir, dry_run, env=None):
    command = [
        python_bin, str(ROOT / "tools" / "eval_on_real_distribution_prompt.py"),
        "--experiment-dir", str(experiment_dir),
        "--eval-file", str(REAL_DIST_EVAL_FILE),
    ]
    run(command, dry_run, env=env)


def select_best(python_bin, experiment_dirs, iteration, dry_run, recall_threshold=None):
    output_path = ITERATIVE_ROOT / f"iter{iteration}_selection.json"
    command = [python_bin, str(ROOT / "tools" / "select_best_of_group.py")]
    for d in experiment_dirs:
        command += ["--experiment-dir", str(d)]
    command += ["--output", str(output_path)]
    if recall_threshold is not None:
        command += ["--recall-threshold", str(recall_threshold)]
    run(command, dry_run)
    if dry_run:
        return None
    with output_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_iteration(python_bin, iteration, previous_experiment_dir, previous_model_path, dry_run,
                  mine_batch_size=None, env=None, no_dedupe_positives=False,
                  hard_negative_ratio=None, recall_threshold=None, no_early_stop=False,
                  buffer_multiplier=None):
    if iteration == 1:
        train_file = ORIGINAL_TRAIN_FILE
        start_model_path = BASE_MODEL_PATH
    else:
        train_file = mine_negatives_for_iteration(
            python_bin, iteration, previous_experiment_dir, dry_run,
            batch_size=mine_batch_size, env=env, no_dedupe_positives=no_dedupe_positives,
            hard_negative_ratio=hard_negative_ratio, no_early_stop=no_early_stop,
            buffer_multiplier=buffer_multiplier,
        )
        # 继续 fine-tune 用的是上一轮胜出实验的 best_model 权重目录，
        # 挖掘用的是实验目录本身（见 mine_negatives_for_iteration 的说明）。
        start_model_path = previous_model_path

    group_results = []
    for group_index, params in enumerate(param_grid_for_iteration(iteration), start=1):
        print(f"\n=== 第{iteration}轮 / 第{group_index}组参数：{params} ===", flush=True)
        output_dir, exp_dir = train_one_group(
            python_bin, iteration, group_index, params, start_model_path, train_file, dry_run,
            env=env,
        )
        if not dry_run:
            evaluate_group(python_bin, exp_dir, dry_run, env=env)
        group_results.append({
            "group_index": group_index,
            "params": params,
            "output_dir": str(output_dir),
            "experiment_dir": str(exp_dir) if exp_dir else None,
        })

    if dry_run:
        return {
            "iteration": iteration,
            "train_file": str(train_file),
            "start_model_path": str(start_model_path),
            "groups": group_results,
            "selection": None,
        }

    selection = select_best(
        python_bin, [g["experiment_dir"] for g in group_results], iteration, dry_run,
        recall_threshold=recall_threshold,
    )
    best_experiment_dir = selection["best"]["experiment_dir"]
    best_model_path = str(Path(best_experiment_dir) / "best_model")

    return {
        "iteration": iteration,
        "train_file": str(train_file),
        "start_model_path": str(start_model_path),
        "groups": group_results,
        "selection": selection,
        "winner_experiment_dir": best_experiment_dir,
        "winner_model_path": best_model_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iterations", type=int, default=1,
                        help="本次调用最多跑几轮（累加在历史已完成轮次之后）")
    parser.add_argument("--python-bin", default=DEFAULT_MODEL_PYTHON_BIN,
                        help="训练/推理步骤使用的 python3 路径（需装 torch/transformers）")
    parser.add_argument("--cuda-device", default=None,
                        help="覆盖各子进程使用的 CUDA_VISIBLE_DEVICES（如共享多卡机器上卡0被"
                             "占满时可传 --cuda-device 1 切到别的卡）；不传则用各脚本默认的卡0")
    parser.add_argument("--mine-batch-size", type=int, default=None,
                        help="挖掘阶段（mine_hard_negatives.py）批量推理的 batch size；"
                             "共享GPU显存紧张时可调小（默认64），避免 OOM")
    parser.add_argument("--no-dedupe-positives", action="store_true",
                        help="挖掘阶段不对正例去重，原样使用 train.jsonl 里的全部 label=1 行"
                             "（默认从第2轮起会按 文本+上下文 去重，避免"
                             " augment_train_positives.py 复制过的正例被持续放大）")
    parser.add_argument("--hard-negative-ratio", type=float, default=None,
                        help="挖掘阶段负例中难负例（模型误判为1）的占比[0,1]；不传则用"
                             "mine_hard_negatives.py 的默认值1.0（全部为难负例）。传小于1"
                             "的值（如0.7）会混入普通负例，增加多样性，缓解全负例都是"
                             "边界样本导致模型过度保守、recall崩塌的问题（见工作日志 iter2 记录）")
    parser.add_argument("--recall-threshold", type=float, default=None,
                        help="选优阶段（select_best_of_group.py）的 recall 门槛；不传则用"
                             "select_best_of_group.py 的默认值0.80")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="挖掘阶段关闭“边标边停”，改为对候选池全量推理"
                             "（候选池上百万条时会很慢，默认不建议开启，调试/对照实验时可用）")
    parser.add_argument("--buffer-multiplier", type=float, default=None,
                        help="边标边停时，难负例缓冲池目标大小的倍数；不传则用"
                             "mine_hard_negatives.py 的默认值3.0")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令，不实际运行")
    args = parser.parse_args()

    env = {"CUDA_VISIBLE_DEVICES": args.cuda_device} if args.cuda_device else None

    history = read_history(HISTORY_FILE)
    completed = len(history["iterations"])
    previous_experiment_dir = (
        history["iterations"][-1]["winner_experiment_dir"] if completed else None
    )
    previous_model_path = (
        history["iterations"][-1]["winner_model_path"] if completed else None
    )

    for offset in range(args.max_iterations):
        iteration = completed + offset + 1
        print(f"\n########## 第 {iteration} 轮 ##########", flush=True)
        result = run_iteration(
            args.python_bin, iteration, previous_experiment_dir, previous_model_path, args.dry_run,
            mine_batch_size=args.mine_batch_size, env=env,
            no_dedupe_positives=args.no_dedupe_positives,
            hard_negative_ratio=args.hard_negative_ratio,
            recall_threshold=args.recall_threshold,
            no_early_stop=args.no_early_stop,
            buffer_multiplier=args.buffer_multiplier,
        )
        if args.dry_run:
            continue
        history["iterations"].append(result)
        write_history(HISTORY_FILE, history)
        previous_experiment_dir = result["winner_experiment_dir"]
        previous_model_path = result["winner_model_path"]
        print(f"第 {iteration} 轮完成，胜出模型：{previous_model_path}")

    if not args.dry_run:
        print(f"\n流水线历史已写入：{HISTORY_FILE}")


if __name__ == "__main__":
    main()

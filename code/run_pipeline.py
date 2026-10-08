#!/usr/bin/env python3
"""从 workwork 根目录串行运行整套反讽识别实验。"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = {
    "collect": ("data", "data_label/mqread78.py"),
    "label": ("local", "data_label/label_sarcasm.py"),
    "build_pending": ("local", "data_label/build_pending.py"),
    "thr_label": ("local", "data_label/thr_label.py"),
    "split": ("model", "model_train/split_data.py"),
    "train": ("model", "model_train/train_cls.py"),
    "scan_disagreements": ("model", "model_train/tools/scan_label_disagreements.py"),
}
# 采集/标注/切分/train(train_cls.py，仅作参照基线) 这条链路数据已构建完毕，
# 不再是日常要重跑的流水线；主力训练已转为 model_train/run_iterative_pipeline.py
# （基于 train_cls_prompt.py 的迭代式主动学习流水线，见 pipeline_config.py 顶部说明），
# 不接入本文件。
# scan_disagreements（用已训练模型反查标注错误）也不纳入 all：它依赖已有
# best_model，且不是每次跑数据流水线都要做，需要时单独执行。
# dedup/balance 已从 all 中移除：split_data.py 直接读取 LABELED_FILE，并一次生成
# train=1:1、val/test=1:70 的互斥数据。本发布版不提供缺失的旧步骤。
# build_pending 必须排在 thr_label 之前：thr_label 现在默认读 PENDING_FILE，
# 少了这一步会直接报"输入文件不存在"。
# 注意 clean_labeled.py 不在这里——它是针对历史数据的一次性补齐，只需跑一次。
ALL_STEPS = ["collect", "build_pending", "thr_label", "split", "train"]


# 需要 torch/transformers 的步骤：docker exec 默认非登录 shell 不会激活 conda
# 环境（手动 `docker exec -it ... bash` 才会通过 .bashrc 自动激活 hbert 环境），
# 必须显式指定这个环境下的 python3 绝对路径，否则会 fall back 到容器系统 python3
# （没装 transformers），报 ModuleNotFoundError。
GPU_STEPS = {"train", "scan_disagreements"}


def command_for(step, args):
    container, script = STEPS[step]
    extra = ["--resume"] if step in {"label", "thr_label"} and args.resume_label else []
    if step == "scan_disagreements":
        extra += ["--threshold", str(args.scan_threshold), "--direction", args.scan_direction]
        if args.scan_experiment_dir:
            extra += ["--experiment-dir", args.scan_experiment_dir]

    python_bin = args.model_python_bin if step in GPU_STEPS else "python3"

    if container == "local" or args.runtime == "local":
        return [python_bin if step in GPU_STEPS else sys.executable, str(ROOT / script), *extra]
    container_name = args.data_container if container == "data" else args.model_container
    command = ["docker", "exec", "-i", "-w", args.container_dir]
    return command + [container_name, python_bin, script, *extra]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "steps", nargs="+", choices=[*STEPS, "all"],
        help="运行阶段；all=采集、生成待标注文件、三模型投票标注、切分、仅评论训练"
             "（迭代式主动学习训练不在这里，见 model_train/run_iterative_pipeline.py）",
    )
    parser.add_argument("--runtime", choices=["docker", "local"], default="docker")
    parser.add_argument("--data-container", default="data")
    parser.add_argument("--model-container", default="model")
    parser.add_argument("--container-dir", default="/workspace/code")
    parser.add_argument("--resume-label", action="store_true", help="label 或 thr_label 阶段断点续跑")
    parser.add_argument("--scan-threshold", type=float, default=0.90,
                        help="scan_disagreements 阶段的置信度阈值")
    parser.add_argument("--scan-direction", choices=["0to1", "1to0", "both"], default="both",
                        help="scan_disagreements 阶段的误判方向："
                             "both=标注不一致即收（默认）；0to1/1to0=只看单一方向")
    parser.add_argument("--scan-experiment-dir",
                        help="scan_disagreements 使用的新实验目录，自动读取模型及业务阈值")
    parser.add_argument(
        "--model-python-bin",
        default="python3",
        help="train/scan_disagreements 步骤使用的 python3 路径"
             "（需装 torch/transformers 的 conda(hbert) 环境，docker exec 默认不会自动激活）。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    args = parser.parse_args()

    # "all" 展开为 ALL_STEPS，同时保留其余显式指定的步骤（如 train_context），
    # 而不是整体替换掉 —— 这样 "all train_context" 才能等于
    # ALL_STEPS + train_context，而不是丢掉 train_context。
    requested = []
    for step in args.steps:
        requested.extend(ALL_STEPS if step == "all" else [step])

    # 去重且保持出现顺序。
    requested = list(dict.fromkeys(requested))
    for index, step in enumerate(requested, 1):
        command = command_for(step, args)
        print(f"[{index}/{len(requested)}] {step}: {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

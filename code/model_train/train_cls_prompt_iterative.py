"""迭代式主动学习流水线的单轮训练入口，薄包装 train_cls_prompt.py。

写法与已删除的 train_cls_context.py 一致：monkeypatch train_cls_prompt 模块
的全局变量，再调用其 main()，不重写任何训练/评估/实验记录逻辑。区别是本文件
所有覆盖值都来自命令行参数，供 run_iterative_pipeline.py 按轮次/按参数组调用。

TEST_FILE 默认不覆盖：train_cls_prompt.py 的 TEST_FILE 指向 SPLIT_DIR/test.jsonl
（split_data.py 产出的原始测试集），主线迭代过程中永远不变。只有换场景重建了
配套测试集时（如 gov_scene 政务场景第5轮，train/test 都来自新构建的政务数据）
才需要显式传 --test-file，否则会拿旧的通用测试集评估新场景的模型。

用法：
    python3 model_train/train_cls_prompt_iterative.py \\
        --model-path /path/to/base/or/previous/best_model \\
        --train-file outputs/sarcasm_cls_prompt_iterative/train_iter2.jsonl \\
        --output-dir outputs/sarcasm_cls_prompt_iterative \\
        --learning-rate 1.5e-5 --focal-alpha 0.45 --num-epochs 2.5

    # 换场景（同时替换 train 和 test）：
    python3 model_train/train_cls_prompt_iterative.py \\
        --model-path .../best_model \\
        --train-file outputs/gov_scene_round5/train_gov.jsonl \\
        --test-file outputs/gov_scene_round5/test_gov.jsonl \\
        --output-dir outputs/gov_scene_round5/group1 \\
        --learning-rate 5e-6 --focal-alpha 0.6 --num-epochs 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_cls_prompt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True,
                        help="本轮训练起点权重：base 底座，或上一轮胜出模型的 best_model 目录")
    parser.add_argument("--train-file", required=True, help="本轮 train JSONL")
    parser.add_argument("--test-file", default=None,
                        help="本轮 test JSONL；不传则用 train_cls_prompt.py 里的默认值"
                             "（SPLIT_DIR/test.jsonl）。换场景重建了配套测试集时必须传，"
                             "否则会拿旧的通用测试集评估新场景的模型。")
    parser.add_argument("--output-dir", required=True, help="本次实验输出根目录")
    parser.add_argument("--learning-rate", type=float, default=train_cls_prompt.LEARNING_RATE)
    parser.add_argument("--focal-alpha", type=float, default=train_cls_prompt.FOCAL_ALPHA)
    parser.add_argument("--num-epochs", type=float, default=train_cls_prompt.NUM_EPOCHS)
    parser.add_argument("--max-length", type=int, default=train_cls_prompt.MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=train_cls_prompt.BATCH_SIZE)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_cls_prompt.MODEL_PATH = args.model_path
    train_cls_prompt.OUTPUT_DIR = Path(args.output_dir)
    train_cls_prompt.LEARNING_RATE = args.learning_rate
    train_cls_prompt.FOCAL_ALPHA = args.focal_alpha
    train_cls_prompt.NUM_EPOCHS = args.num_epochs
    train_cls_prompt.MAX_LENGTH = args.max_length
    train_cls_prompt.BATCH_SIZE = args.batch_size
    if args.test_file is not None:
        train_cls_prompt.TEST_FILE = args.test_file
    # train_cls_prompt.main() 自己也会 parse_args(sys.argv)，用来读 --train-file；
    # 保持 sys.argv 与本次调用一致，让它読到我们指定的 train_file，不重新实现一套解析。
    sys.argv = [sys.argv[0], "--train-file", args.train_file]
    train_cls_prompt.main()


if __name__ == "__main__":
    main()

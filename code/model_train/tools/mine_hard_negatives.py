"""迭代式主动学习流水线的核心步骤：用上一轮模型挖出新一批标0负例，拼出下一轮 train。

背景（工作日志 2026-08-18 前后讨论）：train/test 是 1:1 平衡集，标0数据在训练中
覆盖不够，导致真实比例下准确率远低于 1:1 测试集。本脚本不改变正例（讽刺类）
部分，只把训练集里的标0数据整体替换成"当前模型最容易判错的那批真实标0数据"：

    候选池 = LABELED_FILE 中 is_sarcasm==0，且从未进入过
             train.jsonl / test.jsonl / REAL_DIST_EVAL_FILE 的记录
             （后两者一旦用过就一直排除，见 ITERATIVE_STATE_FILE）
    难负例 = 候选池中被传入模型判为 1（讽刺）的记录 —— 即模型当前的假阳性，
             占所需负例数的 hard_negative_ratio 比例
    普通负例 = 候选池中模型也判 0（判断一致）的记录，占剩余比例，用于避免
             负例全是边界样本导致模型过度保守（工作日志记录过 iter2 recall
             从84%崩到53~56%的教训）
    新负例 = 难负例 + 普通负例，凑够所需数量；任一类不足直接报错停止，
             多了则固定种子随机降采样（不按置信度筛选）。

正例（label=1）永远原样取自最初的 train.jsonl，不随轮次变化（默认按
文本+上下文去重，见 load_original_train_positives）。

"边标边停"（early_stop，默认开启）：候选池可能有上百万条，逐条推理成本很高，
但往往扫描一小部分就能凑够所需的难负例数量。本脚本默认不会对候选池做全量
推理，而是流式扫描（stream_candidate_pool + predict_probs_stream），难负例
缓冲池累计达到"所需难负例数 * buffer_multiplier"（默认3倍，凑够3倍缓冲池再
随机降采样到刚好数量，避免结果偏向候选池里靠前的部分）就停止扫描；普通负例
凑够所需数量也停止收集。两者都满足才真正停止扫描。可用 --no-early-stop 关闭，
回退到对候选池全量推理（原始行为，用于调试/对照实验）。

模型推理复用 eval_on_real_distribution_prompt.py 里已验证过的
resolve_experiment / predict_probs / predict_probs_stream（Prompt+MLM 范式，
[MASK] 位置"是/否" verbalizer），不重新实现一遍推理逻辑。

用法：
    # 第2轮起：给定第1轮/上一轮胜出模型的实验目录，挖出本轮负例（默认边标边停）
    python3 model_train/tools/mine_hard_negatives.py \\
        --experiment-dir outputs/sarcasm_cls_prompt_iterative/experiments/exp_xxx \\
        --output-train-file outputs/sarcasm_cls_prompt_iterative/train_iter2.jsonl \\
        --iteration 2 --hard-negative-ratio 0.6

    python3 model_train/tools/mine_hard_negatives.py --selfcheck   # 不加载模型的纯逻辑自检
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import ITERATIVE_STATE_FILE, LABELED_FILE, REAL_DIST_EVAL_FILE, SPLIT_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 1:1 口径与 split_data.py 保持一致，不重复定义。
from split_data import TRAIN_NEGATIVE_PER_POSITIVE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_on_real_distribution_prompt import predict_probs, predict_probs_stream, resolve_experiment
from scan_label_disagreements import normalize_text


ORIGINAL_TRAIN_FILE = SPLIT_DIR / "train.jsonl"
ORIGINAL_TEST_FILE = SPLIT_DIR / "test.jsonl"
RANDOM_SEED = 42
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# 基础 IO：与项目里其它脚本保持同一套流式读取风格，不整份载入内存。
# ---------------------------------------------------------------------------
def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 已用池状态：ITERATIVE_STATE_FILE 记录哪些 source_line 已经被用作训练/测试/
# 真实分布评估，避免同一条数据被反复挖到、或被同时用作评估和训练。
# ---------------------------------------------------------------------------
def _require_source_lines(rows, file_label):
    """split_data.py 当前版本会给每行写 source_line；缺失说明数据是旧版产物。"""
    lines = set()
    for row in rows:
        source_line = row.get("source_line")
        if not isinstance(source_line, int):
            raise ValueError(
                f"{file_label} 中存在缺少整数 source_line 的记录，"
                "无法安全排重（这通常说明该文件是旧版 split_data.py 产出的，"
                "请重新运行 model_train/split_data.py 生成带 source_line 的版本）。"
            )
        lines.add(source_line)
    return lines


def init_used_source_lines(train_file, test_file, real_dist_eval_file):
    """首次运行时的已用池初始值：原始 train + test + 真实分布评估集里的负例。"""
    train_rows = [row for _, row in iter_jsonl(train_file)]
    test_rows = [row for _, row in iter_jsonl(test_file)]
    used = _require_source_lines(train_rows, str(train_file))
    used |= _require_source_lines(test_rows, str(test_file))

    real_dist_path = Path(real_dist_eval_file)
    if real_dist_path.exists():
        # 真实分布评估集里只有负例携带 source_line（正例复用 test.jsonl，
        # 已在上面算过），只需要额外收集负例部分，见 build_real_distribution_eval.py。
        for _, row in iter_jsonl(real_dist_path):
            if row.get("label") == 0 and isinstance(row.get("source_line"), int):
                used.add(row["source_line"])
    return used


def load_state(state_file, train_file, test_file, real_dist_eval_file):
    state_file = Path(state_file)
    if state_file.exists():
        with state_file.open("r", encoding="utf-8") as f:
            state = json.load(f)
        state["used_source_lines"] = set(state["used_source_lines"])
        return state
    return {
        "used_source_lines": init_used_source_lines(train_file, test_file, real_dist_eval_file),
        "iterations": [],
    }


def save_state(state_file, state):
    state_file = Path(state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "used_source_lines": sorted(state["used_source_lines"]),
        "iterations": state["iterations"],
    }
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 正例：内容上永远取自最初的 train.jsonl，不随轮次变化；但默认按
# (text, retweeted_content) 去重——train.jsonl 如果曾经跑过
# augment_train_positives.py（正样本复制一份），重复的正例会跟着每一轮
# 一直复制下去。本脚本只在第2轮及以后被调用（第1轮直接用原始 train.jsonl，
# 不经过这里），所以这里去重只影响从第2轮起新构造的训练集，不改动
# train.jsonl 本身，也不影响已经跑完的第1轮。
# ---------------------------------------------------------------------------
def load_original_train_positives(train_file, dedupe=True):
    rows = [row for _, row in iter_jsonl(train_file)]
    _require_source_lines(rows, str(train_file))
    positives = [row for row in rows if row.get("label") == 1]
    if not positives:
        raise ValueError(f"{train_file} 中没有找到任何 label=1 的正例")
    if not dedupe:
        return positives

    seen = set()
    deduped = []
    for row in positives:
        key = (normalize_text(row.get("text", "")), normalize_text(row.get("retweeted_content", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    removed = len(positives) - len(deduped)
    if removed:
        print(f"正例去重：{len(positives)} 条中发现 {removed} 条重复（同文本+同上下文），去重后剩 {len(deduped)} 条")
    return deduped


# ---------------------------------------------------------------------------
# 候选池与挖掘
# ---------------------------------------------------------------------------
def stream_candidate_pool(labeled_file, used_source_lines):
    """逐行生成 LABELED_FILE 中标0、且从未进入已用池的记录，不整份物化成列表。

    与 build_candidate_pool 的区别：这是一个生成器，配合 predict_probs_stream
    使用时，调用方可以在扫描过程中随时 break（比如凑够所需数量），文件里
    尚未读到的部分永远不会被打开/解析，这是"边标边停"能省下标注成本的
    关键——候选池可能有上百万条，往往只需要扫一小部分就够用。
    """
    for line_no, row in iter_jsonl(labeled_file):
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
        }


def build_candidate_pool(labeled_file, used_source_lines, limit=None):
    """LABELED_FILE 中标0、且从未进入已用池的记录，一次性物化成列表。

    供 precompute_candidate_predictions.py 等需要拿到候选池全量条数/推理全部
    候选池的场景使用；"边标边停"场景请用 stream_candidate_pool + predict_probs_stream
    （见 scan_and_collect_negatives），不要在扫描前把候选池整份读进内存。
    """
    rows = []
    for row in stream_candidate_pool(labeled_file, used_source_lines):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def scan_and_collect_negatives(candidate_iterable, model_path, max_length, batch_size,
                                threshold, hard_needed, random_needed, buffer_multiplier=3):
    """边推理边判断是否已凑够，一旦达到停止条件就不再继续扫描候选池。

    停止条件（先满足哪个就停）：
      1. 难负例（模型判1，即假阳性）缓冲池达到 hard_needed * buffer_multiplier 条
         （默认3倍缓冲，之后从缓冲池里随机抽 hard_needed 条，而不是直接取"扫到的
         前 hard_needed 条"，避免结果偏向候选池里靠前的部分）；
      2. 普通负例（模型判0）已凑够 random_needed 条（这部分候选池里通常占多数，
         不设缓冲倍数，够用即停）；
      3. 候选池被扫完（数据不够，交给上层 select_hard_negatives 报错）。

    返回 (难负例缓冲池, 普通负例列表, 实际扫描条数)；难负例缓冲池可能超过
    hard_needed（最多到 hard_needed*buffer_multiplier），由调用方按
    select_hard_negatives 同样的随机降采样逻辑抽到刚好数量。
    """
    hard_buffer = []
    random_collected = []
    scanned = 0
    hard_buffer_target = hard_needed * buffer_multiplier

    for row, prob in predict_probs_stream(candidate_iterable, model_path, max_length, batch_size):
        scanned += 1
        predicted = int(prob >= threshold)
        if predicted == 1:
            if len(hard_buffer) < hard_buffer_target:
                hard_buffer.append(row)
        else:
            if len(random_collected) < random_needed:
                random_collected.append(row)

        hard_done = hard_needed == 0 or len(hard_buffer) >= hard_buffer_target
        random_done = random_needed == 0 or len(random_collected) >= random_needed
        if hard_done and random_done:
            break

    return hard_buffer, random_collected, scanned


def select_hard_negatives(candidate_rows, predicted_labels, needed_count, seed=RANDOM_SEED,
                          hard_negative_ratio=1.0):
    """从候选池里筛出负例，抽到刚好 needed_count 条。

    hard_negative_ratio 控制难负例（模型判1的假阳性）占比，默认1.0即全部为
    难负例（原始行为）。传小于1的值时，会把一部分难负例换成候选池里的普通
    负例（模型判0、真实也是0的样本），增加数据多样性，缓解"整批训练数据都是
    模型当前最容易犯错的边界样本"导致的决策边界过度收缩（工作日志记录过：
    iter2 四组全负例都是难负例时，recall 从84%崩到53~56%，模型变得极度保守）。

    命中数（难负例池）不足所需的难负例数量时直接报错停止，不做静默降级；
    某一类别数量超出所需时固定种子随机降采样（不按置信度挑选）。
    """
    if len(candidate_rows) != len(predicted_labels):
        raise ValueError("候选行数与预测标签数量不一致")
    if not 0 <= hard_negative_ratio <= 1:
        raise ValueError(f"hard_negative_ratio 必须在 [0, 1] 范围内，实际为 {hard_negative_ratio}")

    hits = [row for row, pred in zip(candidate_rows, predicted_labels) if pred == 1]
    misses = [row for row, pred in zip(candidate_rows, predicted_labels) if pred == 0]

    hard_needed = round(needed_count * hard_negative_ratio)
    random_needed = needed_count - hard_needed

    if len(hits) < hard_needed:
        raise RuntimeError(
            f"候选池里的假阳性（难负例）数量不足：本轮需要 {hard_needed} 条，"
            f"候选池 {len(candidate_rows)} 条里只挖到 {len(hits)} 条模型误判为1的记录。"
            "请检查候选池是否已被耗尽（历史轮次挖太多），或模型是否已经收敛到"
            "很少犯错的状态。"
        )
    if len(misses) < random_needed:
        raise RuntimeError(
            f"候选池里的普通负例（模型判0）数量不足：本轮需要 {random_needed} 条，"
            f"候选池 {len(candidate_rows)} 条里只有 {len(misses)} 条模型判0的记录。"
        )

    rng = random.Random(seed)
    hard_sample = hits if len(hits) == hard_needed else rng.sample(hits, hard_needed)
    random_sample = misses if len(misses) == random_needed else rng.sample(misses, random_needed)
    return hard_sample + random_sample


def assemble_train_rows(positives, negatives, seed=RANDOM_SEED):
    rows = list(positives) + list(negatives)
    random.Random(seed).shuffle(rows)
    return rows


def load_predictions_cache(cache_path, candidate_rows):
    """读取 precompute_candidate_predictions.py 产出的缓存，按 source_line 对齐
    到当前候选池，返回与 candidate_rows 顺序一致的 probability 列表。

    要求缓存里的 source_line 集合是当前候选池 source_line 集合的超集——候选池
    只会因为已用池增大而变小（历史轮次挖走一些），不会变大，超出的缓存行忽略；
    如果候选池里有 source_line 不在缓存里，说明缓存是用不同/更旧的候选池算的，
    直接报错，不做静默降级（比如临时对这部分重新跑推理），避免悄悄混用两份
    不同来源的预测结果。
    """
    cache_path = Path(cache_path)
    prob_by_line = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_meta"):
                continue
            prob_by_line[row["source_line"]] = row["probability"]

    missing = [row["source_line"] for row in candidate_rows if row["source_line"] not in prob_by_line]
    if missing:
        raise ValueError(
            f"{cache_path} 缺少 {len(missing)} 条当前候选池样本的预测结果"
            f"（如 source_line={missing[:5]}），缓存可能是用不同的候选池/已用池算的，"
            "请重新运行 precompute_candidate_predictions.py 生成配套缓存。"
        )
    return [prob_by_line[row["source_line"]] for row in candidate_rows]


def select_from_scan_result(hard_buffer, random_collected, hard_needed, random_needed,
                            seed=RANDOM_SEED):
    """对 scan_and_collect_negatives 的结果做最终筛选：难负例缓冲池随机降采样到
    刚好 hard_needed 条（缓冲池不足则报错），普通负例要求已凑够 random_needed 条。
    """
    if len(hard_buffer) < hard_needed:
        raise RuntimeError(
            f"候选池扫描完仍未凑够难负例缓冲池：本轮需要 {hard_needed} 条难负例，"
            f"候选池扫描期间只收集到 {len(hard_buffer)} 条模型误判为1的记录。"
            "请检查候选池是否已被耗尽（历史轮次挖太多），或模型是否已经收敛到"
            "很少犯错的状态。"
        )
    if len(random_collected) < random_needed:
        raise RuntimeError(
            f"候选池扫描完仍未凑够普通负例：本轮需要 {random_needed} 条，"
            f"候选池扫描期间只收集到 {len(random_collected)} 条模型判0的记录。"
        )
    rng = random.Random(seed)
    hard_sample = (
        hard_buffer if len(hard_buffer) == hard_needed else rng.sample(hard_buffer, hard_needed)
    )
    return hard_sample + random_collected[:random_needed]


def run_mining(experiment_dir, train_file, test_file, labeled_file, real_dist_eval_file,
              state_file, output_train_file, iteration, batch_size=BATCH_SIZE,
              seed=RANDOM_SEED, candidate_limit=None, dedupe_positives=True,
              hard_negative_ratio=1.0, predictions_cache=None, early_stop=True,
              buffer_multiplier=3):
    """完整一轮挖掘：加载模型配置 -> 建候选池 -> 推理（或读缓存） -> 筛选 -> 写新 train -> 更新状态。

    early_stop=True（默认）：边扫描候选池边推理，难负例凑够
    hard_needed*buffer_multiplier（默认3倍缓冲池）就停止扫描，不需要标注全量
    候选池（可能上百万条）。predictions_cache 不为空时忽略 early_stop（缓存
    本身就是一次性对全量候选池算好的，没有"边标边停"的意义）。
    """
    resolved = resolve_experiment(experiment_dir)
    positives = load_original_train_positives(train_file, dedupe=dedupe_positives)
    needed = len(positives) * TRAIN_NEGATIVE_PER_POSITIVE
    hard_needed = round(needed * hard_negative_ratio)
    random_needed = needed - hard_needed

    state = load_state(state_file, train_file, test_file, real_dist_eval_file)

    if predictions_cache is not None:
        candidate_rows = build_candidate_pool(
            labeled_file, state["used_source_lines"], limit=candidate_limit
        )
        if not candidate_rows:
            raise RuntimeError("候选池为空：所有标0数据都已被历史轮次用过或已进入 train/test。")
        probs = load_predictions_cache(predictions_cache, candidate_rows)
        predicted = [int(p >= resolved["threshold"]) for p in probs]
        negatives = select_hard_negatives(
            candidate_rows, predicted, needed, seed=seed, hard_negative_ratio=hard_negative_ratio
        )
        scanned_count = len(candidate_rows)
        hit_count = sum(predicted)
    elif early_stop and candidate_limit is None:
        candidate_iterable = stream_candidate_pool(labeled_file, state["used_source_lines"])
        hard_buffer, random_collected, scanned_count = scan_and_collect_negatives(
            candidate_iterable, resolved["model_path"], resolved["max_length"], batch_size,
            resolved["threshold"], hard_needed, random_needed, buffer_multiplier=buffer_multiplier,
        )
        if scanned_count == 0:
            raise RuntimeError("候选池为空：所有标0数据都已被历史轮次用过或已进入 train/test。")
        negatives = select_from_scan_result(hard_buffer, random_collected, hard_needed,
                                            random_needed, seed=seed)
        hit_count = len(hard_buffer)
    else:
        # early_stop=False 或调试用 candidate_limit：整份物化候选池再全量推理，
        # 与最初版本行为一致，用于对照验证或需要固定候选池规模的场景。
        candidate_rows = build_candidate_pool(
            labeled_file, state["used_source_lines"], limit=candidate_limit
        )
        if not candidate_rows:
            raise RuntimeError("候选池为空：所有标0数据都已被历史轮次用过或已进入 train/test。")
        probs = predict_probs(candidate_rows, resolved["model_path"], resolved["max_length"], batch_size)
        predicted = [int(p >= resolved["threshold"]) for p in probs]
        negatives = select_hard_negatives(
            candidate_rows, predicted, needed, seed=seed, hard_negative_ratio=hard_negative_ratio
        )
        scanned_count = len(candidate_rows)
        hit_count = sum(predicted)

    out_rows = assemble_train_rows(positives, negatives, seed=seed)
    write_jsonl(out_rows, output_train_file)

    state["used_source_lines"] |= {row["source_line"] for row in negatives}
    state["iterations"].append({
        "iteration": iteration,
        "experiment_dir": str(experiment_dir),
        "model_path": str(resolved["model_path"]),
        "threshold": resolved["threshold"],
        "scanned_count": scanned_count,
        "hit_count": hit_count,
        "needed_count": needed,
        "hard_needed": hard_needed,
        "random_needed": random_needed,
        "sampled_count": len(negatives),
        "early_stop": early_stop and predictions_cache is None and candidate_limit is None,
        "output_train_file": str(output_train_file),
        "positive_count": len(positives),
    })
    save_state(state_file, state)

    return {
        "positive_count": len(positives),
        "needed_count": needed,
        "hard_needed": hard_needed,
        "random_needed": random_needed,
        "scanned_count": scanned_count,
        "hit_count": hit_count,
        "sampled_count": len(negatives),
        "early_stop": early_stop and predictions_cache is None and candidate_limit is None,
        "output_train_file": str(output_train_file),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True,
                        help="上一轮胜出模型的实验目录（含 best_model/train_config.json/decision_config.json）")
    parser.add_argument("--output-train-file", required=True, help="本轮新 train JSONL 输出路径")
    parser.add_argument("--iteration", type=int, required=True, help="轮次编号，仅用于状态记录")
    parser.add_argument("--train-file", default=str(ORIGINAL_TRAIN_FILE),
                        help="最初的 train.jsonl（正例来源，永远不变）")
    parser.add_argument("--test-file", default=str(ORIGINAL_TEST_FILE))
    parser.add_argument("--labeled-file", default=str(LABELED_FILE))
    parser.add_argument("--real-dist-eval-file", default=str(REAL_DIST_EVAL_FILE))
    parser.add_argument("--state-file", default=str(ITERATIVE_STATE_FILE))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--candidate-limit", type=int, default=None,
                        help="调试用：只扫候选池前 N 条，默认不限制（结合 --no-early-stop"
                             "才是原始的全量物化行为；否则该参数会关闭边标边停）")
    parser.add_argument("--no-dedupe-positives", action="store_true",
                        help="不对正例做去重，原样使用 train.jsonl 里的全部 label=1 行"
                             "（默认会按 文本+上下文 去重，避免 augment_train_positives.py"
                             " 复制过的正例被后续每一轮持续放大）")
    parser.add_argument("--hard-negative-ratio", type=float, default=1.0,
                        help="负例中难负例（模型误判为1的假阳性）的占比，范围[0,1]，"
                             "默认1.0即全部为难负例。传小于1的值（如0.6）会混入一部分"
                             "普通负例（模型判0），增加数据多样性，缓解全负例都是边界样本"
                             "导致模型过度保守、recall崩塌的问题。")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="关闭“边标边停”，改为对全量候选池做一次性推理"
                             "（原始行为，候选池上百万条时会很慢；调试/对照实验时可用）")
    parser.add_argument("--buffer-multiplier", type=float, default=3.0,
                        help="边标边停时，难负例缓冲池目标大小 = 所需难负例数 * 该倍数"
                             "（默认3倍），凑够缓冲池就停止扫描候选池，再从缓冲池随机降采样"
                             "到刚好所需数量")
    parser.add_argument("--predictions-cache",
                        help="复用 precompute_candidate_predictions.py 预先算好的候选池推理"
                             "结果（JSONL），跳过模型加载和推理，只重跑筛选逻辑；用于反复调整"
                             "--hard-negative-ratio 时省掉重复推理的耗时。缓存必须与当前候选池"
                             "（同一 --labeled-file + --state-file）配套，否则报错。传此参数时"
                             "--no-early-stop/--buffer-multiplier 不生效（缓存本身就是全量的）。")
    args = parser.parse_args()

    summary = run_mining(
        args.experiment_dir, args.train_file, args.test_file, args.labeled_file,
        args.real_dist_eval_file, args.state_file, args.output_train_file, args.iteration,
        batch_size=args.batch_size, seed=args.seed, candidate_limit=args.candidate_limit,
        dedupe_positives=not args.no_dedupe_positives,
        hard_negative_ratio=args.hard_negative_ratio,
        predictions_cache=args.predictions_cache,
        early_stop=not args.no_early_stop,
        buffer_multiplier=args.buffer_multiplier,
    )
    print(f"正例数（不变）：{summary['positive_count']}")
    print(f"本轮需要标0负例：{summary['needed_count']}"
          f"（难负例 {summary['hard_needed']} / 普通负例 {summary['random_needed']}）")
    print(f"实际扫描候选池：{summary['scanned_count']} 条"
          f"（{'边标边停' if summary['early_stop'] else '全量推理'}），"
          f"命中假阳性（难负例）：{summary['hit_count']} 条")
    print(f"实际写入负例：{summary['sampled_count']} 条")
    print(f"新 train 已写入：{summary['output_train_file']}")


def _selfcheck():
    import tempfile

    # 1) init_used_source_lines：train/test 的 source_line 全部计入，
    #    real_dist_eval 只计入 label=0 的部分（正例来自 test.jsonl，没有独立 source_line）。
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_path = tmp_path / "train.jsonl"
        test_path = tmp_path / "test.jsonl"
        real_dist_path = tmp_path / "real_dist.jsonl"

        with train_path.open("w", encoding="utf-8") as f:
            for i in range(1, 4):
                f.write(json.dumps({"text": f"pos{i}", "retweeted_content": "",
                                    "label": 1, "source_line": i}) + "\n")
            for i in range(4, 7):
                f.write(json.dumps({"text": f"neg{i}", "retweeted_content": "",
                                    "label": 0, "source_line": i}) + "\n")
        with test_path.open("w", encoding="utf-8") as f:
            for i in range(7, 9):
                f.write(json.dumps({"text": f"t{i}", "retweeted_content": "",
                                    "label": 0, "source_line": i}) + "\n")
        with real_dist_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "p", "retweeted_content": "", "label": 1,
                                "source_line": None, "origin": "test_positive"}) + "\n")
            f.write(json.dumps({"text": "n", "retweeted_content": "", "label": 0,
                                "source_line": 100, "origin": "labeled_unused_negative"}) + "\n")

        used = init_used_source_lines(train_path, test_path, real_dist_path)
        assert used == {1, 2, 3, 4, 5, 6, 7, 8, 100}, used

        # load_original_train_positives 只取 label=1
        positives = load_original_train_positives(train_path)
        assert [p["text"] for p in positives] == ["pos1", "pos2", "pos3"], positives

        # 正例去重：同文本+同上下文的重复行（模拟 augment_train_positives.py
        # 复制过一遍的正例）默认应被去重；--no-dedupe-positives 等价的
        # dedupe=False 则应保留全部重复。
        dup_train_path = tmp_path / "dup_train.jsonl"
        with dup_train_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "p1", "retweeted_content": "c1", "label": 1, "source_line": 1}) + "\n")
            f.write(json.dumps({"text": "p1", "retweeted_content": "c1", "label": 1, "source_line": 2}) + "\n")
            f.write(json.dumps({"text": "p2", "retweeted_content": "", "label": 1, "source_line": 3}) + "\n")
        deduped = load_original_train_positives(dup_train_path)
        assert len(deduped) == 2, deduped
        undeduped = load_original_train_positives(dup_train_path, dedupe=False)
        assert len(undeduped) == 3, undeduped

        # 缺 source_line 时应明确报错，而不是静默用行号兜底。
        bad_train_path = tmp_path / "bad_train.jsonl"
        with bad_train_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "x", "retweeted_content": "", "label": 1}) + "\n")
        try:
            load_original_train_positives(bad_train_path)
            raise AssertionError("缺少 source_line 时应抛出 ValueError")
        except ValueError:
            pass

        # state 读写往返
        state_path = tmp_path / "state.json"
        state = load_state(state_path, train_path, test_path, real_dist_path)
        assert state["used_source_lines"] == used
        state["iterations"].append({"iteration": 1})
        save_state(state_path, state)
        reloaded = load_state(state_path, train_path, test_path, real_dist_path)
        assert reloaded["used_source_lines"] == used
        assert reloaded["iterations"] == [{"iteration": 1}]

    # 2) build_candidate_pool：排除已用池、排除非标0、排除空文本。
    with tempfile.TemporaryDirectory() as tmp:
        labeled_path = Path(tmp) / "labeled.jsonl"
        with labeled_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"content": "a", "is_sarcasm": 0, "source_line": 1}) + "\n")
            f.write(json.dumps({"content": "b", "is_sarcasm": 0, "source_line": 2}) + "\n")
            f.write(json.dumps({"content": "c", "is_sarcasm": 1, "source_line": 3}) + "\n")
            f.write(json.dumps({"content": "  ", "is_sarcasm": 0, "source_line": 4}) + "\n")
        pool = build_candidate_pool(labeled_path, used_source_lines={1})
        assert [row["source_line"] for row in pool] == [2], pool

    # 3) select_hard_negatives：命中不足报错；命中刚好/超出时行为正确、可复现。
    rows = [{"source_line": i} for i in range(10)]
    try:
        select_hard_negatives(rows, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], needed_count=3)
        raise AssertionError("命中数不足时应抛出 RuntimeError")
    except RuntimeError:
        pass

    exact = select_hard_negatives(rows, [1] * 5 + [0] * 5, needed_count=5)
    assert {r["source_line"] for r in exact} == set(range(5)), exact

    oversupplied_a = select_hard_negatives(rows, [1] * 10, needed_count=4, seed=1)
    oversupplied_b = select_hard_negatives(rows, [1] * 10, needed_count=4, seed=1)
    assert len(oversupplied_a) == 4
    assert oversupplied_a == oversupplied_b, "同一 seed 必须可复现"

    # 3b) hard_negative_ratio：混入普通负例增加多样性，缓解全负例都是边界样本
    # 导致模型过度保守的问题（工作日志记录的 iter2 recall 崩塌问题）。
    # 候选池 20 条：source_line 0-11 是难负例候选（12条），12-19 是普通负例候选（8条），
    # 数量留足余量，避免 round() 取整后任一类别不够。
    mixed_rows = [{"source_line": i} for i in range(20)]
    mixed_preds = [1] * 12 + [0] * 8
    mixed = select_hard_negatives(mixed_rows, mixed_preds, needed_count=10, hard_negative_ratio=0.6)
    hard_part = [r for r in mixed if r["source_line"] < 12]
    random_part = [r for r in mixed if r["source_line"] >= 12]
    assert len(hard_part) == 6, mixed  # round(10*0.6)=6
    assert len(random_part) == 4, mixed
    assert len(mixed) == 10

    # ratio=1.0（默认）应与旧行为完全一致：全部来自难负例。
    all_hard = select_hard_negatives(mixed_rows, mixed_preds, needed_count=5, hard_negative_ratio=1.0)
    assert all(r["source_line"] < 12 for r in all_hard), all_hard

    # ratio=0.0：全部来自普通负例。
    all_random = select_hard_negatives(mixed_rows, mixed_preds, needed_count=4, hard_negative_ratio=0.0)
    assert all(r["source_line"] >= 12 for r in all_random), all_random

    # 普通负例池不够时应报错（而不是静默用难负例补足）。
    try:
        select_hard_negatives(mixed_rows, mixed_preds, needed_count=20, hard_negative_ratio=0.0)
        raise AssertionError("普通负例数量不足时应抛出 RuntimeError")
    except RuntimeError:
        pass

    # ratio 越界应报错。
    try:
        select_hard_negatives(mixed_rows, mixed_preds, needed_count=5, hard_negative_ratio=1.5)
        raise AssertionError("hard_negative_ratio 越界时应抛出 ValueError")
    except ValueError:
        pass

    # 3c) load_predictions_cache：按 source_line 对齐候选池；缺失样本应报错，
    # 不做静默降级。
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "cache.jsonl"
        with cache_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": True, "experiment_dir": "exp_x", "candidate_pool_size": 2}) + "\n")
            f.write(json.dumps({"source_line": 10, "text": "a", "retweeted_content": "",
                                "label": 0, "probability": 0.8}) + "\n")
            f.write(json.dumps({"source_line": 20, "text": "b", "retweeted_content": "",
                                "label": 0, "probability": 0.2}) + "\n")

        candidates = [{"source_line": 10}, {"source_line": 20}]
        probs = load_predictions_cache(cache_path, candidates)
        assert probs == [0.8, 0.2], probs

        # 候选池缩小（部分被历史轮次用掉）时，多余的缓存行应被忽略，不报错。
        shrunk = load_predictions_cache(cache_path, [{"source_line": 20}])
        assert shrunk == [0.2], shrunk

        # 候选池里出现缓存没有的 source_line 应报错。
        try:
            load_predictions_cache(cache_path, [{"source_line": 10}, {"source_line": 99}])
            raise AssertionError("候选池样本缺少对应缓存时应抛出 ValueError")
        except ValueError:
            pass

    # 4) assemble_train_rows：正负例总数不变，且包含所有输入行。
    positives = [{"text": f"p{i}", "label": 1} for i in range(3)]
    negatives = [{"text": f"n{i}", "label": 0} for i in range(3)]
    combined = assemble_train_rows(positives, negatives)
    assert len(combined) == 6
    assert {row["text"] for row in combined} == {p["text"] for p in positives} | {n["text"] for n in negatives}

    # 5) stream_candidate_pool：逐行生成，排除结果与 build_candidate_pool 一致
    # （同一份 filter 逻辑，只是不整份物化）。
    with tempfile.TemporaryDirectory() as tmp:
        labeled_path = Path(tmp) / "labeled.jsonl"
        with labeled_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"content": "a", "is_sarcasm": 0, "source_line": 1}) + "\n")
            f.write(json.dumps({"content": "b", "is_sarcasm": 0, "source_line": 2}) + "\n")
            f.write(json.dumps({"content": "c", "is_sarcasm": 1, "source_line": 3}) + "\n")
        streamed = list(stream_candidate_pool(labeled_path, used_source_lines={1}))
        assert [row["source_line"] for row in streamed] == [2], streamed

    # 6) scan_and_collect_negatives：用假的 predict_probs_stream 验证"边扫边停"
    # 调度逻辑本身——难负例凑够 hard_needed*buffer_multiplier 就停，普通负例
    # 凑够 random_needed 就停，两者都满足才真正停止，不会把整个可迭代对象耗尽。
    def _fake_stream(row_iterable, model_path, max_length, batch_size):
        for row in row_iterable:
            # source_line 为偶数的模型判1（难负例候选），奇数判0（普通负例候选）。
            prob = 0.9 if row["source_line"] % 2 == 0 else 0.1
            yield row, prob

    original_stream = globals()["predict_probs_stream"]
    globals()["predict_probs_stream"] = _fake_stream
    try:
        # 候选池给足够多（100条），验证确实提前停止、没有扫完全部。
        candidate_iterable = ({"source_line": i} for i in range(100))
        hard_buffer, random_collected, scanned = scan_and_collect_negatives(
            candidate_iterable, model_path=None, max_length=384, batch_size=8,
            threshold=0.5, hard_needed=3, random_needed=2, buffer_multiplier=3,
        )
        # hard_needed=3, buffer_multiplier=3 -> 缓冲池目标9条；random_needed=2。
        assert len(hard_buffer) == 9, hard_buffer
        assert len(random_collected) == 2, random_collected
        assert scanned < 100, "应该提前停止，不应该扫完全部候选池"
        assert all(row["source_line"] % 2 == 0 for row in hard_buffer), hard_buffer
        assert all(row["source_line"] % 2 == 1 for row in random_collected), random_collected

        # 候选池本身不够时，应该扫完全部（不会死循环/报错，交给上层判断是否足够）。
        small_iterable = ({"source_line": i} for i in range(4))
        hard_buffer2, random_collected2, scanned2 = scan_and_collect_negatives(
            small_iterable, model_path=None, max_length=384, batch_size=8,
            threshold=0.5, hard_needed=10, random_needed=10, buffer_multiplier=3,
        )
        assert scanned2 == 4, scanned2
        assert len(hard_buffer2) == 2, hard_buffer2  # source_line 0, 2
        assert len(random_collected2) == 2, random_collected2  # source_line 1, 3
    finally:
        globals()["predict_probs_stream"] = original_stream

    # 7) select_from_scan_result：缓冲池随机降采样到刚好数量；任一类不足报错。
    hard_buf = [{"source_line": i} for i in range(9)]
    random_col = [{"source_line": 100 + i} for i in range(2)]
    selected = select_from_scan_result(hard_buf, random_col, hard_needed=3, random_needed=2, seed=1)
    assert len(selected) == 5
    hard_part = [r for r in selected if r["source_line"] < 9]
    random_part = [r for r in selected if r["source_line"] >= 100]
    assert len(hard_part) == 3 and len(random_part) == 2

    try:
        select_from_scan_result(hard_buf[:2], random_col, hard_needed=3, random_needed=2)
        raise AssertionError("难负例缓冲池不足时应抛出 RuntimeError")
    except RuntimeError:
        pass

    try:
        select_from_scan_result(hard_buf, random_col[:1], hard_needed=3, random_needed=2)
        raise AssertionError("普通负例不足时应抛出 RuntimeError")
    except RuntimeError:
        pass

    print("自检通过")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

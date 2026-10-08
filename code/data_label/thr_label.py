#!/usr/bin/env python3
"""使用三种模型标注反讽，并按投票规则生成最终标签。

最终标签的判定规则只在 relabel_votes.final_label 里定义一处（"0 票数 <= 1 且
1 票数 > 0 记为 1"），本脚本直接复用，不再自己实现——历史上这里有过一份独立
实现，与文档描述和离线重标脚本三者互不一致，导致产出的 is_sarcasm 不可信。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from label_sarcasm import SARCASM_RULES_PROMPT
# 上下文清理逻辑与"生成待标注文件"共用一份实现，避免两处口径不一致
# （build_pending.py 的 dry-run 报告压缩比，这里实际应用）。
from build_pending import clean_context
# 投票规则唯一实现，与离线重标脚本共用，保证新标数据与重标后的历史数据同一套定义。
from relabel_votes import final_label

# 与 label_sarcasm.py 使用相同的模型、凭据和输出文件配置。
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import (
    CLEANED_FILE,
    DASHSCOPE_API_KEY,
    LABELED_CSV_FILE,
    LABELED_FILE,
    LABEL_ERRORS_FILE,
    PENDING_FILE,
    TENCENT_TOKENHUB_API_KEY,
    THREE_MODEL_DISCARDED_FILE,
)


DEFAULT_MAX_WORKERS = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5
# 默认读 build_pending.py 产出的待标注文件，而不是 CLEANED_FILE：后者没做去重、
# 没清 OCR，直接标会为重复 content 重复付费（build_pending.py 的 dry-run 会报出
# 具体省下多少次请求）。PENDING_FILE 的每条记录自带指向 CLEANED_FILE 的
# source_line，与历史标注共用同一套编号，--resume 不会错位。
DEFAULT_INPUT_FILE = str(PENDING_FILE)
DEFAULT_OUTPUT_FILE = str(LABELED_FILE)
DEFAULT_OUTPUT_CSV = str(LABELED_CSV_FILE)
DEFAULT_ERROR_OUTPUT = str(LABEL_ERRORS_FILE)
DEFAULT_DISCARDED_OUTPUT = str(THREE_MODEL_DISCARDED_FILE)
DEFAULT_STATS_OUTPUT = str(LABELED_FILE.with_name(f"{LABELED_FILE.stem}_three_model_stats.json"))

MODELS = (
    {
        "name": "qwen",
        "label": "千问 Plus（qwen-plus）",
        "model": "qwen-plus",
        "api_key": DASHSCOPE_API_KEY,
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}',
        },
    },
    {
        "name": "deepseek_flash",
        "label": "DeepSeek V4 Flash（deepseek-v4-flash-202605）",
        "model": "deepseek-v4-flash-202605",
        "api_key": TENCENT_TOKENHUB_API_KEY,
        "api_url": "https://tokenhub.tencentmaas.com/v1/chat/completions",
        "headers": {"Content-Type": "application/json"},
    },
    {
        "name": "deepseek_pro",
        "label": "DeepSeek V4 Pro（deepseek-v4-pro-202606）",
        "model": "deepseek-v4-pro-202606",
        "api_key": TENCENT_TOKENHUB_API_KEY,
        "api_url": "https://tokenhub.tencentmaas.com/v1/chat/completions",
        "headers": {"Content-Type": "application/json"},
    },
)
LABEL_FIELDS = tuple(f"{model['name']}_label" for model in MODELS)

# 只复用三分类标注规则；输入与无理由输出格式由本脚本单独定义。
LABEL_ONLY_PROMPT = SARCASM_RULES_PROMPT + """\
# 输入格式
评论文本：待判断的评论或弹幕
原帖内容：对应上下文，可能为空；只用于理解和验证评论

# 输出要求
只输出一个合法 JSON 对象，不要输出 Markdown、代码块、解释、前后缀或多个候选结果。

字段名和字段类型必须严格如下：
{"is_sarcasm":0}
{"is_sarcasm":1}
{"is_sarcasm":2}

is_sarcasm 必须是数字 0、1 或 2。不要输出 reason 或任何额外字段。
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_messages(record: dict) -> list:
    """仅向模型提供评论与原帖上下文。

    上下文的 clean_context 清理已由调用方（process_one_record）提前做掉并写回
    record，这里直接用，不再重复清理——保证发给模型的文本与最终落盘的文本一致。
    """
    user_content = f"评论文本：{record.get('content', '')}"
    if original_post := record.get("retweeted_content", ""):
        user_content += f"\n原帖内容：{original_post}"
    return [
        {"role": "system", "content": LABEL_ONLY_PROMPT},
        {"role": "user", "content": user_content},
    ]


def call_model(model: dict, messages: list, timeout: int) -> str:
    """按既有重试策略调用一个 OpenAI 兼容接口。"""
    if not model["api_key"]:
        raise RuntimeError(f"{model['label']} 的 API Key 未设置，请检查 pipeline_config.py")

    payload = {
        "model": model["model"],
        "messages": messages,
        "temperature": 0.001,
        "top_p": 0.001,
        "max_tokens": 32,
        "stream": False,
    }
    if model["model"].startswith("deepseek-"):
        payload["thinking"] = {"type": "disabled"}

    headers = {**model["headers"], "Authorization": f"Bearer {model['api_key']}"}
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(model["api_url"], headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        except requests.exceptions.Timeout:
            last_error = "请求超时"
        except requests.exceptions.ConnectionError as exc:
            last_error = f"连接错误: {exc}"
        time.sleep(RETRY_BACKOFF * (attempt + 1))

    raise RuntimeError(f"重试 {MAX_RETRIES} 次后仍失败: {last_error}")


def parse_label(raw: str) -> int:
    """解析并校验模型返回的唯一三分类标签。"""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()).strip()
    result = json.loads(text)
    value = result.get("is_sarcasm")
    if isinstance(value, bool):
        raise ValueError("is_sarcasm 不能为布尔值")
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"is_sarcasm 不是数字标签: {value!r}") from exc
    if label not in (0, 1, 2) or str(value).strip() not in {"0", "1", "2"}:
        raise ValueError(f"is_sarcasm 必须为 0、1 或 2，实际为 {value!r}")
    return label


def process_one_record(line_no: int, record: dict, timeout: int) -> tuple:
    """依次请求三种模型，返回成功、丢弃或失败结果。"""
    try:
        # 先把上下文清理好再用，之后发给模型的和写进 labeled 的是同一份文本。
        # 原先只在 build_messages 里临时清理、不写回 record，输入是 CLEANED_FILE
        # 时（run_pipeline.py thr_label 这条路径）会导致模型判的是清理后的文本、
        # 存下来供训练的却是未清理的 OCR 原文，标注依据与训练输入错位。
        record = dict(record)
        record["retweeted_content"] = clean_context(record.get("retweeted_content", ""))

        messages = build_messages(record)
        labels = []
        for model in MODELS:
            labels.append(parse_label(call_model(model, messages, timeout)))

        # 三个模型均无法可靠判定时，记录为已处理的丢弃样本，避免续跑重复请求。
        if labels.count(2) == len(MODELS):
            return line_no, None, None, {
                "source_line": line_no,
                "discard_reason": "all_three_models_uncertain",
            }

        result = dict(record)
        result.setdefault("id", line_no)
        result["source_line"] = line_no
        result.update(dict(zip(LABEL_FIELDS, labels)))
        result["positive_votes"] = labels.count(1)
        result["is_sarcasm"] = final_label(labels)
        return line_no, result, None, None
    except Exception as exc:
        log.warning("第 %d 行处理失败：%s", line_no, exc)
        return line_no, None, str(exc), None


def iter_jsonl(filepath: str):
    """流式读取 JSONL，自动处理 UTF BOM，行号从 1 开始。"""
    with open(filepath, "rb") as file:
        bom = file.read(4)
    if bom.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encoding = "utf-32"
    elif bom.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    else:
        encoding = "utf-8-sig"

    with open(filepath, "r", encoding=encoding) as file:
        for line_no, line in enumerate(file, start=1):
            if not (line := line.strip()):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("输入文件第 %d 行 JSON 解析失败，跳过", line_no)
                continue
            # 优先用记录自带的 source_line：build_pending.py 产出的待标注文件
            # 会带上原始 CLEANED_FILE 行号。这样过滤后的文件和原始文件共用
            # 同一套编号，--resume 不会与历史标注错位。没有该字段时（直接读
            # CLEANED_FILE）回退到物理行号，行为与改动前一致。
            source_line = record.get("source_line")
            yield (source_line if isinstance(source_line, int) else line_no), record


def load_completed_lines(output_path: str) -> set[int]:
    """读取已写入的 source_line，供 --resume 跳过成功记录。"""
    if not Path(output_path).exists():
        return set()
    completed = set()
    with open(output_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                completed.add(int(json.loads(line)["source_line"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def process_chunk(chunk: list, max_workers: int, timeout: int) -> tuple:
    """并发处理多条记录；同一记录内的三种模型始终依次调用。"""
    results, errors, discarded = [], [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one_record, line_no, record, timeout): line_no
            for line_no, record in chunk
        }
        for future in as_completed(futures):
            line_no, result, error, discarded_record = future.result()
            if result is not None:
                results.append((line_no, result))
            elif error is not None:
                errors.append({"source_line": line_no, "error": error})
            else:
                discarded.append(discarded_record)
    # 显式按行号排序：原先的 sorted(results) 在两条记录 line_no 相同时会去比较
    # 后面的 dict，抛 TypeError 并让整批白跑（source_line 重复就会触发）。
    return (
        sorted(results, key=lambda item: item[0]),
        sorted(errors, key=lambda item: item["source_line"]),
        sorted(discarded, key=lambda item: item["source_line"]),
    )


def write_csv(jsonl_path: str, csv_path: str) -> None:
    """将合并后的 JSONL 转为 Excel 兼容 CSV，兼容原始记录的可变字段。

    扫两遍文件：第一遍只收集列名，第二遍边读边写。原先是把全部记录 append 进
    列表再写，百万级带 OCR 上下文的数据会吃掉数 GB 内存并在收尾阶段 OOM。
    """
    fieldnames, seen_fields, total = [], set(), 0
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            for field in row:
                if field not in seen_fields:
                    seen_fields.add(field)
                    fieldnames.append(field)
    if not total:
        log.warning("JSONL 为空，跳过 CSV 转换")
        return

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
            open(csv_path, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for line in fin:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })
    log.info("CSV 已生成：%s（%d 行）", csv_path, total)


def write_statistics(jsonl_path: str, stats_path: str) -> dict:
    """统计各模型标签、正例票数、最终标签与三模型完全一致率。"""
    model_counts = {field: Counter() for field in LABEL_FIELDS}
    final_counts, vote_counts = Counter(), Counter()
    total = all_agree = 0
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                row = json.loads(line)
                labels = [row[field] for field in LABEL_FIELDS]
            except (json.JSONDecodeError, KeyError):
                continue
            total += 1
            for field, label in zip(LABEL_FIELDS, labels):
                model_counts[field][str(label)] += 1
            final_counts[str(row["is_sarcasm"])] += 1
            vote_counts[str(row["positive_votes"])] += 1
            all_agree += len(set(labels)) == 1

    stats = {
        "total_success": total,
        "models": {field: dict(sorted(counts.items())) for field, counts in model_counts.items()},
        "positive_votes": dict(sorted(vote_counts.items())),
        "final_is_sarcasm": dict(sorted(final_counts.items())),
        "all_three_models_agree": all_agree,
        "all_three_models_agreement_rate": all_agree / total if total else 0,
        # 与 relabel_votes.final_label 保持同步，改规则时两处一起改。
        "rule": "标签 0 的数量 <= 1 且标签 1 的数量 > 0 时 is_sarcasm=1；其余为 0。"
                "标签 2（模型无法判定）既不算赞成也不算反对。",
    }
    with open(stats_path, "w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE, help="输入 JSONL 文件路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="合并后的 JSONL 输出路径")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="合并后的 CSV 输出路径")
    parser.add_argument("--error-output", default=DEFAULT_ERROR_OUTPUT, help="失败记录 JSONL 路径")
    parser.add_argument("--discarded-output", default=DEFAULT_DISCARDED_OUTPUT, help="三模型均为 2 的已丢弃记录路径")
    parser.add_argument("--stats-output", default=DEFAULT_STATS_OUTPUT, help="统计 JSON 输出路径")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="跨记录并发线程数")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="每批处理条数")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="单次请求超时秒数")
    parser.add_argument("--resume", action="store_true", help="跳过已成功或已丢弃的 source_line")
    parser.add_argument("--limit", type=int, default=0, help="最多处理 N 条未完成记录，0 表示不限制")
    args = parser.parse_args()

    # 输入不存在时明确指路，而不是让 open() 抛一个看不出该怎么办的 FileNotFoundError。
    input_path = Path(args.input)
    if not input_path.exists():
        hint = (f"先生成待标注文件：python3 data_label/build_pending.py --dry-run 看数字，"
                f"确认后去掉 --dry-run 重跑。\n（它会从 {CLEANED_FILE.name} 里剔除已标注和重复的记录）"
                if input_path == Path(PENDING_FILE) else
                f"检查 --input 路径是否正确。")
        raise SystemExit(f"输入文件不存在：{input_path}\n{hint}")

    for path in (args.output, args.output_csv, args.error_output, args.discarded_output, args.stats_output):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # 不带 --resume 时是 "w" 模式，会把已有标注整份清空。这里挡住：已标注数据是
    # 几周的时间和真金白银的 API 调用换来的，不能靠"记得加 flag"来保护。
    if not args.resume:
        for path in (args.output, args.discarded_output):
            existing = Path(path)
            if existing.exists() and existing.stat().st_size > 0:
                raise SystemExit(
                    f"拒绝执行：{existing} 已存在且非空（{existing.stat().st_size:,} 字节）。\n"
                    f"不带 --resume 会以覆盖模式打开它，已有标注将全部丢失。\n"
                    f"继续标注请加 --resume；确实要从零重标，先把该文件移走或改名。"
                )

    successful_lines = load_completed_lines(args.output) if args.resume else set()
    discarded_lines = load_completed_lines(args.discarded_output) if args.resume else set()
    completed = successful_lines | discarded_lines
    if completed:
        log.info(
            "断点续跑：跳过 %d 条已完成记录（成功 %d 条，丢弃 %d 条）",
            len(completed),
            len(successful_lines),
            len(discarded_lines),
        )

    processed = success = discarded = failed = 0
    mode = "a" if args.resume else "w"
    # 错误文件也跟随 mode：原先固定 "w"，续跑时会把上一轮的失败记录清空，
    # 排查线索就没了。续跑同一条记录若再次失败会追加一条新记录，按 source_line 看最新的即可。
    with open(args.output, mode, encoding="utf-8") as output_file, open(
        args.discarded_output, mode, encoding="utf-8"
    ) as discarded_file, open(args.error_output, mode, encoding="utf-8") as error_file:
        chunk = []
        for line_no, record in iter_jsonl(args.input):
            if line_no in completed:
                continue
            chunk.append((line_no, record))
            if len(chunk) < args.chunk_size and not (args.limit and processed + len(chunk) >= args.limit):
                continue

            results, errors, discarded_records = process_chunk(chunk, args.max_workers, args.timeout)
            for _, result in results:
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            for discarded_record in discarded_records:
                discarded_file.write(json.dumps(discarded_record, ensure_ascii=False) + "\n")
            for error in errors:
                error_file.write(json.dumps(error, ensure_ascii=False) + "\n")
            processed += len(chunk)
            success += len(results)
            discarded += len(discarded_records)
            failed += len(errors)
            output_file.flush()
            discarded_file.flush()
            error_file.flush()
            log.info("已处理 %d 条 | 成功 %d | 丢弃 %d | 失败 %d", processed, success, discarded, failed)
            chunk = []
            if args.limit and processed >= args.limit:
                break

        if chunk and (not args.limit or processed < args.limit):
            results, errors, discarded_records = process_chunk(chunk, args.max_workers, args.timeout)
            for _, result in results:
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            for discarded_record in discarded_records:
                discarded_file.write(json.dumps(discarded_record, ensure_ascii=False) + "\n")
            for error in errors:
                error_file.write(json.dumps(error, ensure_ascii=False) + "\n")
            processed += len(chunk)
            success += len(results)
            discarded += len(discarded_records)
            failed += len(errors)

    write_csv(args.output, args.output_csv)
    stats = write_statistics(args.output, args.stats_output)
    log.info("全部完成：本次处理 %d 条，成功 %d 条，丢弃 %d 条，失败 %d 条", processed, success, discarded, failed)
    log.info("各模型标签分布：%s", stats["models"])
    log.info("最终标签分布：%s", stats["final_is_sarcasm"])
    log.info("统计已写入：%s", args.stats_output)


if __name__ == "__main__":
    main()

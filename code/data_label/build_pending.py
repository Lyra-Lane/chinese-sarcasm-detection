#!/usr/bin/env python3
"""筛出"仍需标注"的记录并去重，输出待标注文件，降低三模型标注的调用成本。

背景：CLEANED_FILE 有 400 多万条，已标注约 100 万条，剩余 300 多万条如果原样
送去标注，会在两个地方白花钱：
  1. 有些 content 和已标注数据完全重复，将来 dedup_content.py 那一步反正要删掉；
  2. 有些 content 在未标注数据内部重复（水评论/模板评论，同一句话出现在不同视频下）。
本脚本在标注前把这两类去掉，只留真正需要标的记录。

关键约束（不要改动 CLEANED_FILE）：
thr_label.py 的 --resume 靠 source_line 跳过已完成记录，而 source_line 是
CLEANED_FILE 里的**物理行号**。一旦重写该文件，行号平移，已标注的 100 万条会
全部错位（一部分重复标、一部分被静默跳过，且不报错）。因此本脚本：
  - 只读 CLEANED_FILE，绝不修改它；
  - 输出的每条记录显式带上**原始 source_line**；
  - 配合 thr_label.py 里"优先用记录自带 source_line"的改动，
    让 pending 文件和原始文件共用同一套编号，--resume 继续可用。

去重口径与 dedup_content.py 保持一致：按 normalize_text(content) 判重，
忽略 retweeted_content。附带好处是同一句 content 只会得到一个标签，
不再产生此前那 208 组"同一 content 在不同上下文下被标成不同标签"的冲突。

OCR 上下文清理（clean_context）和模板水军过滤：写 pending 文件前即折叠
retweeted_content 中重复 OCR 帧，并丢弃命中已确认 emoji 或纯文字固定收尾的 content。
这样三模型不会为模板水军消耗 token；剩余清理后的文本会一路写入
sarcasm_labeled.jsonl。`remove_emoji_spam.py` 仅保留给已标注历史数据的回溯清理。

用法：
    python3 data_label/build_pending.py --dry-run   # 只报数字，不写文件（先跑这个）
    python3 data_label/build_pending.py             # 生成 PENDING_FILE
    python3 data_label/build_pending.py --selfcheck
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import (
    CLEANED_FILE,
    LABELED_FILE,
    PENDING_FILE,
    THREE_MODEL_DISCARDED_FILE,
)

# ============================================================
# 【可调参数区】
# ============================================================
# 送给标注模型的 retweeted_content 字符上限。
# 训练侧 MAX_LENGTH=384 token，下游模型永远看不到超出部分，全量发送纯属浪费；
# 2000 字远高于该预算，留足余量，只砍掉极端长尾（实测最长的一条 32767 字）。
CONTEXT_MAX_CHARS = 2000

# n-gram 新颖度阈值：一个 OCR 帧里"没见过的 4-gram"占比低于该值就丢掉。
# 逐帧 OCR 的重复内容错字各不相同，精确去重无效（实测只压到 96.5%），
# 字符 n-gram 重叠对错字不敏感（实测压到 79.8%）。
NOVELTY_THRESHOLD = 0.5

# 标注前直接丢弃已确认的模板水军，避免为它们支付三模型调用费用。
# 规则同时供政务采集和历史标注集回溯清理复用，避免口径分叉。
DEFAULT_SPAM_EMOJIS = ["😉"]
DEFAULT_SPAM_SUFFIXES = ["鼓掌"]

# ============================================================


def contains_spam_emoji(content, emojis=DEFAULT_SPAM_EMOJIS):
    """content 含任一已确认的模板水军 emoji 时返回 True。"""
    return bool(emojis) and any(emoji in str(content or "") for emoji in emojis)


def ends_with_spam_suffix(content, suffixes=DEFAULT_SPAM_SUFFIXES):
    """匹配纯文字模板收尾，不把 `[鼓掌]` 等正常表情标签误判为水军。"""
    text = str(content or "").strip()
    for suffix in suffixes or ():
        if text.endswith(suffix):
            prefix_len = len(text) - len(suffix)
            if prefix_len == 0 or text[prefix_len - 1] != "]":
                return True
    return False


def normalize_text(value):
    """与 split_data.py / dedup_content.py 的 normalize_text 保持一致：压缩多余空白。"""
    return " ".join(str(value or "").split())


def fingerprint(text):
    """content 的定长指纹，用于低内存判重。

    ponytail: 用 8 字节 blake2b 摘要而不是直接存原字符串，把 400 万条的判重集合
    从 GB 级压到百 MB 级。代价是理论上存在哈希碰撞（64 位、400 万条量级下
    碰撞概率约 1e-6，会让极少数本该保留的记录被误判为重复而少标几条），
    可接受。若将来要求零碰撞，把 digest_size 加大或改回存原字符串即可。
    """
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def iter_jsonl(path):
    """流式读 JSONL，产出 (物理行号, dict)，跳过空行和坏行。"""
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                continue


def load_completed(labeled_path, discarded_path):
    """已完成的 source_line 集合，以及已标注 content 的指纹集合。

    与 thr_label.py 的 --resume 口径一致：成功标注的和三模型均判 2 被丢弃的，
    都算已完成，不该再送去标注。
    """
    completed_lines = set()
    labeled_fps = set()
    for path, collect_content in ((labeled_path, True), (discarded_path, False)):
        if not Path(path).exists():
            continue
        for _, row in iter_jsonl(path):
            source_line = row.get("source_line")
            if isinstance(source_line, int):
                completed_lines.add(source_line)
            if collect_content:
                text = normalize_text(row.get("content", ""))
                if text:
                    labeled_fps.add(fingerprint(text))
    return completed_lines, labeled_fps


def _segments(text):
    """按 OCR 帧标记切分。抖音 OCR 会把视频逐帧识别，帧间用 IMG: / [BR] 分隔。"""
    return [p for p in (s.strip() for s in re.split(r"IMG:|\[BR\]", text)) if p]


def _grams(text, n=4):
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def clean_context(text, max_chars=CONTEXT_MAX_CHARS, threshold=NOVELTY_THRESHOLD):
    """清理上下文里重复的 OCR 帧，并按字符上限截断。

    只丢"几乎没带来新信息"的帧，保留首次出现的内容，因此不丢独有信息——
    这是标注仍然有效、不需要重标的前提。

    幂等：对已清理过的文本再跑一次结果不变（清理后不再有 IMG:/[BR] 帧标记，
    会走下面的快速路径原样返回）。
    """
    text = str(text or "")
    if not text:
        return ""
    # 快速路径：没有帧标记且未超长的文本，清理后必然等于原文（单帧的新颖度
    # 恒为 1.0，必然保留），直接返回省掉 n-gram 计算。百万级数据里绝大多数
    # 记录走这条路（实测上下文长度中位数只有几十字）。
    if len(text) <= max_chars and "IMG:" not in text and "[BR]" not in text:
        return text
    seen, kept, seen_short = set(), [], set()
    for seg in _segments(text):
        grams = _grams(seg[:max_chars])
        if not grams:
            # 帧比 n-gram 还短（如"水印x"），算不出重叠率，退化为精确判重，
            # 否则这类短帧会被无条件保留、重复十几遍。
            if seg in seen_short:
                continue
            seen_short.add(seg)
            kept.append(seg)
            continue
        if len(grams - seen) / len(grams) >= threshold:
            kept.append(seg)
            seen |= grams
    cleaned = " ".join(kept) if kept else text
    return cleaned[:max_chars]


def build_pending(cleaned_path, labeled_path, discarded_path):
    """遍历 CLEANED_FILE，产出待标注记录，同时统计各环节被丢弃的数量。

    生成器，逐条 yield，避免把 300 多万条全读进内存。
    统计信息通过 stats 字典原地更新，调用方在迭代结束后读取。
    """
    completed_lines, labeled_fps = load_completed(labeled_path, discarded_path)
    stats = {
        "already_labeled_or_discarded": len(completed_lines),
        "labeled_unique_contents": len(labeled_fps),
        "total_lines": 0,
        "skipped_completed": 0,
        "skipped_empty_content": 0,
        "dropped_template_spam": 0,
        "dropped_dup_with_labeled": 0,
        "dropped_dup_within_pending": 0,
        "pending": 0,
        "context_chars_before": 0,
        "context_chars_after": 0,
    }
    seen_fps = set()

    def _iter():
        for line_no, row in iter_jsonl(cleaned_path):
            stats["total_lines"] += 1
            if line_no in completed_lines:
                stats["skipped_completed"] += 1
                continue
            text = normalize_text(row.get("content", ""))
            if not text:
                stats["skipped_empty_content"] += 1
                continue
            if contains_spam_emoji(text) or ends_with_spam_suffix(text):
                stats["dropped_template_spam"] += 1
                continue
            fp = fingerprint(text)
            if fp in labeled_fps:
                stats["dropped_dup_with_labeled"] += 1
                continue
            if fp in seen_fps:
                stats["dropped_dup_within_pending"] += 1
                continue
            seen_fps.add(fp)
            stats["pending"] += 1
            # 其余字段原样保留，只做三件事：
            #   1. 在标注前丢弃已确认的模板水军，避免三模型调用；
            #   2. 清理 retweeted_content 里重复的 OCR 帧；
            #   3. 补上原始 source_line，供 --resume 对齐编号。
            out = dict(row)
            ctx = str(out.get("retweeted_content", "") or "")
            cleaned = clean_context(ctx)
            stats["context_chars_before"] += len(ctx)
            stats["context_chars_after"] += len(cleaned)
            out["retweeted_content"] = cleaned
            out["source_line"] = line_no
            yield out

    return _iter(), stats


def main():
    parser = argparse.ArgumentParser(description="生成去重后的待标注文件")
    parser.add_argument("--cleaned", default=str(CLEANED_FILE))
    parser.add_argument("--labeled", default=str(LABELED_FILE))
    parser.add_argument("--discarded", default=str(THREE_MODEL_DISCARDED_FILE))
    parser.add_argument("--output", default=str(PENDING_FILE))
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计并报告数字，不写输出文件")
    args = parser.parse_args()

    print("读取已完成记录（labeled + discarded）...")
    rows, stats = build_pending(args.cleaned, args.labeled, args.discarded)

    out_path = Path(args.output)
    if args.dry_run:
        for _ in rows:
            pass
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n===== 统计 =====")
    print(f"CLEANED_FILE 总行数                : {stats['total_lines']:,}")
    print(f"  已标注/已丢弃，跳过              : {stats['skipped_completed']:,}")
    print(f"  content 为空，跳过               : {stats['skipped_empty_content']:,}")
    print(f"  模板水军，标注前过滤             : {stats['dropped_template_spam']:,}")
    print(f"  与已标注数据 content 重复，丢弃  : {stats['dropped_dup_with_labeled']:,}")
    print(f"  未标注数据内部 content 重复，丢弃: {stats['dropped_dup_within_pending']:,}")
    print(f"待标注记录数                       : {stats['pending']:,}")

    saved = (stats["dropped_template_spam"] + stats["dropped_dup_with_labeled"]
             + stats["dropped_dup_within_pending"])
    remaining_before_filter = stats["pending"] + saved
    if remaining_before_filter:
        print(f"\n标注前过滤省下的调用条数: {saved:,} / {remaining_before_filter:,}"
              f"（{saved / remaining_before_filter:.1%}），"
              f"按每条 3 次模型调用算，省 {saved * 3:,} 次请求")

    before, after = stats["context_chars_before"], stats["context_chars_after"]
    if before:
        print(f"\ncontext OCR 重复清理：{before:,} 字 -> {after:,} 字"
              f"（压缩到 {after / before:.1%}，上限 {CONTEXT_MAX_CHARS} 字/条）")

    if args.dry_run:
        print("\n[dry-run] 未写任何文件。确认数字合理后去掉 --dry-run 重跑。")
    else:
        print(f"\n已写入：{out_path}")
        print("下一步（务必带 --resume，否则会清空已有标注）：")
        print(f"  python3 data_label/thr_label.py --input {out_path} --resume")


def _selfcheck():
    # clean_context：重复帧被折叠，独有信息保留
    dup_frames = "IMG:抖音号：abc[BR]IMG:抖音号：abc[BR]IMG:抖音号：abc"
    cleaned = clean_context(dup_frames)
    assert cleaned.count("抖音号：abc") == 1, cleaned

    unique_frames = "IMG:今天天气很好啊真不错[BR]IMG:完全不同的另一段文字内容"
    cleaned2 = clean_context(unique_frames)
    assert "今天天气很好啊真不错" in cleaned2 and "完全不同的另一段文字内容" in cleaned2, cleaned2

    # 比 n-gram 还短的帧也要能去重（曾漏掉这个分支）
    assert clean_context("IMG:水印x[BR]IMG:水印x").count("水印x") == 1

    # 幂等：清理过的文本再清一次结果不变
    once = clean_context(dup_frames)
    assert clean_context(once) == once

    # 字符上限生效
    assert len(clean_context("啊" * 5000, max_chars=100)) == 100
    # 短文本原样返回
    assert clean_context("很短") == "很短"
    assert clean_context("") == ""

    # fingerprint：规范化后相同的文本指纹一致，不同文本不一致
    assert fingerprint(normalize_text("你好  世界")) == fingerprint(normalize_text("你好 世界"))
    assert fingerprint("a") != fingerprint("b")

    # 模板水军规则：emoji 和纯文字收尾被过滤，[鼓掌] 表情标签保留。
    assert contains_spam_emoji("真不错呢😉")
    assert ends_with_spam_suffix("真不错呀鼓掌")
    assert not ends_with_spam_suffix("真不错呀[鼓掌]")

    # build_pending 的筛选逻辑（决定花多少钱，必须验）
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def dump(name, rows):
            p = tmp / name
            p.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8")
            return p

        # 第1行已标注、第2行已丢弃、第3行 content 与已标注重复、
        # 第5行与第4行 content 重复、第6行 content 为空；第8/9行为模板水军，
        # 第10行的 [鼓掌] 是正常表情标签，必须保留。
        cleaned = dump("cleaned.jsonl", [
            {"content": "已经标过的评论"},          # line 1 -> 已完成
            {"content": "三模型都判2被丢弃的"},      # line 2 -> 已完成
            {"content": "已经标过的评论"},          # line 3 -> 与已标注重复
            {"content": "全新的评论A"},             # line 4 -> 保留
            {"content": "全新的评论A"},             # line 5 -> 内部重复
            {"content": "   "},                    # line 6 -> 空
            {"content": "全新的评论B"},             # line 7 -> 保留
            {"content": "模板夸奖😉"},               # line 8 -> 模板水军
            {"content": "模板夸奖鼓掌"},             # line 9 -> 模板水军
            {"content": "正常评论[鼓掌]"},           # line 10 -> 保留
        ])
        labeled = dump("labeled.jsonl", [
            {"content": "已经标过的评论", "source_line": 1, "is_sarcasm": 0},
        ])
        discarded = dump("discarded.jsonl", [
            {"source_line": 2, "discard_reason": "all_three_models_uncertain"},
        ])

        rows, stats = build_pending(cleaned, labeled, discarded)
        out = list(rows)

        assert stats["total_lines"] == 10, stats
        assert stats["skipped_completed"] == 2, stats
        assert stats["dropped_dup_with_labeled"] == 1, stats
        assert stats["dropped_dup_within_pending"] == 1, stats
        assert stats["skipped_empty_content"] == 1, stats
        assert stats["dropped_template_spam"] == 2, stats
        assert stats["pending"] == 3, stats
        # 只剩第4、7、10行，且带上原始行号（不是重新编号）。
        assert [r["content"] for r in out] == ["全新的评论A", "全新的评论B", "正常评论[鼓掌]"], out
        assert [r["source_line"] for r in out] == [4, 7, 10], out

        # 输出的 retweeted_content 必须是清理后的（重复 OCR 帧被折叠）
        dirty = "IMG:抖音号：zz[BR]IMG:抖音号：zz[BR]IMG:抖音号：zz"
        cleaned2 = dump("cleaned2.jsonl", [{"content": "新评论", "retweeted_content": dirty}])
        rows2, stats2 = build_pending(cleaned2, tmp / "nope.jsonl", tmp / "nope2.jsonl")
        out2 = list(rows2)
        assert out2[0]["retweeted_content"].count("抖音号：zz") == 1, out2
        assert stats2["context_chars_before"] > stats2["context_chars_after"], stats2

    print("自检通过")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

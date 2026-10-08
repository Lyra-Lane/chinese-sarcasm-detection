# -*- coding:utf-8 -*-
# mq_read_wtype7_8_clean.py
#
# 用途：从 RocketMQ 持续读取消息，筛选 wtype=7(评论) 和 wtype=8(弹幕)，
#       清洗后统一字段结构，达到采集阈值后保存为 JSON 文件。
#
# 输出字段结构（每条记录统一）：
#   {
#     "wtype": 7 或 8,
#     "content": "评论/弹幕正文",
#     "summary": "系统摘要",
#     "retweeted_content": "被评论原作品扩展内容（wtype=7）/ 原 retweeted_title（wtype=8）",
#     "post_url": "原作品链接（wtype=7: post_url, wtype=8: retweeted_status_url）",
#     "url": "消息原始 url"
#   }
#
# 过滤规则：content 长度不足 8、retweeted_content 为空，或中文字符占 content 总字符数不足 50% 的数据会被丢弃。
#
# 修改采集阈值：只需调整项目根目录 pipeline_config.py。

import json
import os
import re
import signal
import logging
import queue
import threading
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import CLEANED_FILE, TARGET_COUNT_PER_TYPE_7, TARGET_COUNT_PER_TYPE_8

from rocketmq import (
    ConsumeStatus,
    DefaultMQPushConsumer,
    MessageListenerConcurrently,
)

# ══════════════════════════════════════════════════════════════
# 采集阈值：wtype=7 和 wtype=8 各需采集多少条有效数据
# ══════════════════════════════════════════════════════════════

# ── MQ 配置 ──
NAME_SERVER = os.environ.get("ROCKETMQ_NAME_SERVER", "")
TOPIC = os.environ.get("ROCKETMQ_TOPIC", "")
GROUP_CONSUMER = os.environ.get("ROCKETMQ_CONSUMER_GROUP", "sarcasm_research_consumer")

# ── 输出文件 ──
OUTPUT_FILE = CLEANED_FILE

# ── 进度打印间隔（秒） ──
PROGRESS_LOG_INTERVAL = 50

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── 复用 SafeMQConsumer 模式 ──

class _Listener(MessageListenerConcurrently):
    def __init__(self, fetch_queue):
        self._fq = fetch_queue

    def consume_message(self, msgs):
        for msg in msgs:
            body = msg.body
            while True:
                try:
                    self._fq.put(body, block=True, timeout=3)
                    break
                except queue.Full:
                    pass
        return ConsumeStatus.CONSUME_SUCCESS


class SafeMQConsumer:
    def __init__(self):
        self.fetch_queue = queue.Queue(maxsize=1)
        self.consumer = None
        self._consumer_stopped = threading.Event()
        self._stop_requested = threading.Event()

    def start(self):
        if not NAME_SERVER or not TOPIC:
            raise ValueError("请设置 ROCKETMQ_NAME_SERVER 和 ROCKETMQ_TOPIC")
        self.consumer = DefaultMQPushConsumer(GROUP_CONSUMER)
        self.consumer.namesrv_addr = NAME_SERVER
        self.consumer.consume_thread_num = 1
        self.consumer.consume_message_batch_max_size = 1
        self.consumer.registerMessageListener(_Listener(self.fetch_queue))
        self.consumer.subscribe(TOPIC, "*")
        self.consumer.start()
        logger.info("consumer started: %s / %s", NAME_SERVER, TOPIC)

    def fetch(self, timeout=3):
        try:
            return self.fetch_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def shutdown(self, timeout=10):
        if self.consumer:
            logger.info("shutting down consumer...")
            # ponytail: consumer.shutdown() 是底层 C 扩展的同步调用，网络异常
            # 或 broker 响应慢时可能永久阻塞。用子线程 + join 超时兜底，超时
            # 就不再等待，交给后续的 os._exit(0) 强制收尾。上限：超时后底层
            # 连接可能没有真正关闭，仅适用于进程即将退出的场景。
            t = threading.Thread(target=self.consumer.shutdown, daemon=True)
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning("consumer.shutdown() timed out after %ds, giving up waiting", timeout)
            else:
                logger.info("consumer stopped")
            self._consumer_stopped.set()

    def is_drained(self):
        return self._consumer_stopped.is_set() and self.fetch_queue.empty()

    def request_stop(self):
        self._stop_requested.set()

    @property
    def stopping(self):
        return self._stop_requested.is_set()


# ── 数据解析与清洗 ──

def parse_message(raw_body):
    """解析 MQ 消息体，返回 dict 或 None（解析失败时）"""
    if isinstance(raw_body, dict):
        return raw_body
    try:
        text = raw_body if isinstance(raw_body, str) else raw_body.decode("utf-8", "ignore")
        return json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
        logger.warning("JSON parse failed: %s", e)
        return None


def normalize_record(data):
    """
    如果 data 是目标类型(wtype=7 或 8)，返回统一结构的 dict；否则返回 None。
    过滤：content 长度不足 8、最终 retweeted_content 为空，或 content 中中文字符占比 < 50% 的数据返回 None。
    """
    wtype_raw = data.get("wtype")
    try:
        wtype = int(wtype_raw)
    except (TypeError, ValueError):
        return None

    if wtype not in (7, 8):
        return None

    content = data.get("content") or ""
    if len(content) < 8:
        return None

    # ponytail: 中文占比过滤，一行正则计数
    if not content or len(re.findall(r'[\u4e00-\u9fff]', content)) / len(content) < 0.5:
        return None

    # 极短文本过滤：token 数用字符数近似（中文1字≈1token，表情算1token）
    # 去除 [] 包裹的表情标签后统计有效 token，< 5 的直接丢弃
    # ponytail: 用字符近似而非真实 tokenizer，省去导入依赖；对中文短评已足够准确
    # 上限：emoji 序列可能被低估，如需精确过滤可换成 tokenizer.tokenize(content)
    _stripped = re.sub(r'\[.*?\]', ' ', content).strip()
    if len(_stripped.split()) + len(re.findall(r'[\u4e00-\u9fff]', _stripped)) < 5:
        return None

    if wtype == 7:
        retweeted_content = data.get("retweeted_content") or ""
        post_url = data.get("post_url") or ""
    else:  # wtype == 8
        retweeted_content = data.get("retweeted_title") or ""
        post_url = data.get("retweeted_status_url") or data.get("post_url") or ""

    if not retweeted_content.strip():
        return None

    return {
        "wtype": wtype,
        "content": content,
        "retweeted_content": retweeted_content,
        "post_url": post_url,
        "url": data.get("url") or "",
    }


def dedup_key(record):
    """优先按消息 URL 去重；URL 缺失时按清洗后的内容组合去重。"""
    if url := str(record["url"]).strip():
        return ("url", url)
    return (
        "content",
        record["wtype"],
        record["content"],
        record["retweeted_content"],
        record["post_url"],
    )


def should_stop(count7, count8):
    return count7 >= TARGET_COUNT_PER_TYPE_7 and count8 >= TARGET_COUNT_PER_TYPE_8


def load_existing(filepath):
    """读取已有清洗文件（若存在），返回按 wtype 统计的已有数量和去重键集合，供续采时累计计数和跳过重复。"""
    existing_count7 = existing_count8 = 0
    seen_keys = set()
    if not filepath.exists():
        return existing_count7, existing_count8, seen_keys

    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("已有清洗文件第 %d 行解析失败，跳过", line_no)
                continue
            seen_keys.add(dedup_key(record))
            if record.get("wtype") == 7:
                existing_count7 += 1
            elif record.get("wtype") == 8:
                existing_count8 += 1

    logger.info(
        "已有清洗数据：wtype=7 %d 条，wtype=8 %d 条，将在此基础上续采",
        existing_count7, existing_count8,
    )
    return existing_count7, existing_count8, seen_keys


def save_json(records, filepath):
    """以追加模式写入本次新采集的记录，不覆盖已有内容。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("appended %d new records to %s", len(records), filepath)


# ── 主流程 ──

def main():
    consumer = SafeMQConsumer()
    consumer.start()

    # 信号处理
    def on_signal(signum, frame):
        logger.info("received signal %d, requesting stop...", signum)
        consumer.request_stop()
        threading.Thread(target=consumer.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    existing_count7, existing_count8, seen_keys = load_existing(OUTPUT_FILE)

    results = []
    duplicate_count = 0
    # count7 / count8 是累计口径（已有 + 本次新增），用于判断是否达到采集目标。
    count7 = existing_count7
    count8 = existing_count8

    logger.info("target: wtype=7 -> %d, wtype=8 -> %d", TARGET_COUNT_PER_TYPE_7, TARGET_COUNT_PER_TYPE_8)
    logger.info("processing loop started")

    last_progress_log = time.monotonic()

    while True:
        # ponytail: 定时打印进度，不额外起线程/写文件，主循环里顺手判断即可
        now = time.monotonic()
        if now - last_progress_log >= PROGRESS_LOG_INTERVAL:
            logger.info(
                "progress: wtype=7 %d/%d, wtype=8 %d/%d",
                count7, TARGET_COUNT_PER_TYPE_7, count8, TARGET_COUNT_PER_TYPE_8,
            )
            last_progress_log = now

        if should_stop(count7, count8) and not consumer.stopping:
            logger.info("both types reached target, stopping consumer...")
            consumer.request_stop()
            # ponytail: 不在主线程同步调用 shutdown()。listener 回调线程可能
            # 正阻塞在 fetch_queue.put() 重试里（队列满、没人再 fetch），
            # 若此时同步 shutdown 会等待该回调线程退出而卡住/超时。改成后台
            # 线程 shutdown + 主循环继续 fetch 把队列排空，回调的 put 能成功
            # 返回，shutdown 才能顺利完成，再靠 is_drained() 正常跳出循环。
            threading.Thread(target=consumer.shutdown, daemon=True).start()

        raw_body = consumer.fetch(timeout=3)

        if raw_body is not None:
            data = parse_message(raw_body)
            if data is None:
                continue
            record = normalize_record(data)
            if record is None:
                continue

            record_key = dedup_key(record)
            if record_key in seen_keys:
                duplicate_count += 1
                continue
            seen_keys.add(record_key)

            wtype = record["wtype"]
            if wtype == 7 and count7 < TARGET_COUNT_PER_TYPE_7:
                results.append(record)
                count7 += 1
            elif wtype == 8 and count8 < TARGET_COUNT_PER_TYPE_8:
                results.append(record)
                count8 += 1
            # else: 该类型已满，跳过


        if consumer.is_drained():
            break

    # 兜底：consumer 可能通过信号处理路径异步 shutdown，这里确认已经关闭
    if consumer.consumer and not consumer._consumer_stopped.is_set():
        consumer.shutdown()

    # 保存结果
    save_json(results, OUTPUT_FILE)

    # 打印示例
    print("\n" + "=" * 60)
    print("示例数据：")
    for wt in (7, 8):
        sample = next((r for r in results if r["wtype"] == wt), None)
        if sample:
            print(f"\n[wtype={wt}]")
            print(json.dumps(sample, ensure_ascii=False, indent=2))

    # 打印统计（区分已有数据、本次新增和累计合计，因为输出文件是追加写入）
    new_count7 = count7 - existing_count7
    new_count8 = count8 - existing_count8
    print("\n" + "=" * 60)
    print(f"本次新增数量: {len(results)}")
    print(f"wtype=7 (评论): 已有 {existing_count7} + 新增 {new_count7} = 合计 {count7}")
    print(f"wtype=8 (弹幕): 已有 {existing_count8} + 新增 {new_count8} = 合计 {count8}")
    print(f"去重跳过: {duplicate_count}")
    print(f"输出文件: {OUTPUT_FILE}")

    # ponytail: rocketmq 底层 C 扩展对象在解释器 finalize 阶段析构会
    # 触发段错误（已知问题，与业务逻辑无关）。该做的事（保存文件、打印
    # 统计）都已经完成，直接跳过 Python 正常收尾，避免这个崩溃传导成
    # 非零退出码。上限：如果以后需要在退出前再做别的清理工作，要放在
    # 这行之前，os._exit 之后的代码不会执行。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

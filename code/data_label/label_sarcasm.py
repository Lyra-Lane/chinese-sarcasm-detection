#!/usr/bin/env python3
"""
反讽识别批量标注脚本 —— 调用大模型对 JSONL 数据逐条判断是否为反讽。

使用示例:
    python label_sarcasm.py \
        --max-workers 4 \
        --chunk-size 100 \
        --timeout 30 \
        --limit 4 \
        --resume
"""

import os
import csv
import json
import time
import re
import argparse
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import (
    CLEANED_FILE,
    DASHSCOPE_API_KEY,
    LABELED_CSV_FILE,
    LABELED_FILE,
    LABEL_ERRORS_FILE,
    TENCENT_TOKENHUB_API_KEY,
)


# ============================================================
# 可调全局参数 —— 集中放置，便于快速修改
# ============================================================
DEFAULT_MAX_WORKERS = 10       # 并发线程数
DEFAULT_CHUNK_SIZE = 200      # 每个 chunk 处理条数
DEFAULT_TIMEOUT = 30           # 单次 HTTP 请求超时(秒)
DEFAULT_INPUT_FILE = str(CLEANED_FILE)
DEFAULT_OUTPUT_FILE = str(LABELED_FILE)
DEFAULT_OUTPUT_CSV = str(LABELED_CSV_FILE)
DEFAULT_ERROR_OUTPUT = str(LABEL_ERRORS_FILE)
MAX_RETRIES = 3                # 请求重试次数
RETRY_BACKOFF = 1.5            # 重试退避基数(秒)

# ============================================================
# API 配置 —— 启动时选择模型
# ============================================================
QWEN_MODEL = {
    "label": "千问 Plus（qwen-plus）",
    "model": "qwen-plus",
    "api_key": DASHSCOPE_API_KEY,
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "headers": {
        "Content-Type": "application/json",
        "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}',
    },
}
DEEPSEEK_MODELS = {
    "2": ("DeepSeek V4 Flash（deepseek-v4-flash-202605）", "deepseek-v4-flash-202605"),
    "3": ("DeepSeek V4 Pro（deepseek-v4-pro-202606）", "deepseek-v4-pro-202606"),
}
TEMPERATURE = 0.001
TOP_P = 0.001
MAX_TOKENS = 128  # 反讽判断只需短 JSON 回答，节省 token
ACTIVE_MODEL = None

# ============================================================
# 系统 Prompt —— 反讽识别规则
# ============================================================
SARCASM_RULES_PROMPT = """\
# 角色与任务
你是中文社交媒体评论的反讽识别与话题标注专家。

你将收到“评论文本”和“原帖内容”。请判断评论属于以下哪一类：

- is_sarcasm=1：存在证据充分的反讽。
- is_sarcasm=0：能够可靠判断为不属于本任务定义的严格反话型反讽。
- is_sarcasm=2：根据评论和给定上下文，无法在标签 0 与标签 1 之间作出可靠判断。

# 核心原则

1. 判断的是评论自身是否存在“表层立场—实际立场”的明确反转。
2. 原帖只用于验证评论的真实含义，不能代替评论自身提供明确的表层立场。
3. 评论含义清楚，且真诚表达、直接表达或其他非反话解释明显更自然时标 0；如果 0 和 1 两种解释同样合理且无法排除，标 2。
4. 标签 2 是为了剔除无法可靠标注的样本，不仅限于完全无法理解的乱码。如果关键意图、立场或反转关系依赖未提供的画面、评论链或其他信息，导致 0 和 1 之间无法可靠区分，应标 2。

# 标签 1：严格的反讽

反讽是指：评论字面上对某个明确对象表达了一种清晰的评价或态度，但结合给定文本可以确认，评论者并不认同该字面立场，而是在传递与之相反或明显冲突的实际立场，以讽刺、挖苦、嘲弄或贬抑该对象。

只有同时满足以下五项，才标 is_sarcasm=1：

1. 明确的表层立场
   按照评论的通常字面含义，评论者对某个明确对象表达了清晰、可复述的评价或态度。该立场必须来自评论文本本身。
2. 明确的实际立场
   能够具体复述评论者真正表达的评价或态度，不能只说“有讽刺意味”“像在阴阳怪气”。
3. 明确的立场反转
   实际立场与表层立场相反或明显冲突。例如表面肯定、实际否定；表面感谢、实际责怪；表面关心、实际嘲弄。
4. 充分的反转证据
   评论内部或原帖中存在明确事实、结果、逻辑矛盾、荒谬情境或强语境线索，足以排除真诚的字面理解。评论与原帖观点不同、情绪不同或事实不一致，本身不构成反转证据。
5. 明确的讽刺目标
   能够指出评论在讽刺谁、什么行为、什么观点、什么产品、什么结果或什么现象。

任一项缺失、模糊、存在多种合理解释或仅为“可能”时，均不得标 1。

# 标签 0：可靠判断为不属于严格反话

当评论的基本意思可以理解，并且能够可靠判断为直接表达、真诚互动或其他非严格反话类型时，标 is_sarcasm=0。仅仅“不满足标签 1”并不必然标 0；如果无法在 0 和 1 之间作出可靠选择，应标 2。

以下情况必须标 0，不能仅凭语气或言外之意判为反讽：

1. 直接批评、否定、抱怨、辱骂、诅咒、攻击、幸灾乐祸或直接负面评价。
   例：“太差了”“多行不义必自毙”“这个活动就是骗人”。这些话可能刻薄，但字面立场与实际立场一致。

2. 真诚赞扬、祝福、感谢、安慰、鼓励、支持、认同或关心。
   原帖包含失败、辛苦、自嘲或负面内容，不足以证明评论中的正面表达是假话。

3. 评论与原帖观点、体验、事实判断或情绪不同。
   例：原能给帖说“疼惨了”，评论说“根本不疼”，可能只是在直接表达不同体验。

4. 反问、设问、暗示、影射、双关、异常陈述、刻意类比、对比、委婉否定或间接批评。
   即使这些表达可能带有挖苦或攻击性，只要不存在清晰的表层立场及其明确反转，就标 0。
5. 普通玩笑、幽默调侃、熟人互动、接梗、角色扮演、昵称、卖萌或自嘲。
   有幽默效果或攻击性不等于反话型反讽。
6. 水军、营销、控评、模板化好评或堆砌赞美。
   措辞生硬、空泛、人机感强、感叹号很多或表情密集，不证明评论者在说反话。
7. 正常提问、咨询、建议、推测、功能性表达、事实转述、广告、交易或信息补充。
8. 只有“哈哈”“笑死”“服了”“捂脸”“偷笑”“狗头”“鼓掌”等语气词或表情，但其互动功能基本清楚。
9. 原帖自身包含自嘲、反讽或攻击，但评论只是在正常回应、认同、安慰、称赞或表达不同看法。
10. 评论可以理解，且真诚字面解释明显比反话解释更自然。此时属于“反讽证据不足”，应标 0；如果两种解释同样自然且缺少区分证据，应标 2。

# 标签 2：无法可靠标注

标签 2 用于筛除依据当前输入无法在 0 与 1 之间作出可靠判断的样本。不要为了得出 0 或 1 而猜测未提供的信息。

符合以下任一情况，且原帖不能有效补全时，标 is_sarcasm=2：

1. 文本本身无法稳定理解
   如关键成分缺失、指代不明、方言或措辞严重混乱、多个残句无法恢复关系、文本被乱码或错误 OCR 破坏，因而无法稳定复述其意思。

2. 判断依赖缺失信息
   评论的真实意图依赖未提供的图片、视频画面、语音语调、上级评论、完整对话或特定背景。只根据当前文本，无法确定它是真诚表达还是反话。

3. 存在多个同样合理且会导致不同标签的解释
   真诚理解和反话理解都很自然，给定文本无法排除其中任何一种；或者连评论者在评价谁、评价什么都无法可靠确定。

4. 评论表面上存在可能的反话线索，但决定性证据不在当前输入中
   例如仅有“真厉害”“真便宜”“你真棒”等可真诚也可反话的表达，但原帖没有提供足以排除真诚理解的事实或语境。

以下情况不得标 2：

- 评论属于明确的直接批评、真诚祝福、正常提问、事实陈述、普通玩笑、自嘲、模板评论或其他可靠识别的非反话表达，应标 0。
- 评论的互动功能很清楚，只是无法确定对应的人名、地点、具体事件或事实真假，应标 0。
- 仅仅是反讽证据不足，但存在明显更自然的真诚或直接表达解释，应标 0。

标签 2 的核心不是“不能证明反讽”，而是“给定信息不足以对 0 和 1 作出可靠选择”。

# 三标签判定顺序
请在内部依次完成以下步骤，但不要输出分析过程：
## 第一步：判断是否能可靠标注
尝试用一句通顺、具体的话复述评论的字面意思、交际意图和评价对象，并检查是否有足够信息区分真诚表达与反话。
- 如果文本本身无法稳定理解，或 0 和 1 两种解释同样合理且给定信息无法区分，标 2。
- 如果能够可靠识别为直接表达、正常互动、玩笑、自嘲、模板评论或其他非反话类型，不得因为缺少更多背景而标 2，应标 0。
- 只有通过可靠性检查后，才继续第二步。

## 第二步：检查严格反话
分别写出评论的：
- 表层立场；
- 实际立场；
- 两者的反转关系；
- 排除真诚理解的文本证据；
- 讽刺目标。
五项全部明确才标 1。如果某项缺失或依赖猜测，但存在明显更自然的非反话解释，标 0；如果真诚解释和反话解释同样合理，且给定信息无法区分，标 2。

## 第三步：反证检查
在判 1 前，主动尝试构造最合理的非反话解释：真诚评价、不同观点、不同体验、普通玩笑、模板话术或正常互动。如果非反话解释明显更自然，改判 0；如果非反话解释与反话解释同样自然，且给定文本无法排除任何一种，改判 2。

# 典型正例：标 1
评论：“这服务真周到，排了三小时还没轮到。”
原帖：明确描述窗口效率低，长时间无人办理。
判定：1。
理由：表面肯定服务周到，排队事实证明实际在批评低效。

评论：“能做成这样也是一种本事。”
原帖：明确展示项目因低级错误彻底失败。
判定：1。
理由：表面肯定有本事，失败事实证明实际在嘲讽能力差。

评论：“啊对对对，你最懂。”
原帖：对方刚作出明显错误且自相矛盾的判断。
判定：1。
理由：表面附和对方，明确错误语境证明实际在否定对方。

# 典型反例：标 0
评论：“祝您生意兴隆，万事顺心。”
原帖：博主说最近生意不好做。
判定：0。
原因：可以自然理解为真诚祝愿，语境反差不能证明立场反转。

评论：“根本不疼，一点感觉都没有。”
原帖：作者说自己使用产品时很疼。
判定：0。
原因：评论可能在直接表达不同体验，评论与原帖冲突不是反讽。

评论：“新人刚进来就没了。”
原帖：宣传面向新人的活动。
判定：0。
原因：字面是在陈述结果；即使可能暗指机制有问题，也没有明确表层立场及其反转。

评论：“多行不义必自毙，上帝终于把他带走了。”
原帖：批评某人物并报道其死亡。
判定：0。
原因：评论直接表达反感和幸灾乐祸，字面立场与实际立场一致。

评论：“外观有质感，性价比很高，你在苛刻什么？”
原帖：批评该产品表现不好。
判定：0。
原因：评论者直接为产品辩护，不能把观点冲突当作反讽。

评论：“打针还这么乖，太可爱了！”
原帖：生病的宝宝打针后仍在吃东西。
判定：0。
原因：存在自然的真诚夸奖解释，原帖负面不能反转评论含义。

# 不可判定示例：标 2

评论：“那个这边弄了以后你说的又不是这样就……”
原帖：仅描述一次普通商品促销，没有人物或事件能对应评论中的指代。
判定：2。
原因：关键指代和命题残缺，结合原帖仍无法复述其意思。

评论：“这给来不的上又整那一……后头就个了”
原帖：为空。
判定：2。
原因：词序和成分严重混乱，无法形成稳定解释。

评论：“25块钱，真便宜。”
原帖：只提到某项服务，没有市场价格、服务内容或评价线索。
判定：2。
原因：真诚评价和反话评价都合理，给定信息无法区分。

# 标签 0 与标签 2 的最小差异

评论：“是不是经过我公司？”
原帖：为空。
判定：0。
原因：问题含义清楚，只是缺少回答所需的上下文。

评论：“对对对。”
原帖：为空。
判定：2。
原因：可能是真诚附和，也可能是反话否定，缺少区分两者的上下文。

评论：“他那个之后不是你说的那边就给了……”
原帖：没有相关人物、地点或前文。
判定：2。
原因：指代、事件和结果均无法确定，基本意思不可恢复。

"""

SARCASM_LABEL_OUTPUT_PROMPT = """\
# 理由要求

- reason 必须基于输入中的可见证据，不得虚构作者意图。
- 标 1 时，reason 必须同时指出表层立场和实际反转，例如：“表面夸技术好，结合翻车事实实为嘲讽”。
- 标 0 时，reason 应指出最关键的排除原因，例如：“直接批评，无立场反转”“不同体验，不能确认反讽”“真诚祝福解释同样合理”。
- 标 2 时，reason 应指出具体的不可靠原因，例如：“指代和命题残缺，原帖无法补全”“真诚与反话均可能，上下文不足”“关键画面缺失，无法可靠判断”。
- reason 不超过 30 个中文字符。

# 输入格式
评论文本：待判断的评论或弹幕
原帖内容：对应上下文，可能为空；只用于理解和验证评论

话题应根据评论与原帖共同讨论的核心事件判断。评论过短或标 2 时，优先依据原帖核心内容；如果评论和原帖都无法确定话题，选择 10。话题判断不得改变 is_sarcasm 标签。

# 输出要求

只输出一个合法 JSON 对象，不要输出 Markdown、代码块、解释、前后缀或多个候选结果。

字段名和字段类型必须严格一致：

{"is_sarcasm":1,"reason":"不超过30字中文理由"}

或

{"is_sarcasm":0,"reason":"不超过30字中文理由"}

或

{"is_sarcasm":2,"reason":"不超过30字中文理由"}

禁止输出额外字段；is_sarcasm 必须是数字 0、1 或 2。

"""

SARCASM_PROMPT = SARCASM_RULES_PROMPT + SARCASM_LABEL_OUTPUT_PROMPT


# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def select_model() -> dict:
    """启动时选择本次标注使用的模型。"""
    print("请选择标注模型：")
    print("1. 千问 Plus（qwen-plus）")
    print("2. DeepSeek V4 Flash（deepseek-v4-flash-202605）")
    print("3. DeepSeek V4 Pro（deepseek-v4-pro-202606）")

    while True:
        choice = input("输入 1、2 或 3：").strip()
        if choice == "1":
            if DASHSCOPE_API_KEY:
                return QWEN_MODEL
            print("DASHSCOPE_API_KEY 未设置，请检查 pipeline_config.py。")
        elif choice in DEEPSEEK_MODELS:
            if TENCENT_TOKENHUB_API_KEY:
                label, model = DEEPSEEK_MODELS[choice]
                return {
                    "label": label,
                    "model": model,
                    "api_key": TENCENT_TOKENHUB_API_KEY,
                    "api_url": "https://tokenhub.tencentmaas.com/v1/chat/completions",
                    "headers": {"Content-Type": "application/json"},
                }
            print("TENCENT_TOKENHUB_API_KEY 未设置，请检查 pipeline_config.py。")
        else:
            print("输入无效，请输入 1、2 或 3。")


# ============================================================
# 核心函数
# ============================================================

def build_messages(record: dict) -> list:
    """
    构造发送给大模型的 messages 列表。

    ==== 修改指南 ====
    当前策略：只送 content（核心判断对象）+ retweeted_content（上下文）。
    - 如果你想加 summary，取消下面注释即可。
    - 如果你想去掉原帖上下文，把 original_post 那段注释掉。

    字段说明:
      record["content"]            — 当前评论/弹幕文本（核心判断对象）
      record["retweeted_content"]  — 被评论/被转发的原帖内容（反讽判断的关键上下文）
      record["post_url"]           — 原帖链接（对文本分析无用，不送）
      record["url"]                — 评论链接（对文本分析无用，不送）
      record["wtype"]              — 数据类型标记（不送）
    """
    # 核心：当前评论文本
    comment_text = record.get("content", "")
    # 上下文：原帖内容（反讽判断的关键依据）
    original_post = record.get("retweeted_content", "")

    # 拼接 user 消息
    user_content = f"评论文本：{comment_text}"
    if original_post:
        user_content += f"\n原帖内容：{original_post}"

    messages = [
        {"role": "system", "content": SARCASM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages


def call_llm(messages: list, timeout: int) -> str:
    """调用启动时选定的大模型，返回原始文本输出。含重试。"""
    if ACTIVE_MODEL is None:
        raise RuntimeError("未选择标注模型")

    headers = {**ACTIVE_MODEL["headers"], "Authorization": f"Bearer {ACTIVE_MODEL['api_key']}"}
    data = {
        "model": ACTIVE_MODEL["model"],
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    if ACTIVE_MODEL["model"].startswith("deepseek-"):
        data["thinking"] = {"type": "disabled"}

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(ACTIVE_MODEL["api_url"], headers=headers, json=data, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        except requests.exceptions.Timeout:
            last_err = "请求超时"
            time.sleep(RETRY_BACKOFF * (attempt + 1))
        except requests.exceptions.ConnectionError as e:
            last_err = f"连接错误: {e}"
            time.sleep(RETRY_BACKOFF * (attempt + 1))

    raise RuntimeError(f"重试 {MAX_RETRIES} 次后仍失败: {last_err}")


def parse_model_output(raw: str) -> dict:
    """
    解析模型输出为 {"is_sarcasm": "是"/"否", "reason": "..."}。
    处理 ```json 包裹等常见非标格式。
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    result = json.loads(text)

    # 规范化 is_sarcasm 为 1/0
    val = str(result.get("is_sarcasm", "")).strip()
    # reason 截断到 30 字
    reason = str(result.get("reason", ""))[:30]
    topic = str(result.get("topic", "其他")).strip()
    return {"is_sarcasm": result["is_sarcasm"], "reason": reason, "topic": topic}


def process_one_record(line_no: int, record: dict, timeout: int) -> tuple:
    """处理单条记录，返回 (line_no, 结果或 None, 错误信息或 None)。"""
    raw_output = ""
    try:
        messages = build_messages(record)
        raw_output = call_llm(messages, timeout)
        label = parse_model_output(raw_output)
        record_with_label = {
            "id": line_no,
            "content": record.get("content", ""),
            "retweeted_content": record.get("retweeted_content", ""),
            "is_sarcasm": label["is_sarcasm"],
            "reason": label["reason"],
            "post_url": record.get("post_url", ""),
            "topic": label.get("topic", "其他"),
        }
        return (line_no, record_with_label, None)
    except json.JSONDecodeError:
        error = f"模型输出 JSON 解析失败，raw={raw_output[:80]}"
        log.warning(f"第 {line_no} 行：{error}")
        return (line_no, None, error)
    except Exception as e:
        log.warning(f"第 {line_no} 行：处理失败 {e}")
        return (line_no, None, str(e))


def iter_jsonl(filepath: str):
    """逐行流式读取 JSONL，yield (line_no, record)。line_no 从 1 开始。"""
    with open(filepath, "rb") as f:
        bom = f.read(4)
    if bom.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encoding = "utf-32"
    elif bom.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    else:
        encoding = "utf-8-sig"

    with open(filepath, "r", encoding=encoding) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield (i, json.loads(line))
            except json.JSONDecodeError:
                log.warning(f"输入文件第 {i} 行 JSON 解析失败，跳过")


def load_existing_ids(output_path: str) -> set:
    """读取已成功写出的原始行号，断点续跑时只跳过这些行。"""
    if not os.path.exists(output_path):
        return set()
    ids = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(int(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return ids


def process_chunk(chunk: list, max_workers: int, timeout: int) -> tuple:
    """并发处理一个 chunk，按 line_no 排序返回成功结果与错误记录。"""
    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one_record, line_no, record, timeout): line_no
            for (line_no, record) in chunk
        }
        for future in as_completed(futures):
            line_no, result, error = future.result()
            if result is not None:
                results.append((line_no, result))
            else:
                errors.append({"id": line_no, "error": error})

    results.sort(key=lambda x: x[0])
    errors.sort(key=lambda x: x["id"])
    return results, errors


def jsonl_to_csv(jsonl_path: str, csv_path: str):
    """将 JSONL 文件转为 CSV（utf-8-sig，Excel 兼容）。"""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        log.warning("JSONL 为空，跳过 CSV 转换")
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"CSV 已生成：{csv_path}（{len(rows)} 行）")


def main():
    parser = argparse.ArgumentParser(description="批量调用大模型进行反讽标注")
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE, help="输入 JSONL 文件路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="输出 JSONL 文件路径")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="输出 CSV 文件路径")
    parser.add_argument("--error-output", default=DEFAULT_ERROR_OUTPUT, help="失败记录 JSONL 路径")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="并发线程数")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="每批处理条数")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="单次请求超时(秒)")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已处理的行")
    parser.add_argument("--limit", type=int, default=0, help="只处理前N条（用于测试，0=不限制）")
    args = parser.parse_args()

    global ACTIVE_MODEL
    ACTIVE_MODEL = select_model()
    log.info(f"本次标注模型：{ACTIVE_MODEL['label']}")

    # 断点续跑
    completed_ids = set()
    if args.resume:
        completed_ids = load_existing_ids(args.output)
        if completed_ids:
            log.info(f"断点续跑：检测到 {len(completed_ids)} 条成功记录，将重试其余记录")

    total_processed = 0
    total_success = 0
    total_error = 0
    chunk_idx = 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.error_output).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output, "a" if args.resume else "w", encoding="utf-8")
    # 每次只保留本次仍失败的记录；成功输出在 resume 时继续追加。
    error_f = open(args.error_output, "w", encoding="utf-8")

    try:
        chunk = []
        records_seen = 0

        for line_no, record in iter_jsonl(args.input):
            records_seen += 1
            if line_no in completed_ids:
                continue

            chunk.append((line_no, record))

            if args.limit and total_processed + len(chunk) >= args.limit:
                break

            if len(chunk) >= args.chunk_size:
                chunk_idx += 1
                results, errors = process_chunk(chunk, args.max_workers, args.timeout)

                for _, result in results:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                for error in errors:
                    error_f.write(json.dumps(error, ensure_ascii=False) + "\n")
                total_success += len(results)
                total_error += len(errors)

                total_processed += len(chunk)
                out_f.flush()
                error_f.flush()

                log.info(
                    f"Chunk {chunk_idx} 完成 | "
                    f"已处理: {total_processed} | 成功: {total_success} | 失败: {total_error}"
                )
                chunk = []

                if args.limit and total_processed >= args.limit:
                    break

        # 尾块
        if chunk:
            chunk_idx += 1
            results, errors = process_chunk(chunk, args.max_workers, args.timeout)

            for _, result in results:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            for error in errors:
                error_f.write(json.dumps(error, ensure_ascii=False) + "\n")
            total_success += len(results)
            total_error += len(errors)
            total_processed += len(chunk)
            out_f.flush()
            error_f.flush()

            log.info(
                f"Chunk {chunk_idx} (尾块) 完成 | "
                f"已处理: {total_processed} | 成功: {total_success} | 失败: {total_error}"
            )

    finally:
        out_f.close()
        error_f.close()

    # ============================================================
    # 标注结束：转 CSV + 统计 + 示例输出
    # ============================================================
    output_csv = getattr(args, "output_csv")
    jsonl_to_csv(args.output, output_csv)

    log.info("=" * 50)
    log.info(f"全部完成！总处理: {total_processed} | 成功: {total_success} | 失败: {total_error}")
    log.info(f"JSONL: {args.output}")
    log.info(f"CSV:   {output_csv}")

    # 统计标签分布并打印前4条示例
    if os.path.exists(args.output):
        count_sarcasm = 0
        count_not_sarcasm = 0
        samples = []
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("is_sarcasm") == 1:
                        count_sarcasm += 1
                    else:
                        count_not_sarcasm += 1
                    if len(samples) < 4:
                        samples.append(obj)
                except json.JSONDecodeError:
                    pass

        log.info("-" * 50)
        log.info("标签统计:")
        log.info(f"  is_sarcasm=1 (反讽): {count_sarcasm} 条")
        log.info(f"  is_sarcasm=0 (非反讽): {count_not_sarcasm} 条")
        log.info(f"  合计: {count_sarcasm + count_not_sarcasm} 条")
        log.info("-" * 50)
        log.info("输出示例（前4条）:")
        for s in samples:
            log.info(f"  {json.dumps(s, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

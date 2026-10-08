"""实验记录模块：为每次反讽分类训练生成结构统一、自包含、不可覆盖的实验目录。

这个模块只负责"记录"，不改变任何训练算法、指标定义或数据处理方式。
train_cls.py 在 main() 里调用这里的函数，把配置、数据统计、指标、训练历史和
逐样本预测写进独立的实验目录，供后续实验诊断 Agent 读取和比较。

设计约束（对应任务要求）：
- 每次训练一个全新目录，时间戳冲突时追加短随机 ID，绝不覆盖历史实验。
- JSON 用 UTF-8 / ensure_ascii=False / 缩进；JSONL 每行一个合法 JSON。
- 所有写入用"临时文件 + 原子替换"，避免训练中断留下半个文件。
- 正确处理 NumPy / PyTorch 数据类型。
- 绝不保存 API Key、token、密码等敏感信息。
"""

import os
import io
import re
import json
import math
import shutil
import hashlib
import platform
import subprocess
from datetime import datetime, timezone

# numpy / torch / transformers 都是可选依赖：本模块在没有它们时也能被导入，
# 这样单元测试无需加载大模型即可运行；缺失时相关信息记为 null。
try:
    import numpy as _np
except Exception:  # pragma: no cover - 环境无 numpy
    _np = None

try:
    import torch as _torch
except Exception:  # pragma: no cover - 环境无 torch
    _torch = None


SCHEMA_VERSION = "1.0"

# 命令 / 环境里疑似敏感信息的键名，脱敏时用占位符替换其值。
_SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd|pwd)")
# 单独出现的疑似密钥字面量（如 DashScope 的 sk- 开头 key）。
_SECRET_VALUE_RE = re.compile(r"(?i)\b(sk-[A-Za-z0-9._-]{8,})")


# ---------------------------------------------------------------------------
# 序列化：NumPy / PyTorch 类型转换 + 原子写入
# ---------------------------------------------------------------------------
def to_jsonable(obj):
    """把任意对象转成 json 可序列化结构，正确处理 NumPy / PyTorch 类型。"""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        # NaN / Inf 不是合法 JSON，统一转为 None，避免生成非法文件。
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    # PyTorch 张量 / 标量
    if _torch is not None and isinstance(obj, _torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())

    # NumPy 数组 / 标量
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return to_jsonable(obj.tolist())
        if isinstance(obj, _np.generic):
            return to_jsonable(obj.item())

    # Path 等有 __fspath__ 的对象
    if hasattr(obj, "__fspath__"):
        return os.fspath(obj)

    # 兜底：尽量转字符串，绝不让序列化整体失败。
    return str(obj)


def _atomic_write(path, text):
    """临时文件 + os.replace 原子替换，避免中断留下半个文件。"""
    path = os.fspath(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, obj):
    """UTF-8 / ensure_ascii=False / 缩进，原子写入。"""
    text = json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2)
    _atomic_write(path, text)


def write_jsonl(path, rows):
    """每行一个合法 JSON 对象，UTF-8，原子写入整份文件。"""
    buf = io.StringIO()
    for row in rows:
        buf.write(json.dumps(to_jsonable(row), ensure_ascii=False))
        buf.write("\n")
    _atomic_write(path, buf.getvalue())


# ---------------------------------------------------------------------------
# 脱敏
# ---------------------------------------------------------------------------
def sanitize_command(argv):
    """把启动命令脱敏成一个字符串，去掉疑似密钥/token/密码。"""
    safe = []
    for i, tok in enumerate(argv):
        tok = str(tok)
        # 形如 --api-key=xxx / token=xxx
        if "=" in tok and _SECRET_KEY_RE.search(tok.split("=", 1)[0]):
            safe.append(tok.split("=", 1)[0] + "=***")
            continue
        # 形如 --api-key xxx（上一个参数是敏感键名）
        if i > 0 and _SECRET_KEY_RE.fullmatch(str(argv[i - 1]).lstrip("-")):
            safe.append("***")
            continue
        tok = _SECRET_VALUE_RE.sub("***", tok)
        safe.append(tok)
    return " ".join(safe)


# ---------------------------------------------------------------------------
# 实验目录
# ---------------------------------------------------------------------------
def generate_experiment_id(experiments_dir, now=None):
    """生成 exp_YYYYMMDD_HHMMSS；若目录已存在则追加短随机 ID，保证唯一。"""
    now = now or datetime.now()
    base = "exp_" + now.strftime("%Y%m%d_%H%M%S")
    candidate = base
    while os.path.exists(os.path.join(experiments_dir, candidate)):
        candidate = base + "_" + os.urandom(3).hex()
    return candidate


def create_experiment_dir(output_dir, now=None):
    """在 <output_dir>/experiments/<id>/ 下创建实验目录及子目录。

    返回一个 dict，包含各关键路径。
    """
    experiments_dir = os.path.join(os.fspath(output_dir), "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    exp_id = generate_experiment_id(experiments_dir, now=now)
    exp_dir = os.path.join(experiments_dir, exp_id)
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    best_model_dir = os.path.join(exp_dir, "best_model")
    logs_dir = os.path.join(exp_dir, "logs")
    # sync/：只放要传给 Kiro 分析的小文件，不含 predictions.jsonl / checkpoints / best_model。
    sync_dir = os.path.join(exp_dir, "sync")
    for d in (exp_dir, checkpoints_dir, best_model_dir, logs_dir, sync_dir):
        os.makedirs(d, exist_ok=True)
    return {
        "experiment_id": exp_id,
        "output_dir": os.fspath(output_dir),
        "experiments_dir": experiments_dir,
        "exp_dir": exp_dir,
        "checkpoints_dir": checkpoints_dir,
        "best_model_dir": best_model_dir,
        "logs_dir": logs_dir,
        "sync_dir": sync_dir,
        "manifest_path": os.path.join(exp_dir, "manifest.json"),
        "train_config_path": os.path.join(exp_dir, "train_config.json"),
        "data_summary_path": os.path.join(exp_dir, "data_summary.json"),
        "decision_config_path": os.path.join(exp_dir, "decision_config.json"),
        "metrics_path": os.path.join(exp_dir, "metrics.json"),
        "history_path": os.path.join(exp_dir, "history.jsonl"),
        "predictions_path": os.path.join(exp_dir, "predictions.jsonl"),
        "run_log_path": os.path.join(exp_dir, "run.log"),
        "digest_path": os.path.join(exp_dir, "analysis_digest.json"),
    }


# 需要同步给 Kiro 的小文件清单（原始文件名 -> sync/ 下的文件名，两者相同）。
# 明确排除：predictions.jsonl（900KB+ 原始逐样本数据）、checkpoints/、best_model/、logs/、run.log（体积大且多为噪音）。
SYNC_FILE_KEYS = [
    "manifest_path",
    "train_config_path",
    "data_summary_path",
    "decision_config_path",
    "metrics_path",
    "history_path",
    "digest_path",
]


def populate_sync_dir(paths):
    """把 SYNC_FILE_KEYS 里存在的文件复制一份到 sync/，供整体打包传输。

    sync/ 里不出现 predictions.jsonl、checkpoints/、best_model/ 等大文件，
    这样以后 rsync/scp 整个 sync/ 目录即可，不用逐个挑文件。
    """
    sync_dir = paths["sync_dir"]
    os.makedirs(sync_dir, exist_ok=True)
    copied = []
    for key in SYNC_FILE_KEYS:
        src = paths.get(key)
        if src and os.path.exists(src):
            dst = os.path.join(sync_dir, os.path.basename(src))
            shutil.copyfile(src, dst)
            copied.append(dst)
    return copied


class Tee:
    """把写入同时转发到原始流和 run.log 文件，捕获训练过程中的 print/日志。

    ponytail: 只复制 Python 层的写入（print / logging / tqdm 走 file 对象的部分），
    不拦截 C 扩展直接写的 fd；对诊断用的 run.log 已足够，无需重定向底层 fd。
    """

    def __init__(self, stream, log_path):
        self._stream = stream
        os.makedirs(os.path.dirname(os.fspath(log_path)) or ".", exist_ok=True)
        self._file = io.open(os.fspath(log_path), "a", encoding="utf-8", newline="\n")

    def write(self, data):
        self._stream.write(data)
        try:
            self._file.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

    def __getattr__(self, name):
        # isatty 等属性透传给原始流，保证 tqdm 等行为正常。
        return getattr(self._stream, name)


def update_latest(output_dir, experiment_id):
    """只在训练成功时调用：记录最新成功实验的 id。"""
    _atomic_write(os.path.join(os.fspath(output_dir), "latest.txt"), experiment_id + "\n")


# ---------------------------------------------------------------------------
# 环境 / 版本 / git 指纹
# ---------------------------------------------------------------------------
def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def file_sha256(path):
    """流式计算文件 SHA256，避免一次性读入大文件。"""
    try:
        h = hashlib.sha256()
        with open(os.fspath(path), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _git_commit(repo_dir):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def collect_env_info(source_files=None):
    """收集 Python / PyTorch / Transformers / CUDA 版本与设备信息。"""
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": getattr(_torch, "__version__", None) if _torch else None,
        "transformers_version": _pkg_version("transformers"),
        "cuda_version": None,
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
    }
    if _torch is not None:
        try:
            info["cuda_version"] = getattr(_torch.version, "cuda", None)
            info["cuda_available"] = bool(_torch.cuda.is_available())
            if info["cuda_available"]:
                info["device_count"] = _torch.cuda.device_count()
                info["device_names"] = [
                    _torch.cuda.get_device_name(i) for i in range(info["device_count"])
                ]
        except Exception:
            pass
    return info


def collect_code_fingerprint(repo_dir, source_files):
    """优先 git commit；没有 git 则记录关键源码文件的 SHA256。"""
    commit = _git_commit(repo_dir)
    fingerprint = {"git_commit": commit, "source_sha256": None}
    if commit is None:
        fingerprint["source_sha256"] = {
            os.path.basename(p): file_sha256(p) for p in (source_files or [])
        }
    return fingerprint


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def _now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def init_manifest(paths, task_name, argv, repo_dir, source_files):
    """训练启动时写入 status=running 的 manifest。返回 manifest dict。"""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": paths["experiment_id"],
        "task_name": task_name,
        "status": "running",
        "started_at": _now_iso(),
        "finished_at": None,
        "duration_seconds": None,
        "command": sanitize_command(argv),
        "environment": collect_env_info(),
        "code": collect_code_fingerprint(repo_dir, source_files),
        "error": None,
    }
    write_json(paths["manifest_path"], manifest)
    return manifest


def finalize_manifest(paths, manifest, status, started_at_ts, error=None):
    """训练结束时更新 manifest 状态为 completed / failed。"""
    manifest = dict(manifest)
    manifest["status"] = status
    manifest["finished_at"] = _now_iso()
    manifest["duration_seconds"] = round(_monotonic() - started_at_ts, 3)
    if error is not None:
        # 只记录异常类型和简要信息，不记录堆栈里可能出现的敏感数据。
        manifest["error"] = {
            "type": type(error).__name__,
            "message": sanitize_command([str(error)]),
        }
    write_json(paths["manifest_path"], manifest)
    return manifest


def _monotonic():
    import time
    return time.monotonic()


# ---------------------------------------------------------------------------
# train_config：记录参数覆盖后真正生效的配置
# ---------------------------------------------------------------------------
def build_train_config(training_args, runtime):
    """从真正生效的 TrainingArguments + 运行期变量构造 train_config。

    runtime 提供的字段（模型路径、max_length、标签映射、是否用上下文等）
    都取自实际生效值，而不是源码默认值。项目里不存在的配置一律记 null，不编造。
    """
    ta = training_args
    return {
        "model_name_or_path": runtime.get("model_name_or_path"),
        "tokenizer_name_or_path": runtime.get("tokenizer_name_or_path"),
        "learning_rate": getattr(ta, "learning_rate", None),
        "train_batch_size": getattr(ta, "per_device_train_batch_size", None),
        "eval_batch_size": getattr(ta, "per_device_eval_batch_size", None),
        "num_train_epochs": getattr(ta, "num_train_epochs", None),
        "weight_decay": getattr(ta, "weight_decay", None),
        "warmup_steps": getattr(ta, "warmup_steps", None),
        "warmup_ratio": getattr(ta, "warmup_ratio", None),
        "optimizer": str(getattr(ta, "optim", None)),
        "lr_scheduler_type": str(getattr(ta, "lr_scheduler_type", None)),
        "gradient_accumulation_steps": getattr(ta, "gradient_accumulation_steps", None),
        "fp16": getattr(ta, "fp16", None),
        "bf16": getattr(ta, "bf16", None),
        "label_smoothing_factor": getattr(ta, "label_smoothing_factor", None),
        "max_grad_norm": getattr(ta, "max_grad_norm", None),
        "max_length": runtime.get("max_length"),
        "seed": getattr(ta, "seed", None),
        "data_seed": getattr(ta, "data_seed", None),
        # 当前项目没有 early stopping 回调 -> null
        "early_stopping": runtime.get("early_stopping"),
        "metric_for_best_model": getattr(ta, "metric_for_best_model", None),
        "greater_is_better": getattr(ta, "greater_is_better", None),
        # 实际阈值由验证集选择后写入 decision_config.json。
        "classification_threshold": runtime.get("classification_threshold"),
        "threshold_selection": runtime.get("threshold_selection"),
        "loss_type": runtime.get("loss_type"),
        "class_weight": runtime.get("class_weight"),
        "label_mapping": runtime.get("label2id"),
        "use_context": runtime.get("use_context"),
        # content 与 context 的拼接格式（仅在 use_context 时生效）
        "context_concat_format": runtime.get("context_concat_format"),
        "truncation": runtime.get("truncation"),
        "padding": runtime.get("padding"),
        "save_total_limit": getattr(ta, "save_total_limit", None),
    }


# ---------------------------------------------------------------------------
# token 长度统计（分批、流式，避免大内存拷贝）
# ---------------------------------------------------------------------------
def _input_text(row, use_context):
    """复刻 train_cls.CustomDataset 的输入拼接逻辑，保证统计与训练一致。"""
    text = row.get("text", "") or ""
    if use_context and row.get("retweeted_content"):
        return text + "\n" + row["retweeted_content"]
    return text


def token_lengths(texts, tokenizer, batch_size=512):
    """分批计算 token 长度（不截断、不 padding），返回长度列表。"""
    lengths = []
    if tokenizer is None:
        return [None] * len(texts)
    for i in range(0, len(texts), batch_size):
        batch = [t if t is not None else "" for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, truncation=False, padding=False,
                        add_special_tokens=True)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


# ---------------------------------------------------------------------------
# data_summary
# ---------------------------------------------------------------------------
def summarize_split(rows, tokenizer, max_length, use_context, split_name,
                    file_path, split_method, seed, id2label=None):
    """统计单个数据划分。rows 为已加载的样本列表（复用训练已加载的数据）。"""
    n = len(rows)
    label_counts = {}
    empty_text = 0
    seen = set()
    duplicates = 0
    for r in rows:
        lbl = r.get("label")
        key = id2label.get(lbl, lbl) if id2label else lbl
        label_counts[key] = label_counts.get(key, 0) + 1
        if not str(r.get("text", "") or "").strip():
            empty_text += 1
        dedup_key = (r.get("text", ""), r.get("retweeted_content", ""))
        if dedup_key in seen:
            duplicates += 1
        else:
            seen.add(dedup_key)

    label_ratio = {k: (v / n if n else None) for k, v in label_counts.items()}

    # token 长度：一次流式分批，避免额外的大列表长期驻留。
    lengths = token_lengths([_input_text(r, use_context) for r in rows], tokenizer)
    valid = [l for l in lengths if l is not None]
    avg_len = (sum(valid) / len(valid)) if valid else None
    over_max = sum(1 for l in valid if l > max_length) if max_length else 0
    # truncation=True 且按 max_length 截断，实际被截断的样本即超过 max_length 的样本。
    truncated = over_max

    return {
        "split": split_name,
        "file": os.fspath(file_path) if file_path else None,
        "file_sha256": file_sha256(file_path) if file_path else None,
        "num_samples": n,
        "label_counts": {str(k): v for k, v in label_counts.items()},
        "label_ratio": {str(k): v for k, v in label_ratio.items()},
        "split_method": split_method,
        "split_seed": seed,
        "empty_text_count": empty_text,
        "duplicate_count": duplicates,
        "avg_token_length": avg_len,
        "max_length": max_length,
        "over_max_length_count": over_max,
        "over_max_length_ratio": (over_max / len(valid)) if valid else None,
        "truncated_count": truncated,
        "truncated_ratio": (truncated / len(valid)) if valid else None,
    }


# ---------------------------------------------------------------------------
# softmax / 预测判定
# ---------------------------------------------------------------------------
def softmax(logits):
    """对最后一维做数值稳定的 softmax。接受 numpy 数组或嵌套 list。"""
    if _np is not None:
        arr = _np.asarray(logits, dtype=_np.float64)
        arr = arr - arr.max(axis=-1, keepdims=True)
        exp = _np.exp(arr)
        return exp / exp.sum(axis=-1, keepdims=True)
    # 纯 python 兜底（单条）
    m = max(logits)
    exp = [math.exp(x - m) for x in logits]
    s = sum(exp)
    return [e / s for e in exp]


def error_type(true_label, predicted_label, positive_label=1):
    """二分类混淆判定：TP/TN/FP/FN。"""
    t = int(true_label) == positive_label
    p = int(predicted_label) == positive_label
    if t and p:
        return "TP"
    if not t and not p:
        return "TN"
    if not t and p:
        return "FP"
    return "FN"


# ---------------------------------------------------------------------------
# metrics.json：统一核心指标命名，同时保留原始 HF 指标
# ---------------------------------------------------------------------------
_CORE_KEYS = [
    ("loss", "loss"),
    ("accuracy", "accuracy"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("f1", "f1"),
    ("macro_f1", "macro_f1"),
    ("sarcasm_precision", "sarcasm_precision"),
    ("sarcasm_recall", "sarcasm_recall"),
    ("sarcasm_f1", "sarcasm_f1"),
    ("average_precision", "average_precision"),
    ("business_score", "business_score"),
    ("target_reached", "target_reached"),
    ("precision_at_recall_80", "precision_at_recall_80"),
    ("recall_at_precision_60", "recall_at_precision_60"),
    ("threshold", "threshold"),
    ("false_positive_rate", "false_positive_rate"),
    ("specificity", "specificity"),
    ("predicted_positive_ratio", "predicted_positive_ratio"),
    ("tn", "tn"), ("fp", "fp"), ("fn", "fn"), ("tp", "tp"),
]


def _unify(raw):
    """把 HF 的 eval_* / train_* 前缀指标映射为统一核心指标名。"""
    out = {}
    stripped = {}
    for k, v in (raw or {}).items():
        key = k
        for pref in ("eval_", "train_", "test_"):
            if key.startswith(pref):
                key = key[len(pref):]
                break
        stripped[key] = v
    for core, src in _CORE_KEYS:
        if src in stripped:
            out[core] = stripped[src]
    return out


def build_metrics(train_raw, val_raw, test_raw, primary_metric_name,
                  best_epoch, best_checkpoint):
    """构造统一 metrics.json，含 train/validation/test 三段与原始指标。"""
    val_core = _unify(val_raw)
    return {
        "primary_metric": {
            "name": primary_metric_name,
            "value": val_core.get(_strip_prefix(primary_metric_name)),
        },
        "best_epoch": best_epoch,
        "best_checkpoint": best_checkpoint,
        "train": _unify(train_raw),
        "validation": val_core,
        "test": _unify(test_raw),
        "raw": {
            "train": train_raw or {},
            "validation": val_raw or {},
            "test": test_raw or {},
        },
    }


def _strip_prefix(name):
    if not name:
        return name
    for pref in ("eval_", "train_", "test_"):
        if name.startswith(pref):
            return name[len(pref):]
    return name


# ---------------------------------------------------------------------------
# history.jsonl：从 trainer.state.log_history 安全转换
# ---------------------------------------------------------------------------
def build_history_rows(log_history):
    """把 Trainer 的 log_history 转成统一 history 行。

    log_history 里没有时间戳字段 -> timestamp 记 null（不编造）。
    训练日志有 loss/learning_rate；验证日志有 eval_*。
    """
    rows = []
    for entry in log_history or []:
        row = {
            "timestamp": None,  # trainer_state 不含时间戳
            "epoch": entry.get("epoch"),
            "step": entry.get("step"),
            "train_loss": entry.get("loss"),
            "eval_loss": entry.get("eval_loss"),
            "learning_rate": entry.get("learning_rate"),
            "accuracy": entry.get("eval_accuracy"),
            "macro_f1": entry.get("eval_macro_f1"),
            "sarcasm_precision": entry.get("eval_sarcasm_precision"),
            "sarcasm_recall": entry.get("eval_sarcasm_recall"),
            "sarcasm_f1": entry.get("eval_sarcasm_f1"),
            "average_precision": entry.get("eval_average_precision"),
            "business_score": entry.get("eval_business_score"),
            "target_reached": entry.get("eval_target_reached"),
            "precision_at_recall_80": entry.get("eval_precision_at_recall_80"),
            "recall_at_precision_60": entry.get("eval_recall_at_precision_60"),
            "selected_threshold": entry.get("eval_threshold"),
            "false_positive_rate": entry.get("eval_false_positive_rate"),
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# predictions.jsonl
# ---------------------------------------------------------------------------
def build_prediction_rows(rows, logits, label_ids, split, tokenizer,
                          max_length, use_context, id2label=None,
                          classification_threshold=None):
    """由原始 logits 经 softmax 得到概率，逐样本构造预测记录。

    数据集没有原生 id 字段 -> 用划分内位置索引作为合成 id，并在此说明。
    context 只在 use_context 时取 retweeted_content，否则记 null（模型实际未用）。
    """
    probs = softmax(logits)
    if _np is not None:
        probs = _np.asarray(probs)
        preds = (
            (probs[:, 1] >= classification_threshold).astype(int)
            if classification_threshold is not None else probs.argmax(axis=-1)
        )
    else:  # pragma: no cover
        preds = (
            [int(p[1] >= classification_threshold) for p in probs]
            if classification_threshold is not None
            else [int(max(range(len(p)), key=lambda i: p[i])) for p in probs]
        )

    contents = [r.get("text", "") for r in rows]
    contexts_used = [
        (r.get("retweeted_content") if (use_context and r.get("retweeted_content")) else None)
        for r in rows
    ]
    content_lens = token_lengths(contents, tokenizer)
    # 只对实际使用了 context 的样本计算 context token 长度
    context_lens = token_lengths(
        [c if c is not None else "" for c in contexts_used], tokenizer
    )
    total_lens = token_lengths([_input_text(r, use_context) for r in rows], tokenizer)

    out = []
    for i, r in enumerate(rows):
        pred = int(preds[i])
        true = int(label_ids[i]) if label_ids is not None else None
        prob_row = probs[i]
        prob_map = {
            (str(id2label.get(c, c)) if id2label else str(c)): float(prob_row[c])
            for c in range(len(prob_row))
        }
        ctx_used = contexts_used[i] is not None
        total_len = total_lens[i]
        out.append({
            "id": i,  # 合成 id：数据集无原生 id，用划分内位置索引
            "split": split,
            "content": contents[i],
            "context": contexts_used[i],  # null 表示模型本次未使用上下文
            "true_label": true,
            "predicted_label": pred,
            "classification_threshold": classification_threshold,
            "probabilities": prob_map,
            "is_correct": (true == pred) if true is not None else None,
            "error_type": error_type(true, pred) if true is not None else None,
            "content_token_length": content_lens[i],
            "context_token_length": (context_lens[i] if ctx_used else None),
            "total_token_length": total_len,
            "was_truncated": (bool(total_len > max_length)
                              if (total_len is not None and max_length) else None),
        })
    return out


# ---------------------------------------------------------------------------
# analysis_digest.json：从 pred_rows（内存中，避免重新读大文件）提炼出
# 适合喂给大模型分析的小摘要——聚合统计 + 少量典型错误样本，而不是全量数据。
# ---------------------------------------------------------------------------
_LENGTH_BUCKETS = [(0, 50), (50, 150), (150, 256), (256, None)]


def _bucket_label(lo, hi):
    return f"{lo}-{hi}" if hi is not None else f"{lo}+"


def _error_stats_by_length(rows):
    """按 total_token_length 分桶统计样本数、错误数、错误率。"""
    buckets = []
    for lo, hi in _LENGTH_BUCKETS:
        in_bucket = [
            r for r in rows
            if r.get("total_token_length") is not None
            and r["total_token_length"] >= lo
            and (hi is None or r["total_token_length"] < hi)
        ]
        n = len(in_bucket)
        wrong = sum(1 for r in in_bucket if r.get("is_correct") is False)
        buckets.append({
            "range": _bucket_label(lo, hi),
            "num_samples": n,
            "num_errors": wrong,
            "error_rate": (wrong / n) if n else None,
        })
    return buckets


def _confusion_counts(rows):
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for r in rows:
        et_ = r.get("error_type")
        if et_ in counts:
            counts[et_] += 1
    return counts


def _top_confident_wrong(rows, error_type_name, top_k, id2label=None):
    """挑出模型"很自信但判错"的典型样本：按预测类别概率从高到低排序。

    只取 content/context/概率/token 长度等字段，不带整份 predictions 记录，
    控制单条摘要样本的体量。
    """
    wrong = [r for r in rows if r.get("error_type") == error_type_name]

    def _pred_prob(r):
        pred_key = str(id2label.get(r["predicted_label"], r["predicted_label"])) \
            if id2label else str(r["predicted_label"])
        return r.get("probabilities", {}).get(pred_key, 0.0)

    wrong_sorted = sorted(wrong, key=_pred_prob, reverse=True)[:top_k]
    return [
        {
            "id": r["id"],
            "split": r["split"],
            "content": r["content"],
            "context": r.get("context"),
            "true_label": r["true_label"],
            "predicted_label": r["predicted_label"],
            "confidence": _pred_prob(r),
            "total_token_length": r.get("total_token_length"),
            "was_truncated": r.get("was_truncated"),
        }
        for r in wrong_sorted
    ]


def build_analysis_digest(pred_rows, metrics, top_k=20, id2label=None):
    """从内存中的 pred_rows + metrics 构造精炼摘要，供直接喂大模型分析。

    体量控制在几十 KB 量级（而不是 predictions.jsonl 的近 1MB），
    只保留：整体指标引用、按 split 的错误分布、按长度分桶的错误率、
    以及 FN/FP 中"模型最自信却判错"的 Top-K 典型样本。
    """
    by_split = {}
    for split in sorted({r["split"] for r in pred_rows}):
        split_rows = [r for r in pred_rows if r["split"] == split]
        by_split[split] = {
            "num_samples": len(split_rows),
            "confusion": _confusion_counts(split_rows),
            "error_rate_by_length": _error_stats_by_length(split_rows),
            "top_confident_false_negatives": _top_confident_wrong(
                split_rows, "FN", top_k, id2label),
            "top_confident_false_positives": _top_confident_wrong(
                split_rows, "FP", top_k, id2label),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "primary_metric": metrics.get("primary_metric") if metrics else None,
        "validation_core_metrics": metrics.get("validation") if metrics else None,
        "test_core_metrics": metrics.get("test") if metrics else None,
        "by_split": by_split,
    }

"""专栏续读风险：按读者隔离训练与留出验证，训练集拟合标准化。"""
import hashlib
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["sessions_14d", "completion_pct", "days_inactive", "issues_read"]
LABELS = ["14日访问次数", "阅读完成率", "连续未读天数", "已读期数"]
SUCCESS_LABELS = {1: "已续读", 0: "未续读"}

SPLIT_SEED = 42
SPLIT_FRACTION = 0.25
SPLIT_SALT = "reader-risk-reader-group-v1"
MIN_SAMPLES = 40
MIN_CLASS_SAMPLES = 8


def metadata():
    return {
        "name": "专栏续读风险试算", "short": "reader-risk",
        "tagline": "用行为窗口解释续读风险，再试算编辑关怀的覆盖边界。",
        "client": "纸岸知识出版社 · 6人内容运营组 · 320名匿名年度专栏读者",
        "workflow": "编辑每周合并阅读日志与到期续读结果，在表格中凭访问次数选择关怀对象。",
        "pains": ["各行为量纲不同，人工排序偏向高频用户", "直接使用到期后的数据会泄漏结果", "同一读者多周窗口若按行划分，会把读者同时放进训练与验证"],
        "upstream": "到期前14日阅读行为快照 reader_window_202611.csv",
        "downstream": "主编据验证集覆盖人数安排内容回访实验，不自动触达读者",
        "entities": ["读者行为样本", "续读风险试验", "编辑复核简报"],
        "dataset_name": "纸岸专栏到期前14日匿名样本",
        "columns": [{"key": "reader_id", "label": "匿名读者", "type": "string"}] + [{"key": k, "label": l, "type": "number"} for k, l in zip(FEATURES, LABELS)] + [{"key": "renewed", "label": "实际续读(1/0)", "type": "number"}],
        "parameters": [{"key": "threshold", "label": "高风险概率阈值", "type": "number", "default": 0.5, "min": 0.1, "max": 0.9}, {"key": "regularization", "label": "模型C（越小正则越强）", "type": "number", "default": 1.0, "min": 0.01, "max": 10}],
        "workbench_title": "续读风险阈值试验", "action_label": "训练并复核覆盖范围", "unit": "位读者",
        "accent": "oklch(53% 0.12 30)", "style": "editorial-press",
        "seed_story": ["08:40 编辑导入 reader_window_202611.csv 的320条匿名样本", "08:45 检查14日访问、完成率、未读天数和已读期数", "09:00 以0.5风险阈值创建一次续读试验", "09:01 系统按 reader_id 隔离分组，固定留出约25%读者做验证", "09:05 编辑比较验证AUC、召回率与进入复核的读者人数", "09:10 将阈值改为0.7，复核名单缩小，分组与风险概率保持不变", "09:20 下载含分组与留出读者的验证明细交主编", "10:00 管理员设置周一08:30内部复核简报"],
        "method": "先按 reader_id 去除完全重复记录，并拒绝同一读者续读标签冲突；随后按 reader_id 稳定哈希（种子42）排序、按续读标签分层留出约25%读者，同一读者全部样本只进入训练或验证一侧。仅在训练侧拟合 StandardScaler 与 L2 logistic 回归，目标为 1-renewed；阈值只改变留出侧复核决策，不改变分组与风险概率。",
        "limitations": ["模拟读者行为，不能当作真实留存效果", "相关系数与模型系数不代表因果", "同一读者多周窗口彼此相关，验证AUC按样本计算，不能视为独立读者数", "未包含时间外验证或干预效果评估"],
    }


def seed_records():
    rng = np.random.default_rng(1008)
    rows = []
    for i in range(320):
        sessions = int(rng.poisson(6))
        completion = float(np.clip(rng.normal(62, 21), 4, 99))
        inactive = int(rng.integers(0, 15))
        issues = int(rng.integers(1, 25))
        logit = -0.5 - 0.20 * sessions - 0.025 * (completion - 50) + 0.27 * inactive - 0.035 * issues
        risk = 1 / (1 + math.exp(-logit))
        rows.append({"reader_id": f"RD-{i+1:04}", "sessions_14d": sessions, "completion_pct": round(completion, 1), "days_inactive": inactive, "issues_read": issues, "renewed": int(rng.random() >= risk)})
    return rows


def _field_label(key):
    if key == "reader_id":
        return "匿名读者"
    if key == "renewed":
        return "实际续读(1/0)"
    return LABELS[FEATURES.index(key)]


def _missing(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _reader_hash(reader_id):
    payload = f"{SPLIT_SALT}:{SPLIT_SEED}:{reader_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _raise_reader_conflicts(labels):
    conflicts = sorted(reader for reader, values in labels.items() if len(values) > 1)
    if not conflicts:
        return
    preview = "、".join(conflicts[:5])
    suffix = f"（共{len(conflicts)}位读者）" if len(conflicts) > 1 else ""
    raise ValueError(
        f"读者续读标签冲突：{preview} 同时出现已续读与未续读{suffix}；"
        "同一读者的续读结果必须一致，请核对后续读标签后重新接入。"
    )


def _collect_reader_labels(records):
    labels = {}
    for row in records or []:
        if not isinstance(row, dict):
            continue
        reader = row.get("reader_id")
        if _missing(reader):
            continue
        value = row.get("renewed")
        if isinstance(value, bool):
            continue
        try:
            renewed = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(renewed) or renewed not in (0.0, 1.0):
            continue
        labels.setdefault(str(reader).strip(), set()).add(int(renewed))
    return labels


def reader_label_conflicts(records):
    """Reject a reader whose merged weekly windows carry both renewal labels."""
    _raise_reader_conflicts(_collect_reader_labels(records))


def prepare_samples(records):
    """Normalise, de-duplicate and validate rows before any split or fit."""
    if not isinstance(records, (list, tuple)):
        raise ValueError("行为记录须为数组。")
    samples = []
    seen = set()
    duplicates = 0
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"第 {index+1} 行必须为对象。")
        for key in ["reader_id"] + FEATURES + ["renewed"]:
            if _missing(row.get(key)):
                raise ValueError(f"第 {index+1} 行缺少字段 {_field_label(key)}")
        reader = str(row["reader_id"]).strip()
        if not reader:
            raise ValueError(f"第 {index+1} 行缺少字段 匿名读者")
        if any(isinstance(row.get(key), bool) for key in FEATURES + ["renewed"]):
            raise ValueError("行为字段与续读标签必须为数值。")
        try:
            values = tuple(float(row[key]) for key in FEATURES)
            renewed = float(row["renewed"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("行为字段与续读标签必须完整且为数值。") from exc
        if not all(math.isfinite(value) for value in values) or not math.isfinite(renewed):
            raise ValueError("行为值必须非负且有限，完成率0~100，续读标签只能为0/1。")
        if renewed not in (0.0, 1.0) or any(value < 0 for value in values) or values[1] > 100:
            raise ValueError("行为值必须非负且有限，完成率0~100，续读标签只能为0/1。")
        key = (reader,) + values + (renewed,)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        sample = {"reader_id": reader, "renewed": int(renewed)}
        sample.update({name: value for name, value in zip(FEATURES, values)})
        samples.append(sample)

    labels = _collect_reader_labels(samples)
    _raise_reader_conflicts(labels)
    if len(samples) < MIN_SAMPLES:
        raise ValueError("至少需要40条完整样本，以保留独立验证集。")
    target = np.asarray([1 - sample["renewed"] for sample in samples])
    counts = np.bincount(target, minlength=2)
    if counts.min() < MIN_CLASS_SAMPLES:
        raise ValueError("续读与未续读两类各至少8条样本。")
    return samples, duplicates


def _split_readers(samples):
    """Deterministic, label-stratified reader split; all windows of a reader stay together."""
    reader_labels = {}
    for sample in samples:
        reader_labels.setdefault(sample["reader_id"], sample["renewed"])
    validation = set()
    for renewed in (0, 1):
        candidates = sorted(
            (reader for reader, label in reader_labels.items() if label == renewed),
            key=lambda reader: (_reader_hash(reader), reader),
        )
        if len(candidates) < 2:
            side = SUCCESS_LABELS[renewed]
            raise ValueError(f"按读者隔离验证时，{side}读者至少需要2位不同读者；当前只有{len(candidates)}位。")
        count = max(1, min(len(candidates) - 1, int(len(candidates) * SPLIT_FRACTION + 0.5)))
        validation.update(candidates[:count])
    return set(reader_labels) - validation, validation


def _group_stats(samples, indices):
    selected = [samples[index] for index in indices]
    return {
        "rows": len(selected),
        "readers": len({sample["reader_id"] for sample in selected}),
        "renewed": int(sum(sample["renewed"] for sample in selected)),
        "not_renewed": int(sum(1 - sample["renewed"] for sample in selected)),
    }


def analyze(records, params):
    try:
        threshold = float(params.get("threshold", 0.5))
        c_value = float(params.get("regularization", 1.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("风险阈值与模型C须为有效数值。") from exc
    if not 0.1 <= threshold <= 0.9 or not 0.01 <= c_value <= 10:
        raise ValueError("风险阈值须为0.1~0.9，模型C须为0.01~10。")

    samples, duplicates = prepare_samples(records)
    train_readers, validation_readers = _split_readers(samples)

    ordered = sorted(
        samples,
        key=lambda sample: (sample["reader_id"],) + tuple(sample[key] for key in FEATURES) + (sample["renewed"],),
    )
    x = np.asarray([[sample[key] for key in FEATURES] for sample in ordered], dtype=float)
    y = np.asarray([1 - sample["renewed"] for sample in ordered], dtype=int)
    in_validation = np.asarray([sample["reader_id"] in validation_readers for sample in ordered], dtype=bool)
    train = np.where(~in_validation)[0]
    valid = np.where(in_validation)[0]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=1000, random_state=42, solver="lbfgs"),
    )
    model.fit(x[train], y[train])
    probability = model.predict_proba(x)[:, 1]
    prediction = (probability[valid] >= threshold).astype(int)

    auc = float(roc_auc_score(y[valid], probability[valid]))
    precision = float(precision_score(y[valid], prediction, zero_division=0))
    recall = float(recall_score(y[valid], prediction, zero_division=0))
    accuracy = float(accuracy_score(y[valid], prediction))
    tn, fp, fn, tp = [int(value) for value in confusion_matrix(y[valid], prediction, labels=[0, 1]).ravel()]
    coefficients = model[-1].coef_[0]

    review_readers = sorted({
        ordered[index]["reader_id"]
        for index in valid
        if probability[index] >= threshold
    })
    train_stats = _group_stats(ordered, train)
    valid_stats = _group_stats(ordered, valid)

    rows = []
    for index in range(len(ordered)):
        sample = ordered[index]
        in_holdout = bool(in_validation[index])
        risk = float(probability[index])
        flagged = in_holdout and risk >= threshold
        if flagged:
            decision = "建议编辑复核"
        elif in_holdout:
            decision = "维持观察"
        else:
            decision = "训练样本 · 不参与复核"
        rows.append((
            0 if in_holdout else 1,
            -risk,
            sample["reader_id"],
            tuple(sample[key] for key in FEATURES),
            {
                "reader_id": sample["reader_id"],
                "split": "留出验证" if in_holdout else "训练",
                "risk_pct": round(risk * 100, 2),
                "decision": decision,
                "actual": SUCCESS_LABELS[sample["renewed"]],
            },
        ))
    rows.sort(key=lambda item: item[:4])

    strongest = int(np.argmax(np.abs(coefficients)))
    split = {
        "basis": f"reader_id · 稳定哈希 · 种子{SPLIT_SEED} · 按续读标签分层 · 约25%读者留出 · 同读者同侧",
        "method": "按 reader_id 的稳定哈希排序，并按读者续读标签分层留出约25%读者；同一读者的全部周窗口只进入训练或验证一侧，行序与完全重复行不影响分组。",
        "seed": SPLIT_SEED,
        "validation_fraction": SPLIT_FRACTION,
        "input_rows": len(records),
        "unique_rows": len(samples),
        "duplicate_rows_ignored": duplicates,
        "train_rows": train_stats["rows"],
        "validation_rows": valid_stats["rows"],
        "train_readers": train_stats["readers"],
        "validation_readers": valid_stats["readers"],
        "train": train_stats,
        "validation": valid_stats,
        "holdout_readers": sorted(validation_readers),
    }
    summary = (
        f"去重后{len(samples)}条样本 / {len(train_readers) + len(validation_readers)}位读者"
        f"（忽略{duplicates}条完全重复记录）；训练{train_stats['rows']}条/{train_stats['readers']}位，"
        f"留出验证{valid_stats['rows']}条/{valid_stats['readers']}位；阈值{threshold:.2f}下"
        f"{len(review_readers)}位留出读者进入编辑复核（{int(prediction.sum())}条验证样本）。"
    )
    return {
        "summary": summary,
        "metrics": [
            {"label": "验证AUC", "value": round(auc, 4)},
            {"label": "风险召回率", "value": round(recall * 100, 2), "unit": "%"},
            {"label": "风险精确率", "value": round(precision * 100, 2), "unit": "%"},
            {"label": "验证集高风险", "value": len(review_readers), "unit": "人"},
            {"label": "训练读者", "value": train_stats["readers"], "unit": "人"},
            {"label": "留出验证读者", "value": valid_stats["readers"], "unit": "人"},
        ],
        "split": split,
        "review": {"threshold": threshold, "readers": review_readers, "rows": int(prediction.sum())},
        "table": {
            "columns": [
                {"key": "reader_id", "label": "匿名读者"},
                {"key": "split", "label": "数据分组"},
                {"key": "risk_pct", "label": "风险概率(%)"},
                {"key": "decision", "label": "阈值决策"},
                {"key": "actual", "label": "实际续读结果"},
            ],
            "rows": [item[4] for item in rows],
        },
        "chart": {"type": "bar", "labels": LABELS, "series": [{"name": "标准化log-odds系数", "values": [round(float(value), 4) for value in coefficients]}]},
        "findings": [
            f"按读者隔离分组：训练{train_stats['rows']}条/{train_stats['readers']}位读者，留出验证{valid_stats['rows']}条/{valid_stats['readers']}位读者；{duplicates}条完全重复记录已忽略。",
            f"验证混淆矩阵：命中未续读{tp}、误报{fp}、漏报{fn}、正确排除{tn}。",
            f"绝对系数最大的特征为{LABELS[strongest]}（{float(coefficients[strongest]):.3f}）；正值意味着更高未续读风险。",
            f"验证准确率{accuracy*100:.1f}%；阈值只改变留出侧决策，分组与风险概率沿用同一组读者划分。",
        ],
        "method": metadata()["method"],
        "warnings": [
            "风险排序供编辑复核使用，不能据此断言某项行为导致未续读。",
            "AUC与精确率/召回率按验证样本计算；同一读者的多周窗口彼此相关，复核人数按读者去重。",
        ],
    }

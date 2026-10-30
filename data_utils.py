# ------------- ✦✦✦ IMPS ✦✦✦ -------------
import collections
import csv
import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ------------- ✦✦✦ CONSTANTS ✦✦✦ -------------
SEED = 42

# ------------- ✦ AVAILABLE CLASSES ✦ -------------
CLASS_NAMES = ["Уязвимость кода", "Уязвимость архитектуры", "Уязвимость многофакторная"]
E2E_NAMES = ["Не уязвимость"] + CLASS_NAMES
NO_CLASS = -100


# ------------- ✦✦✦ HELPERS ✦✦✦ -------------
# ------------- ✦✦ TEXT NORMALIZATION ✦✦ -------------
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def near_dup_key(t: str) -> str:
    t = t.lower().replace("ё", "е")
    t = re.sub(r"\d+", "0", t)
    t = re.sub(r"[^\w]+", " ", t)
    return " ".join(t.split())


# ------------- ✦✦✦ DATA ✦✦✦ -------------
# ------------- ✦✦ LOAD AND CLEAN DATA ✦✦ -------------
def load_data(vul_csv_file, non_vul_file):
    # ------------- ✦ READ CSV ✦ -------------
    raw = Path(vul_csv_file).read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(raw), delimiter=";"))
    header, rows = rows[0], rows[1:]
    rep = collections.Counter(csv_records=len(rows))
    quarantine, pos = [], []

    # ------------- ✦ POSITIVE EXAMPLES [vul] ✦ -------------
    for i, r in enumerate(rows, start=2):
        text = normalize_text(r[0]) if r else ""
        label = r[1].strip() if len(r) > 1 else ""
        if not text and not label:
            rep["empty_records"] += 1
            continue
        rec = {"source": Path(vul_csv_file).name, "record": i, "text": text, "label": label}
        if not text:
            quarantine.append({**rec, "reason": "нет текста"})
        elif label not in CLASS_NAMES:
            quarantine.append({**rec, "reason": "нет метки" if not label else "неизвестная метка"})
        else:
            pos.append(rec)

    # ------------- ✦ CONFLICTING LABELS AND DUPLICATES ✦ -------------
    groups = collections.defaultdict(set)
    for p in pos:
        groups[near_dup_key(p["text"])].add(p["label"])
    records, seen = [], set()
    for p in pos:
        if len(groups[near_dup_key(p["text"])]) > 1:
            quarantine.append({**p, "reason": "противоречивые метки"})
            continue
        if p["text"] in seen:
            rep["duplicates_removed"] += 1
            continue
        seen.add(p["text"])
        records.append({"text": p["text"], "is_vul": 1, "vul_class": CLASS_NAMES.index(p["label"]), "source": "bdu"})

    # ------------- ✦ NEGATIVE EXAMPLES [no-vul] ✦ -------------
    neg_seen = set()
    pos_keys = {near_dup_key(r["text"]) for r in records}
    for i, line in enumerate(Path(non_vul_file).read_text(encoding="utf-8").splitlines(), start=1):
        text = normalize_text(line)
        if not text:
            continue
        if text in neg_seen:
            rep["duplicates_removed"] += 1
            continue
        neg_seen.add(text)
        if near_dup_key(text) in pos_keys:
            quarantine.append({"source": Path(non_vul_file).name, "record": i, "text": text,
                               "label": "", "reason": "тот же текст среди позитивов"})
            continue
        records.append({"text": text, "is_vul": 0, "vul_class": NO_CLASS, "source": "original"})

    # ------------- ✦ REPORT ✦ -------------
    for r in records:
        r["group"] = near_dup_key(r["text"])
    rep["quarantine"] = len(quarantine)
    rep["final_records"] = len(records)
    rep["class_counts"] = dict(collections.Counter(e2e_label(r["is_vul"], r["vul_class"]) for r in records))
    return records, {"counts": dict(rep), "header": header, "quarantine": quarantine}


def e2e_label(is_vul, vul_class):
    return E2E_NAMES[vul_class + 1] if is_vul else E2E_NAMES[0]


# ------------- ✦✦ SPLIT DATA ✦✦ -------------
def stratum(r):
    return f"vul{r['vul_class']}" if r["is_vul"] else f"neg_{r.get('source', 'original')}"


def split_records(records, seed=SEED, test_share=0.2, val_share=0.1):
    # ------------- ✦ GROUP SIZES BY STRATUM ✦ -------------
    groups = collections.defaultdict(lambda: collections.Counter())
    for r in records:
        groups[stratum(r)][r["group"]] += 1

    # ------------- ✦ ASSIGN GROUPS TO PARTS ✦ -------------
    where = {}
    for st, cnt in groups.items():
        total, acc = sum(cnt.values()), 0
        for g in sorted(cnt, key=lambda g: hashlib.sha256(f"{seed}|{st}|{g}".encode()).hexdigest()):
            share = acc / total
            where[(st, g)] = "test" if share < test_share else ("val" if share < test_share + val_share else "train")
            acc += cnt[g]

    # ------------- ✦ BUILD PARTS ✦ -------------
    parts = {"train": [], "val": [], "test": []}
    for r in records:
        parts[where[(stratum(r), r["group"])]].append(r)
    return parts["train"], parts["val"], parts["test"]


def split_fingerprint(train, val, test):
    h = hashlib.sha256()
    for name, part in (("train", train), ("val", val), ("test", test)):
        for t in sorted(r["text"] for r in part):
            h.update(f"{name}\t{t}\n".encode())
    return h.hexdigest()


# ------------- ✦✦✦ METRICS ✦✦✦ -------------
def compute_metrics(is_vul_true, is_vul_pred, class_true, class_pred):
    is_vul_true, is_vul_pred = np.asarray(is_vul_true), np.asarray(is_vul_pred)
    class_true, class_pred = np.asarray(class_true), np.asarray(class_pred)
    pos = is_vul_true == 1

    # ------------- ✦ E2E LABELS ✦ -------------
    e2e_true = np.where(is_vul_true == 1, class_true + 1, 0)
    e2e_pred = np.where(is_vul_pred == 1, class_pred + 1, 0)

    # ------------- ✦ METRICS BLOCK ✦ -------------
    def block(y, p, names):
        labels = list(range(len(names)))
        return {
            "n": int(len(y)),
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=labels, zero_division=0)),
            "per_class": classification_report(y, p, labels=labels, target_names=names,
                                               output_dict=True, zero_division=0),
            "confusion_matrix(rows=true)": confusion_matrix(y, p, labels=labels).tolist(),
        }

    m = {
        "binary": block(is_vul_true, is_vul_pred, ["Не уязвимость", "Уязвимость"]),
        "type": block(class_true[pos], class_pred[pos], CLASS_NAMES),
        "e2e": block(e2e_true, e2e_pred, E2E_NAMES),
    }
    m["binary"]["balanced_accuracy"] = float(balanced_accuracy_score(is_vul_true, is_vul_pred))
    return m


def evaluation_report(test_records, is_vul_pred, class_pred):
    v = np.array([r["is_vul"] for r in test_records])
    c = np.array([r["vul_class"] for r in test_records])
    vp, cp = np.asarray(is_vul_pred), np.asarray(class_pred)
    m = compute_metrics(v, vp, c, cp)

    # ------------- ✦ CONFIDENCE INTERVALS ✦ -------------
    pos = v == 1
    e2e_t = np.where(pos, c + 1, 0)
    e2e_p = np.where(vp == 1, cp + 1, 0)
    return {"test": {"binary_f1": m["binary"]["macro_f1"], "multi_f1": m["type"]["macro_f1"],
                     "e2e_f1": m["e2e"]["macro_f1"], "metrics": m,
                     "ci95": {"binary_f1": bootstrap_ci(v, vp, [0, 1]),
                              "multi_f1": bootstrap_ci(c[pos], cp[pos], [0, 1, 2]),
                              "e2e_f1": bootstrap_ci(e2e_t, e2e_p, [0, 1, 2, 3])}}}


def bootstrap_ci(y, p, labels, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[i], p[i], average="macro", labels=labels, zero_division=0))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ------------- ✦✦✦ SAVE ✦✦✦ -------------
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_quarantine(report, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "record", "reason", "label", "text"])
        w.writeheader()
        for q in report["quarantine"]:
            w.writerow({k: q.get(k, "") for k in w.fieldnames})

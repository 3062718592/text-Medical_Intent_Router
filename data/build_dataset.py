# -*- coding: utf-8 -*-
"""
把「医疗分类数据集.csv」转成项目统一的 data/train.json / data/val.json。

为什么不用 pandas.read_csv / csv 模块：
  1. 编码是 GBK（不是 UTF-8），Tab 分隔（不是逗号）；
  2. 每条记录被 "..." 包住，尾部还挂 14 个空列
     （表头长这样："text\t/label",,,,,,,,,,,,,,，是 Excel 转存的残留）；
  3. 有 53 行正文里嵌了 `"`，把引号配对破坏掉，连行首引号都没了。
     csv 模块会把这类行读成畸形字段，所以这里用窄解析：
       剥尾部填充 → 去外层引号 → 按「最后一个 Tab」切分。

去重策略（关键）：
  - 完全相同的文本只保留一条；
  - 但同一文本被标了**不同标签**时（29 组里有 10 组），说明标注本身不可靠，
    整组丢弃 —— 保留任一条都会把噪声带进训练集和评测集。
    被丢弃的样本记进 meta.json，方便人工复核。

用法：
  python data/build_dataset.py                # 默认从 ../../医疗分类数据集.csv 读
  python data/build_dataset.py --csv <path>   # 指定源文件
"""

import argparse
import hashlib
import io
import json
import random
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent          # .../week6
DATA_DIR = CODE_ROOT / "data"
DEFAULT_CSV = CODE_ROOT.parent / "医疗分类数据集.csv"
DEFAULT_TEST = CODE_ROOT.parent / "test.csv"

# 标签顺序固定为「语义顺序」，不按词频排 —— 否则数据一变 label id 就会漂移，
# 之前训练好的权重、写死的映射表全部错位。
LABELS = (
    "定义",            # 0
    "病因",            # 1
    "临床表现(病症表现)",  # 2
    "适用症",          # 3
    "作用",            # 4
    "用法",            # 5
    "治疗方法",        # 6
    "方法",            # 7
    "多问",            # 8
    "无法确定",        # 9
)
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

SEED = 42
VAL_RATIO = 0.2

_WS = re.compile(r"[ \t\u00a0\u3000]+")
_COMMA_QUOTE = re.compile(r'[,，]+"')
_REPEAT_COMMA = re.compile(r"([,，])\1+")


def read_raw(path: Path):
    """读原始字节并解码。GBK 解不了就退 gb18030（GBK 的超集，能兜住生僻字）。"""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    for enc in ("gbk", "gb18030", "utf-8-sig"):
        try:
            return raw.decode(enc), enc, digest, raw
        except UnicodeDecodeError:
            continue
    raise SystemExit("无法解码 %s：GBK/gb18030/UTF-8 都失败" % path)


def repair_quotes(text: str) -> str:
    """把正文里残留的 `,"` 修成「，」，其余孤立引号直接删掉。

    这些引号来自原始导出：正文中含引号时没有转义成 `""`，于是 `text\\tlabel"`
    的引号配对被打乱。去掉引号、把 `,` 变成全角逗号，语义不变。
    注意有不只一个逗号的形态（`,,"`），所以按「任意长度逗号串 + 引号」一起替换。
    """
    if '"' not in text:
        return text
    text = _COMMA_QUOTE.sub("，", text)      # ," / ,," / ，" → ，
    text = text.replace('"', "")             # 无逗号前缀的孤立引号直接删
    return text.replace("，，", "，")


def parse_rows(text: str):
    """窄解析：返回 [(src_line, sentence, label_raw)]，src_line 从 1 计（含表头）。"""
    rows, seen_header = [], False
    for line_no, ln in enumerate(text.splitlines(), start=1):
        s = ln.strip()
        if not s:
            continue
        if not seen_header:
            seen_header = True          # 第一条是表头 "text\t/label"
            continue
        s = s.rstrip(",").rstrip()      # 剥掉尾部 14 个空列
        if s.endswith('"'):
            s = s[:-1]
        if s.startswith('"'):
            s = s[1:]
        if "\t" not in s:
            continue
        sentence, label = s.rsplit("\t", 1)
        rows.append((line_no, sentence, label.strip()))
    return rows


def normalize(sentence: str) -> str:
    """只做无损规整：修引号、合并连续空白、收掉重复逗号、去首尾空白。

    不改标点宽度（半角逗号保留原样）、不改字符 —— 这些留给「近似重复检测」
    那一层去做，这里只清掉零信息量的噪声（如源数据里的 `，，，` 笔误）。
    """
    s = _WS.sub(" ", repair_quotes(sentence)).strip()
    return _REPEAT_COMMA.sub(r"\1", s)


def read_test(path: Path):
    """读无标注测试集。与训练集格式完全不同：UTF-8、单列 `text`、没标签、没有空列填充。

    注意有 1 行是标准 CSV 引号转义（`"为什么会有""十男九痔""的说法"`），
    需要去掉外层引号并把 `""` 还原成 `"`；其余行就是裸文本。
    """
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    if not lines:
        raise SystemExit("测试集为空：%s" % path)
    out = []
    for ln in lines[1:]:                       # 第一行是表头 `text`
        s = ln.strip()
        if not s:
            continue
        if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
            s = s[1:-1].replace('""', '"')     # 标准 CSV 转义还原
        out.append(_WS.sub(" ", s).strip())
    return out, digest


def stratified_split(rows, ratio, seed):
    """按标签分层切分，保证每一类在 train/val 里的比例一致。"""
    rng = random.Random(seed)
    by_label = OrderedDict()
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    train, val = [], []
    for label in LABELS:
        group = by_label.get(LABEL2ID[label], [])
        group = list(group)
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * ratio))) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--test-csv", default=str(DEFAULT_TEST))
    ap.add_argument("--with-test", action="store_true",
                    help="同时导出无标注测试集 test.json（默认关闭，评测只用带 label 的 val）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    args = ap.parse_args(argv)

    src = Path(args.csv)
    if not src.exists():
        raise SystemExit("源文件不存在：%s" % src)

    text, enc, digest, raw = read_raw(src)
    parsed = parse_rows(text)
    print("编码 %s | %.1f KB | sha256 %s… | 解析出 %d 行"
          % (enc, len(raw) / 1024, digest[:12], len(parsed)))

    # ---------- 清洗 ----------
    n_repaired = 0
    cleaned = []
    bad_label = Counter()
    for line_no, sentence, label_raw in parsed:
        if '"' in sentence:
            n_repaired += 1
        sentence = normalize(sentence)
        if not sentence:
            continue
        if label_raw not in LABEL2ID:
            bad_label[label_raw] += 1
            continue
        cleaned.append({"src": line_no, "sentence": sentence,
                        "label": LABEL2ID[label_raw], "label_raw": label_raw})

    if bad_label:
        raise SystemExit("出现未登记标签：%s" % dict(bad_label))
    print("清洗后 %d 条（其中 %d 条修了正文里的引号）" % (len(cleaned), n_repaired))

    # ---------- 去重 ----------
    groups = OrderedDict()
    for r in cleaned:
        groups.setdefault(r["sentence"], []).append(r)

    kept, dropped_conflict, dropped_dup = [], [], []
    n_same_dup, n_conflict_groups = 0, 0
    for sentence, group in groups.items():
        labels = {r["label"] for r in group}
        if len(labels) > 1:
            n_conflict_groups += 1
            dropped_conflict.extend(
                {"src": r["src"], "sentence": sentence, "label_raw": r["label_raw"]}
                for r in group
            )
            continue
        kept.append(group[0])
        # 同标签重复：保留首条，其余记档（保证「少了多少条、少在哪」完全可查）
        for r in group[1:]:
            dropped_dup.append({"src": r["src"], "sentence": sentence,
                                "label_raw": r["label_raw"],
                                "kept_src": group[0]["src"]})
        n_same_dup += len(group) - 1

    n_dup_groups = sum(1 for g in groups.values() if len(g) > 1)
    print("去重：%d 组重复文本 | 同标签重复丢 %d 条 | 标签冲突 %d 组丢 %d 条"
          % (n_dup_groups, n_same_dup, n_conflict_groups, len(dropped_conflict)))
    print("去重后 %d 条" % len(kept))

    if dropped_conflict:
        print("  标签冲突示例：")
        shown = set()
        for r in dropped_conflict:
            if r["sentence"] in shown:
                continue
            shown.add(r["sentence"])
            labs = [x["label_raw"] for x in dropped_conflict
                    if x["sentence"] == r["sentence"]]
            print("    [%s] %s…" % ("/".join(labs), r["sentence"][:38]))
            if len(shown) >= 3:
                break

    # ---------- 切分 ----------
    train, val = stratified_split(kept, args.val_ratio, args.seed)

    def dump(rows, name):
        payload = [{"idx": i, "sentence": r["sentence"], "label": r["label"]}
                   for i, r in enumerate(rows)]
        out = DATA_DIR / name
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return out, payload

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "medical_faq.csv").write_bytes(raw)      # 留一份原始副本，脱离 Excel 也能复现

    p_train, train_json = dump(train, "train.json")
    p_val, val_json = dump(val, "val.json")

    # ---------- 测试集（无标注） ----------
    test_sentences, test_digest = [], ""
    test_json = []
    if args.with_test and Path(args.test_csv).exists():
        test_sentences, test_digest = read_test(Path(args.test_csv))
        test_json = [{"idx": i, "sentence": s} for i, s in enumerate(test_sentences)]
        with open(DATA_DIR / "test.json", "w", encoding="utf-8") as f:
            json.dump(test_json, f, ensure_ascii=False)
        (raw_dir / "test.csv").write_bytes(Path(args.test_csv).read_bytes())
    else:
        if args.with_test:
            print("\n[警告] 找不到测试集 %s，跳过 test.json" % args.test_csv)

    # ---------- 校验 ----------
    tr_text = {d["sentence"] for d in train_json}
    overlap = sum(1 for d in val_json if d["sentence"] in tr_text)
    assert overlap == 0, "train/val 仍有 %d 条重叠" % overlap
    assert len(train_json) + len(val_json) == len(kept)

    def dist(rows):
        c = Counter(d["label"] for d in rows)
        return {LABELS[k]: c.get(k, 0) for k in range(len(LABELS))}

    d_tr, d_va = dist(train_json), dist(val_json)
    print("\n%-18s %8s %8s %8s" % ("label", "train", "val", "val%"))
    for name in LABELS:
        tot = d_tr[name] + d_va[name]
        print("%-18s %8d %8d %7.1f%%"
              % (name, d_tr[name], d_va[name], d_va[name] / tot * 100 if tot else 0))
    print("%-18s %8d %8d" % ("TOTAL", len(train_json), len(val_json)))

    lens = sorted(len(d["sentence"]) for d in train_json + val_json)
    print("\n文本长度 min=%d p50=%d p95=%d max=%d avg=%.1f"
          % (lens[0], lens[len(lens) // 2], lens[int(len(lens) * .95)],
             lens[-1], sum(lens) / len(lens)))
    print("train/val 重叠 %d 条（必须为 0）" % overlap)

    # ---------- 测试集泄漏体检 ----------
    leak_idx, n_test_dup = [], 0
    if test_json:
        va_text = {d["sentence"] for d in val_json}
        n_test_dup = len(test_json) - len({d["sentence"] for d in test_json})
        leak_idx = [d["idx"] for d in test_json
                    if d["sentence"] in tr_text or d["sentence"] in va_text]
        tl = sorted(len(d["sentence"]) for d in test_json)
        print("\n测试集 %d 条（无标签）：内部重复 %d | 与 train+val 文本重叠 %d 条 (%.1f%%)"
              % (len(test_json), n_test_dup, len(leak_idx),
                 len(leak_idx) / len(test_json) * 100))
        print("测试集长度 min=%d p50=%d p95=%d max=%d"
              % (tl[0], tl[len(tl) // 2], tl[int(len(tl) * .95)], tl[-1]))
        if leak_idx:
            print("  注意：这 %d 条在训练/验证集里已出现过，若日后拿到测试集标签，"
                  "它们会虚高成绩" % len(leak_idx))

    meta = {
        "source": str(src.name),
        "source_sha256": digest,
        "source_encoding": enc,
        "labels": list(LABELS),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "n_raw_rows": len(parsed),
        "n_after_clean": len(cleaned),
        "n_quote_repaired": n_repaired,
        "n_dup_groups": n_dup_groups,
        "n_dup_same_label_dropped": n_same_dup,
        "n_conflict_groups": n_conflict_groups,
        "n_conflict_rows_dropped": len(dropped_conflict),
        "n_final": len(kept),
        "n_train": len(train_json),
        "n_val": len(val_json),
        "label_dist_train": d_tr,
        "label_dist_val": d_va,
        "dropped_total": len(dropped_dup) + len(dropped_conflict),
        "dropped_same_label_dups": dropped_dup,
        "dropped_conflicts": dropped_conflict,
        "test_source": Path(args.test_csv).name if test_json else None,
        "test_source_sha256": test_digest or None,
        "n_test": len(test_json),
        "n_test_dup": n_test_dup,
        "test_leak_idx": leak_idx,
    }
    with open(DATA_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n已写出：")
    outs = [p_train, p_val, DATA_DIR / "meta.json", raw_dir / "medical_faq.csv"]
    if test_json:
        outs.insert(2, DATA_DIR / "test.json")
    for p in outs:
        print("  %-28s %8.1f KB" % (p.name, p.stat().st_size / 1024))


if __name__ == "__main__":
    sys.exit(main())

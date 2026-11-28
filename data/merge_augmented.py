# -*- coding: utf-8 -*-
"""
把增强样本合并进训练集，并写 provenance（来源账本）

关键设计：**绝不覆盖 data/train.json**
  原始 1667 条是基线，必须永远可复现。增强版写到 train_aug.json，
  训练脚本用 --data 参数选择，这样 2083 基线和 5000 增强版能并排对比。

为什么 provenance 这么重要：
  面试官/审稿人问「你这 5000 条哪来的」，答案必须在 meta 里能查到：
      人工标注 1667 + 回译增强 N = 总数
  写不清来源的数据集 = 造假。这份脚本把每一类的条数、引擎、生成时间都记下来。

用法：
  python data/merge_augmented.py                    # 合并并写 train_aug.json
  python data/merge_augmented.py --report           # 只报告，不写文件
  python data/merge_augmented.py --max-per-src 1    # 每个原句最多取 1 条变体
"""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_ROOT / "data"
TRAIN = DATA_DIR / "train.json"
AUG = DATA_DIR / "aug_backtranslated.json"
OUT = DATA_DIR / "train_aug.json"
META = DATA_DIR / "meta.json"


def load_json(p, default=None):
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def label_dist(rows):
    from common.labels import LABELS
    c = Counter(d["label"] for d in rows)
    return {LABELS[k]: c.get(k, 0) for k in range(len(LABELS))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="只报告，不写文件")
    ap.add_argument("--max-per-src", type=int, default=0,
                    help="每个原句最多保留几条变体，0=不限")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(CODE_ROOT))

    train = load_json(TRAIN)
    aug = load_json(AUG, [])
    if train is None:
        raise SystemExit("找不到 %s" % TRAIN)
    if not aug:
        raise SystemExit("找不到增强样本 %s，先跑 data/augment_backtranslate.py" % AUG)

    # ---- 过滤 1：与原句/其他变体重复 ----
    seen = {d["sentence"] for d in train}
    kept, n_dup, per_src = [], 0, Counter()
    for a in aug:
        s = a["sentence"].strip()
        if not s or s in seen:
            n_dup += 1
            continue
        if args.max_per_src and per_src[a.get("src_idx")] >= args.max_per_src:
            n_dup += 1
            continue
        seen.add(s)
        per_src[a.get("src_idx")] += 1
        kept.append({"idx": 0, "sentence": s, "label": a["label"],
                     "origin": "aug_backtranslation",
                     "src_idx": a.get("src_idx"), "engine": a.get("engine", "unknown")})

    # ---- 过滤 2：标签合法性 ----
    from common.labels import LABELS, NUM_LABELS
    assert NUM_LABELS == len(LABELS)
    bad = [k for k in kept if not (0 <= k["label"] < NUM_LABELS)]
    if bad:
        print("[warn] 丢弃 %d 条标签越界的增强样本" % len(bad))
        kept = [k for k in kept if 0 <= k["label"] < NUM_LABELS]

    # ---- 合并 + 洗牌（固定 seed）----
    merged = [{"idx": 0, "sentence": d["sentence"], "label": d["label"],
               "origin": "human"} for d in train] + kept
    random.Random(args.seed).shuffle(merged)
    for i, d in enumerate(merged):
        d["idx"] = i

    # ---- 报告 ----
    d_old, d_new = label_dist(train), label_dist(merged)
    print("%-22s %8s %8s %8s" % ("label", "原训练集", "合并后", "增强"))
    for name in LABELS:
        print("%-22s %8d %8d %8d"
              % (name, d_old[name], d_new[name], d_new[name] - d_old[name]))
    print("%-22s %8d %8d %8d"
          % ("TOTAL", len(train), len(merged), len(merged) - len(train)))
    print("\n丢弃重复/超限 %d 条；保留增强 %d 条（人工 %d + 增强 %d）"
          % (n_dup, len(kept), len(train), len(kept)))

    # ---- 相似度体检：变体与原句的字符重合度，防止"只是改了标点" ----
    src_map = {d["idx"]: d["sentence"] for d in train}
    ratios = []
    for k in kept[:500]:
        src = src_map.get(k.get("src_idx"))
        if not src:
            continue
        common = sum(1 for a, b in zip(src, k["sentence"]) if a == b)
        ratios.append(common / max(len(src), 1))
    if ratios:
        ratios.sort()
        print("字符位置重合度 p50=%.2f p95=%.2f（太接近 1.0 说明增强无效）"
              % (ratios[len(ratios) // 2], ratios[int(len(ratios) * 0.95)]))

    if args.report:
        return 0

    OUT.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    print("\n→ %s（%d 条，%.1f KB）" % (OUT, len(merged), OUT.stat().st_size / 1024))

    # ---- 写 provenance ----
    meta = load_json(META, {})
    engines = Counter(k.get("engine", "unknown") for k in kept)
    meta["augmentation"] = {
        "method": "backtranslation",
        "engines": dict(engines),
        "n_human": len(train),
        "n_augmented": len(kept),
        "n_total": len(merged),
        "n_dropped_duplicate": n_dup,
        "max_per_src": args.max_per_src or None,
        "seed": args.seed,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "train_file": "data/train_aug.json",
        "note": ("增强样本由原训练集经「中→英→中」回译生成，标签沿用原句标签，"
                 "未经人工复核；data/train.json 保持原始 1667 条不变，基线可复现。"),
        "label_dist": d_new,
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("→ provenance 已写入 %s" % META)
    print("\n下一步：python llm_lora/train_lora.py --data data/train_aug.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

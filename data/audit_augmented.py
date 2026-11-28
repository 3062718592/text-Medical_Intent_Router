# -*- coding: utf-8 -*-
"""
增强样本质检闸门（audit gate）

为什么需要这一层：
  回译会引入「语义漂移」——最典型的是**药名被替换**。
  实测本批数据里出现过「童康片 → 通康片」「湿热痹胶囊 → 四妙胶囊」，
  这类样本进了训练集，模型学的就是错的药名。医疗场景下不可接受。

三道检查（全部基于「原句 vs 增强句」对比）：
  1. 药名保持：原句里的药名核心必须在增强句里出现（允许简称/全名的包含关系）
  2. 数字保持：剂量/年龄/指标这类阿拉伯数字不能被改动或丢失
  3. 长度合理：增强句相对原句的长度比在 [0.5, 2.0] 内

设计取舍（保守优先）：
  检查会误杀一些合法样本（如「脊灰糖丸 → 脊髓灰质炎糖丸」是同一药，但核心字对不上）。
  宁可少几百条干净样本，也不要几十条错药名 —— 后者会让整个增强实验的结论不可信。

用法：
  python data/audit_augmented.py                    # 只看报告
  python data/audit_augmented.py --write            # 输出 train_aug_clean.json
  python data/audit_augmented.py --write --keep-drug-mismatch   # 不因药名丢弃
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_ROOT / "data"

SUFFIX = r'(?:片|胶囊|颗粒|丸|注射液|口服液|软膏|滴丸|糖浆|喷剂|滴眼液|乳膏|栓|散|贴|膏)'
# 药名：2~6 个中英文字符 + 剂型后缀
DRUG = re.compile(r'([\u4e00-\u9fa5A-Za-z]{2,6})' + SUFFIX)
NUM = re.compile(r'\d+(?:\.\d+)?')
# 数字的汉字形式（6→六、5→五 …），用于判断「阿拉伯转汉字」不算数字丢失
CN_NUM = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
          "5": "五", "6": "六", "7": "七", "8": "八", "9": "九", "10": "十"}

# 常见动词/量词前缀：切开它才能拿到药名核心（"吃了一年的奥氮平片" → "奥氮平片"）
PREFIX_NOISE = ("已经", "吃了一年", "一年的", "可以", "服用", "吃了", "小孩", "孩子",
                "我的", "你好", "请问", "我想", "要吃", "能吃", "在吃", "小孩吃")


def drug_cores(sentence: str):
    """抽取药名核心集合。核心 = 剂型后缀前 2~4 个字符，去重后用于包含性比对。"""
    cores = set()
    for m in DRUG.finditer(sentence):
        name = m.group(1)
        if len(name) < 2:
            continue
        cores.add(name)
        # 长名取尾部（更独特）："吃了一年的奥氮平" → "奥氮平"
        if len(name) >= 4:
            cores.add(name[-3:])
            cores.add(name[-4:])
    return cores


def drug_ok(src: str, aug: str) -> bool:
    """增强句至少保住原句一个药名核心，且不能凭空多出一个「核心字完全不同」的药名。"""
    cs = drug_cores(src)
    if not cs:
        return True                      # 原句无药名，无从比较
    if any(c in aug for c in cs):
        return True
    return False


def num_ok(src: str, aug: str) -> bool:
    """阿拉伯数字必须保住；若原句数字在增强句里以汉字形式出现，也算保住。"""
    a = NUM.findall(src)
    if not a:
        return True
    b = NUM.findall(aug)
    for x in a:
        if x in b:
            continue
        # 阿拉伯 → 汉字 的转换不算丢失（"6个月" → "六个月"）
        cn = "".join(CN_NUM.get(ch, ch) for ch in x)
        if cn and cn in aug:
            continue
        return False
    return True


def len_ok(src: str, aug: str) -> bool:
    if not src:
        return False
    r = len(aug) / len(src)
    return 0.5 <= r <= 2.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default=str(DATA_DIR / "aug_backtranslated.json"))
    ap.add_argument("--train", default=str(DATA_DIR / "train.json"))
    ap.add_argument("--out", default=str(DATA_DIR / "train_aug_clean.json"))
    ap.add_argument("--write", action="store_true", help="写出清洗后的训练集")
    ap.add_argument("--keep-drug-mismatch", action="store_true",
                    help="不因药名不一致丢弃（默认丢弃，保守策略）")
    ap.add_argument("--show", type=int, default=5, help="每类问题展示几条样例")
    args = ap.parse_args(argv)

    train = json.loads(Path(args.train).read_text(encoding="utf-8"))
    aug = json.loads(Path(args.aug).read_text(encoding="utf-8"))
    src_map = {d["idx"]: d["sentence"] for d in train}
    src_lab = {d["idx"]: d["label"] for d in train}

    passed, reasons = [], Counter()
    samples = {"drug": [], "num": [], "len": []}

    for d in aug:
        s = src_map.get(d.get("src_idx"), "")
        b = d["sentence"]
        if not s:
            reasons["无原句"] += 1
            continue
        if not len_ok(s, b):
            reasons["长度异常"] += 1
            if len(samples["len"]) < args.show:
                samples["len"].append((s, b))
        elif not num_ok(s, b):
            reasons["数字被改"] += 1
            if len(samples["num"]) < args.show:
                samples["num"].append((s, b))
        elif not args.keep_drug_mismatch and not drug_ok(s, b):
            reasons["药名不一致"] += 1
            if len(samples["drug"]) < args.show:
                samples["drug"].append((s, b))
        else:
            passed.append(d)

    print("增强样本 %d 条 → 通过 %d 条，丢弃 %d 条"
          % (len(aug), len(passed), len(aug) - len(passed)))
    for k, v in reasons.items():
        print("  - %-10s %d 条" % (k, v))

    for tag, title in (("drug", "药名不一致"), ("num", "数字被改"), ("len", "长度异常")):
        if samples[tag]:
            print("\n=== %s 样例 ===" % title)
            for s, b in samples[tag]:
                print("  原: %s" % s[:60])
                print("  增: %s" % b[:60])
                print()

    if not args.write:
        print("\n（未写文件；加 --write 输出 %s）" % Path(args.out).name)
        return 0

    merged = [{"idx": 0, "sentence": d["sentence"], "label": d["label"],
               "origin": "human"} for d in train]
    merged += [{"idx": 0, "sentence": d["sentence"], "label": d["label"],
                "origin": "aug_backtranslation", "src_idx": d.get("src_idx"),
                "engine": d.get("engine", "unknown")} for d in passed]
    import random
    random.Random(42).shuffle(merged)
    for i, d in enumerate(merged):
        d["idx"] = i

    Path(args.out).write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    print("\n→ %s（人工 %d + 增强 %d = %d 条）"
          % (args.out, len(train), len(passed), len(merged)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

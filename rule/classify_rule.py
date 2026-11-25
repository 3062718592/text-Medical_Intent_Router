"""
方法一：规则 / 关键词（L0 基线）

教学重点：
  1. 这是**最便宜的**方法：不训练、不加载模型，几十行代码，跑 416 条只要毫秒级。
     先跑它，是为了给后面所有方法一个「下限」——连它都没有的东西不配谈提升。
  2. 规则法的上限**完全取决于任务形态**，这一点比看数值更重要：
     - TNEWS 主题分类：标题里几乎不出现类别名，命中率 6%，Macro-F1 0.069 ≈ 随机；
     - 本任务意图分类：用户问"怎么吃"就是在问「用法」，问"什么原因"就是在问「病因」，
       问句里的疑问词本身就是类标签 → 规则命中率天然很高。
     同一份代码，在两种任务上从 0.07 到接近 0.4，差的就是「语言线索是否与标签对齐」。
  3. 规则的真正强项是**处理模型处理不了的输入**：`无`、`？`、`没有` 这种长度 1-2 的
     垃圾输入，训练集里一条都没有，模型只会乱猜且可能给高置信度 —— 用 is_garbage()
     在模型之前拦掉，比让模型背锅便宜得多。

已知弱项（也是本基线的价值）：
  「定义 / 临床表现 / 治疗方法 / 方法」四类的语言线索互相重叠
  （"我这个是不是尖锐湿疣" 既是定义也是临床表现），规则无论怎么排优先级都只能拿其中一类。
  这四类数量占 45%，正是后面 TF-IDF / BERT 要抢占的空间。

用法：
  python rule/classify_rule.py              # 全部 val（416 条）
  python rule/classify_rule.py --n 200      # 抽样
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val, load_splits
from common.labels import LABELS, LABEL2ID, is_garbage
from common.metrics import classification_report, pred_distribution, to_markdown_table

# 规则表：**顺序即优先级**，第一个命中的模式决定标签。
#
# 排序原则：越具体的意图越靠前。
#   - 「多问」是最强的结构信号（问号 ≥ 2），先判；
#   - 「怎么吃」必须排在「怎么治」前面，否则会被泛化的"怎么"吃掉；
#   - 「什么原因引起的」含"什么"，必须先于「定义」的"是什么"；
#   - 「治疗方法」的"怎么治"先于「方法」的"怎么"。
RULES = [
    ("多问",   r"(\?|？).*(\?|？)"),                                  # 一句话里 ≥2 个问号
    ("用法",   r"怎么吃|怎么服|如何服|怎么用|用量|一次吃|一天吃|几次|几粒|几片|饭前|饭后|服用方法|怎么喝"),
    ("适用症", r"能不能用|可以吃吗|能吃吗|可以用吗|能服用|适用|适应症|孕妇|哺乳|经期|儿童能|过敏"),
    ("作用",   r"功效|作用|效果|管用|有用吗|有什么好处|不良反应|副作用"),
    ("病因",   r"什么原因|什么引起|为什么|怎么会|病因|原因|导致的"),
    ("定义",   r"什么是|是什么|什么叫|定义|是指|怎么回事|啥意思|正常吗"),
    ("治疗方法", r"怎么治|如何治|治疗|疗法|怎么处理|吃什么药|用什么药"),
    ("方法",   r"怎么|如何|怎样|办法"),
]
_COMPILED = [(LABEL2ID[name], re.compile(pat), name) for name, pat in RULES]


def rule_match(text: str):
    """返回命中的标签 id；一条都没命中返回 None。

    服务层只用它做「快筛线索」，不直接当成最终答案 ——
    本数据上规则准确率只有 0.26，拿它直接放行比模型还危险。
    单独抽成函数，是为了让 service/ 和评测脚本共用同一张规则表（口径必须同源）。
    """
    for label_id, rx, _ in _COMPILED:
        if rx.search(text):
            return label_id
    return None


def classify(text: str, fallback: int) -> int:
    """按规则表顺序匹配，全不中则退化成训练集最多的那一类（majority baseline）。"""
    if is_garbage(text):
        return fallback            # 垃圾输入：L0 在服务层是 drop，这里退回兜底类以便算分
    hit = rule_match(text)
    return fallback if hit is None else hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部")
    ap.add_argument("--out", default="outputs/rule_pred.json")
    args = ap.parse_args()

    samples, meta = load_clean_val(args.n or None)
    train, _ = load_splits()

    fallback = Counter(d["label"] for d in train).most_common(1)[0][0]
    y_true = [d["label"] for d in samples]
    y_pred = [classify(d["sentence"], fallback) for d in samples]

    hit = sum(1 for d in samples if any(rx.search(d["sentence"]) for _, rx, _ in _COMPILED))
    garbage = sum(1 for d in samples if is_garbage(d["sentence"]))

    report = classification_report(y_true, y_pred)
    report["method"] = "rule_keyword"
    report["hit_rate"] = round(hit / len(samples), 4)
    report["garbage_rate"] = round(garbage / len(samples), 4)
    report["pred_distribution"] = pred_distribution(y_pred)

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"样本 {meta['n_eval']} 条（train {meta['n_train']} / val {meta['n_val']}）")
    print(f"规则命中率 {hit / len(samples):.2%} | 垃圾输入 {garbage} 条 | 兜底类 {LABELS[fallback]}")
    print(to_markdown_table({"规则/关键词": report}))

    print("\n各类召回（看规则吃到了哪些类）：")
    for i, name in enumerate(LABELS):
        r = report["per_class_recall"].get(i)
        print("  %-20s %s" % (name, "—（该样本不在评测集）" if r is None else f"{r:.2f}"))

    print(f"\n→ 结果写入 {out}")


if __name__ == "__main__":
    main()

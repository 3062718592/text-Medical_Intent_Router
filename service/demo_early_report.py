"""
早报面板 Demo：模拟「每天几百条提问进来，先分流再给你看」

教学重点：
  1. 分流做完不是结束——产物要变成**动作**：哪些能自动回答、哪些转人工、哪些直接丢。
  2. 同一个服务，喂一批假数据就能演示产品形态（不用等真实接入）。
  3. 注意「review」这一档：低置信的结果不会被丢掉，而是转 LLM 兜底/人工复核，
     宁可多一步，也不要把不确定的结论推给患者。
  4. 医疗场景和投研的差别在这一屏就能看出来：`多问`（一句话问了三个问题）
     和 `无法确定`（"我不知道自己在问什么"）都必须人工介入，不能自动放行。

用法：
  python service/demo_early_report.py                 # 默认阈值 0.85（auto 通常为空，见末尾提示）
  python service/demo_early_report.py --conf 0.6      # 放宽阈值，看三队列都被填满的样子
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.classify_cli import ShuntService

# 模拟一早晨进来的提问：混合了典型意图、模糊意图、多问题、以及垃圾输入
MESSAGES = [
    "布洛芬一天最多吃几次？",
    "尖锐湿疣是什么病",
    "吃了头孢能喝酒吗",
    "高血压是什么原因引起的",
    "宝宝发烧 38.5 度，用什么退烧药好？需要去医院吗？",
    "痔疮手术后会复发吗",
    "达喜和奥美拉唑能一起吃吗",
    "月经量少会影响怀孕吗",
    "甲状腺结节需要手术吗",
    "孕妇可以用皮炎平吗",
    "龙牡壮骨颗粒的功效",
    "什么是心电图运动负荷试验",
    "晚上睡觉磨牙是怎么回事",
    "颈椎病引起的头晕怎么治疗",
    "尖锐湿疣能治好吗？潜伏期多久？会传染给家人吗？",
    "我这种情况算严重吗",
    "无",
    "？",
    "已经吃了两盒没效果",
    "请问一下",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.85,
                    help="置信度阈值；0.85 是默认口径，10 类校准后的概率很少过线，auto 常为空")
    ap.add_argument("--ckpt", default=None, help="BERT 权重路径，默认 outputs/bert_ckpt.pt")
    args = ap.parse_args()

    # 用 CLI 版的 ShuntService：它带 bert → tfidf → rule 三级降级，
    # 所以没有 BERT 权重时也能给出真实分类（2 秒现训 TF-IDF），而不是满屏「无法确定」
    svc = ShuntService(model="auto", ckpt=args.ckpt, conf_threshold=args.conf)
    rows = [svc.classify(m) for m in MESSAGES]

    buckets = {"auto": [], "review": [], "drop": []}
    for r in rows:
        buckets[r["queue"]].append(r)

    def show(title, items, note):
        print(f"\n{'='*66}\n{title}（{len(items)} 条）　{note}\n{'='*66}")
        for r in items:
            print(f"  [{r['topic']}/{r['sub_topic']:<8}] conf={r['conf']:.2f} "
                  f"via {r['model']:<12} | {r['title'][:30]}")
        if not items:
            print("  （空）")

    show("① 自动回答 auto", buckets["auto"], "相关 + 校准后高置信 → 直接出答案/入库")
    show("② 待复核 review", buckets["review"], "置信度不足 / 多问 / 无法确定 → 转人工")
    show("③ 丢弃 drop", buckets["drop"], "L0 规则拦下的垃圾输入 → 不送模型")

    total = len(rows)
    if not buckets["auto"]:
        if svc.backend in ("rule", "fallback"):
            print("\n[info] auto 为空：没有可用模型（已退化到规则后端），分类仅供演示接口形态。"
                  "装好 TF-IDF 依赖或放好 outputs/bert_ckpt.pt 再看。")
        else:
            print(f"\n[info] auto 为空：阈值 {svc.conf_threshold:.2f} 下没有样本过线 —— "
                  f"这是默认口径（阈值尚未用 risk–coverage 曲线标定），不是链路坏了。"
                  f"加 --conf 0.6 可以看到更多样本被放行。")
    print(f"\n合计 {total} 条：auto {len(buckets['auto'])} / "
          f"review {len(buckets['review'])} / drop {len(buckets['drop'])}")
    print(f"后端={svc.backend} 温度 T={svc.temperature:.2f} 阈值={svc.conf_threshold:.2f}")
    print("多问：", ", ".join(r["title"][:14] for r in rows if r["sub_topic"] == "多问") or "无")


if __name__ == "__main__":
    main()

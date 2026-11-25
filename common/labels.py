"""
标签体系与分流口径（全项目共用）

和 TNEWS 版本的根本差别：
  TNEWS 是**主题分类**（这条新闻讲哪个行业），标签之间有"业务相关性"可言
  （金融/产业相关，娱乐/体育是噪音），所以 relevant 由主题派生。
  这份医疗数据是**意图分类**（用户在问什么类型的问题），10 个意图全都合法，
  没有"噪音类"；真正决定能不能自动处理的是两件事：
      ① 这个意图是否唯一、能被一条答案覆盖（多问 = 不能）；
      ② 这个问题本身是否定得下来型（无法确定 = 不能）。

  所以 relevant 不再是"话题相关"，而是"**可自动路由**"：
      定义/病因/临床表现/适用症/作用/用法/治疗方法/方法  → True
      多问（一条答案覆盖不全，要拆题）                     → False
      无法确定（连标注者都定不了型）                       → False

为什么 drop 不从这里出：
  这份数据里 10 类都是真实提问，没有垃圾样本，所以模型永远不该做"丢弃"决策。
  drop 只能由 L0 规则产生（空串、长度 < 3、纯标点/纯数字）——
  这个判断不需要模型，也不该让模型背锅。见 rule/classify_rule.py。

为什么单独成文件：
  规则法、TF-IDF、BERT、LLM 四路方法的输出标签必须完全一致，
  否则同一个 0.42 的 F1 在不同脚本里含义都不同，无法横向比较。
"""

import re

# 10 个意图（顺序 = data/build_dataset.py 里的 LABELS，即 json 中的 label id）
LABELS = [
    "定义",              # 0  是什么 / 定义
    "病因",              # 1  为什么 / 病因
    "临床表现(病症表现)",   # 2  有什么症状 / 表现
    "适用症",            # 3  我这种情况能不能用 / 适应症
    "作用",              # 4  这药/这疗法有什么用 / 功效
    "用法",              # 5  怎么用 / 用法用量
    "治疗方法",           # 6  怎么治
    "方法",              # 7  怎么做（操作类，比"治疗方法"更泛）
    "多问",              # 8  一句话里问了多个问题
    "无法确定",           # 9  定不了型
]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# ---- 意图分组：给业务看的粗粒度视图（不参与 id，只用于聚合和报表） ----
GROUP_MAP = {
    "定义": "是什么",
    "病因": "为什么",
    "临床表现(病症表现)": "有什么表现",
    "适用症": "能不能用",
    "作用": "有什么用",
    "用法": "怎么用",
    "治疗方法": "怎么处理",
    "方法": "怎么处理",
    "多问": "需拆解",
    "无法确定": "需人工",
}
GROUPS = ["是什么", "为什么", "有什么表现", "能不能用", "有什么用",
          "怎么用", "怎么处理", "需拆解", "需人工"]

# 可自动路由的意图：单一意图 + 内容可定型。
# 用集合而不是 if/else 链 —— 业务口径变了只改这一个集合。
RELEVANT_LABELS = {
    "定义", "病因", "临床表现(病症表现)", "适用症",
    "作用", "用法", "治疗方法", "方法",
}

# 医疗场景的分流风险上限：这些意图就算高置信也不自动出答案，一律人工复核。
# "治疗方法" 占 22%、是最大类，一旦答错直接涉及治疗建议，代价最高。
# 目前留空：先用默认口径跑通链路，等 val 上看到各类准确率再决定要不要收紧。
FORCE_REVIEW_LABELS = set()

DEFAULT_CONF_THRESHOLD = 0.85


# 生成式模型输出的「别名表」：LLM 很少原样吐出 "临床表现(病症表现)"，
# 往往只写「临床表现」或「症状」。三个 LLM 脚本必须共用同一份解析逻辑，
# 否则同一段输出在 zero-shot / LoRA / API 三个脚本里可能被判成不同标签。
ALIASES = {
    "定义": ["定义", "是什么", "什么是", "含义", "概念"],
    "病因": ["病因", "原因", "为什么", "诱因"],
    "临床表现(病症表现)": ["临床表现(病症表现)", "临床表现", "病症表现", "症状", "表现"],
    "适用症": ["适用症", "适应症", "能不能用", "适用"],
    "作用": ["作用", "功效", "效果", "副作用"],
    "用法": ["用法", "用量", "怎么吃", "怎么用"],
    "治疗方法": ["治疗方法", "治疗", "怎么治"],
    "方法": ["方法", "怎么办", "操作"],
    "多问": ["多问", "多个问题", "多问题"],
    "无法确定": ["无法确定", "不确定", "无法判断"],
}
# 按别名长度降序展开：必须先匹配「治疗方法」再匹配「方法」，否则长子串会被短别名抢先
_ALIAS_SORTED = sorted(
    ((alias, LABEL2ID[name]) for name, alts in ALIASES.items() for alias in alts),
    key=lambda x: -len(x[0]),
)


def match_label(text: str) -> int:
    """
    从生成文本里抽标签 id，抽不到返回 -1（记作解析失败）。

    为什么按「最长别名优先」而不是「按 LABELS 顺序取第一个命中」：
      "治疗方法" 里包含 "方法"，"适用症" 里包含 "适用"。
      按顺序扫描会把它们判错；按别名长度降序取第一条命中才是对的。
    """
    for alias, label_id in _ALIAS_SORTED:
        if alias in text:
            return label_id
    return -1


_MEANINGLESS = re.compile(r"^[\s\W\d_]+$")          # 纯标点 / 纯数字 / 纯空白

def is_garbage(text: str) -> bool:
    """
    L0 规则：这类输入直接进 drop 队列，不送模型。

    为什么不让模型判：test.csv 里有 `无`、`？`、`没有` 这种 1~2 字的输入，
    而训练集最短是 3 字、一条都没有 —— 模型没见过的形态，输出是随机的，
    但 softmax 仍可能给 0.9。这种「模型自信地胡说」只能靠规则挡在模型之前。

    判定口径（保守，宁漏不误杀）：
      - 去空白后长度 < 3；
      - 或者整串没有任何汉字/字母（纯标点、纯数字）。
    """
    s = (text or "").strip()
    if len(s) < 3:
        return True
    if _MEANINGLESS.match(s):
        return True
    return False


# ---- 生成式方法的 Prompt 单一来源 ----
# 为什么放在这里而不是各脚本里各写一份：
#   SFT 有个铁律——**训练和推理的模板必须逐字一致**。
#   以前三个脚本各存一份 SYSTEM，改一处忘一处，训练与评测就悄悄错位，
#   指标掉了还以为是模型问题。放进 labels.py 后，改口径只改这一个文件。
# 为什么类别名要带白话定义：
#   只列 10 个词（尤其"方法"/"适用症"/"临床表现"），0.6B 只能猜；
#   给「适用症：这药我能不能用」它才可能对上。
LABEL_GUIDE = [
    ("定义", "询问某个病/药/词是什么、是什么意思"),
    ("病因", "询问为什么会得、原因是什么"),
    ("临床表现(病症表现)", "询问有什么症状、表现、正不正常"),
    ("适用症", "询问某药或疗法我能不能用、适不适合、禁忌"),
    ("作用", "询问有什么功效、效果、副作用"),
    ("用法", "询问怎么吃、怎么用、吃多少、一天几次"),
    ("治疗方法", "询问怎么治、治疗方案、要不要手术"),
    ("方法", "询问怎么做、办事流程、怎么办"),
    ("多问", "一句话里问了两个或更多不同的问题"),
    ("无法确定", "信息太少，判断不出在问什么"),
]

SYSTEM_PROMPT = (
    "你是医疗问答的意图分类助手。把用户的问题归到下面 10 个意图之一，"
    "只输出意图名称本身，不要解释、不要加标点。\n"
    "可选意图（名称：含义）\n"
    + "\n".join(f"- {name}：{desc}" for name, desc in LABEL_GUIDE)
)


def to_group(label_id: int) -> str:
    """10 类 id → 粗粒度意图组"""
    return GROUP_MAP[LABELS[label_id]]


def is_relevant(label_id: int) -> bool:
    """10 类 id → 能否自动路由（意图唯一、可被一条答案覆盖）"""
    return LABELS[label_id] in RELEVANT_LABELS


def queue_name(label_id: int, conf: float,
               conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> str:
    """
    分流结果落到哪个队列 —— 这是「分类」变成「动作」的地方。

    三个队列：
      auto   相关且高置信 → 自动出答案 / 入库
      review 不相关，或相关但置信度低 → 人工（多问走拆题、无法确定走判废）
      drop   只由 L0 规则产生（空/过短/纯标点），模型不参与

    优先级（顺序不能换）：
      1) 不相关（多问/无法确定）→ review，与置信度无关。
         模型哪怕 0.99 自信地判成"无法确定"，也不能自动放行。
      2) 强制人审的意图（FORCE_REVIEW_LABELS）→ review。
      3) 置信度不足 → review。
      4) 其余 → auto。

    conf_threshold 默认 0.85 而不是 TNEWS 那版的 0.8：10 类、类别均衡得多，
    softmax 最大概率的分布完全不同，0.8 在新数据上会放过过多样本。
    真实阈值要用 val 的 risk–coverage 曲线标定（precision ≥ 0.9 的最大覆盖率）。
    """
    if not is_relevant(label_id):
        return "review"
    if LABELS[label_id] in FORCE_REVIEW_LABELS:
        return "review"
    if conf < conf_threshold:
        return "review"
    return "auto"


# ---- 一致性自检：改口径后跑一下，防止把 10 类和 15 类的口径混在一起 ----
def _self_check():
    assert NUM_LABELS == 10, NUM_LABELS
    assert len(LABEL2ID) == NUM_LABELS
    assert set(GROUP_MAP) == set(LABELS), "GROUP_MAP 与 LABELS 必须一一对应"
    assert RELEVANT_LABELS <= set(LABELS), RELEVANT_LABELS - set(LABELS)
    assert FORCE_REVIEW_LABELS <= set(LABELS), FORCE_REVIEW_LABELS - set(LABELS)
    assert set(ALIASES) == set(LABELS), "ALIASES 必须覆盖全部 10 类"
    assert [g[0] for g in LABEL_GUIDE] == LABELS, "LABEL_GUIDE 顺序必须与 LABELS 一致"
    # 最长别名优先：这几个纠缠对必须判对
    assert match_label("治疗方法") == LABEL2ID["治疗方法"]
    assert match_label("方法") == LABEL2ID["方法"]
    assert match_label("临床表现") == LABEL2ID["临床表现(病症表现)"]
    assert match_label("我认为是适用症的") == LABEL2ID["适用症"]
    assert match_label("胡言乱语") == -1
    # 不变量：auto ⇒ relevant；relevant=False ⇒ review
    for i in range(NUM_LABELS):
        for c in (0.0, 0.5, 0.85, 0.99):
            q = queue_name(i, c)
            if q == "auto":
                assert is_relevant(i), (LABELS[i], c)
            if not is_relevant(i):
                assert q == "review", (LABELS[i], c)


if __name__ == "__main__":
    _self_check()
    print("%-20s %-4s %-10s %-8s %s"
          % ("label", "id", "组", "可自动", "queue@0.9 / @0.6"))
    for i, name in enumerate(LABELS):
        print("%-20s %-4d %-10s %-8s %-8s %s"
              % (name, i, to_group(i), is_relevant(i),
                 queue_name(i, 0.9), queue_name(i, 0.6)))
    print("\n自检通过：%d 类，阈值默认 %.2f" % (NUM_LABELS, DEFAULT_CONF_THRESHOLD))

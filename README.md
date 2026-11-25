# 医疗问答意图分流（Medical Intent Router）

把「用户提问」变成「下游动作」：一条问题进来，输出**意图类别 + 校准置信度 + 路由队列**（`auto` / `review` / `drop`），而不是只给一个标签。

同一条流水线上跑了 6 路方法（规则 → TF-IDF → BERT → Qwen3 零样本 / LoRA / 全量微调），全部在**同一批验证样本**上评测，用来回答一个工程问题：

> 在只有 1 667 条训练数据、一张 8 G 显卡、要求离线可跑的前提下，哪条路线的性价比最高？

结论是 **Qwen3-0.6B + LoRA**（Macro-F1 0.49），但真正决定这套系统能不能上线的，不是这 0.49，而是后面那节「为什么置信度必须先校准」。

---

## 结果

评测口径：`data/val.json` 416 条，`seed=42` 分层切分，训练/验证已去重；准确率把解析失败计入分母。

| 方法 | 准确率 | Macro-F1 | 覆盖类别 | 备注 |
|---|---|---|---|---|
| 规则 / 关键词 | 0.2596 | 0.2199 | 8 / 10 | 毫秒级，作下限基线 |
| TF-IDF + SVM | 0.4423 | 0.3772 | — | 20 秒训练，无模型依赖 |
| BERT-base-chinese 微调 | 0.4784 | 0.4736 | — | 最高 Macro-F1/成本比 |
| Qwen3-0.6B 零样本 | 0.1779 | 0.1432 | 7 / 10 | 类别塌缩，不可用 |
| **Qwen3-0.6B + LoRA** | **0.5865** | **0.4898** | 10 / 10 | 最优 |
| Qwen3-0.6B 全量微调 | 0.5481 | 0.4743 | 10 / 10 | 1 667 条上过拟合 |
| API 零样本（deepseek-v4-flash） | — | — | — | 未运行，见「已知问题」 |

> 416 条样本上准确率的 95% 置信区间约 ±4.8 个百分点。**两个方法差 2 个点以内不算有差异** —— 这句话比表里任何数字都重要。

在干净 clone 上跑 `python rule/classify_rule.py` 会得到与上表**完全一致**的 0.2596 / 0.2199（固定 seed、无随机性），可以拿它验证环境是否复现成功。

---

## 6 个关键设计决策

这部分是仓库里最值得看的内容，每条都是"当时踩过坑才这么写"。

### 1. 数据脏到必须写自定义解析器

原始数据是 GBK 编码、Tab 分隔、每条被 `"` 包住、尾部挂 14 个空列，还有 **53 行正文里嵌了引号**把配对破坏掉。`csv` 模块和 `pandas.read_csv` 都会读成畸形字段。

`data/build_dataset.py` 用窄解析：剥尾部填充 → 去外层引号 → 按最后一个 Tab 切分 → 修复 `,"` 形态的残留引号。

去重策略是最关键的一步：**同一文本被标了不同标签的整组丢弃**（10 组 21 条），而不是保留任意一条 —— 保留哪条都是把标注噪声塞进训练集。被丢弃样本全部记进 `data/meta.json`，可人工复核。

`2124` 行 → 去重后 `2083` 条 → `train 1667 / val 416`。

### 2. 评测口径统一，否则数字不可比

四路方法如果各自 `sample()` 抽样，差 2 个点可能只是抽样波动。所以：

- 统一入口 `common/data.py::load_clean_val()`，固定 `seed=42`；
- 统一指标函数 `common/metrics.py::classification_report()`，主指标显式写 `average="macro"`；
- `eval/eval_all.py` 在汇总时**检查各结果的 `n_eval` 和 `seed`**，不一致直接告警；
- 生成式方法解析失败的样本记 `-1`，按答错计入 `acc_incl_unparsed` —— 不能靠丢失败样本虚高成绩。

### 3. 主指标用 Macro-F1，不用 Accuracy

类别不均 4.7×（`治疗方法` 368 条 vs `定义` 77 条）。只看 Accuracy 会被头部类骗：把 `治疗方法` 全猜对，尾部类全错也有 0.22。所以同时输出 Macro-F1、每类 recall 和混淆矩阵，并观察 **Macro-F1 与 Accuracy 的差值本身就是诊断信号**。

### 4. 模型永远不做「丢弃」决策

`无`、`？`、`没有` 这种 1~2 字输入在训练集里一条都没有，模型没见过这种形态，输出是随机的 —— 但 softmax 仍可能给 0.9。

所以 `drop` 队列只由 L0 规则产生（空串、长度 < 3、纯标点/数字，见 `common/labels.py::is_garbage`），挡在模型之前。模型只负责 `auto` / `review`。

### 5. 置信度必须先校准，阈值才有意义

这是本项目最有价值的一条。

BERT 在本数据上的 **ECE 高达 0.22** —— 模型说 0.9 时实际只有 0.68 命中。这种概率直接拿去和 0.85 阈值比较，等于用刻度错位的尺子量长度：要么放走一批错答案，要么把大批对的答案推给人工。

做法：在验证集上拟合温度缩放 `T`（`common/metrics.py::fit_temperature`），把 `T` 和 `max_length` 一起存进 `outputs/calibration.json` 当作**制品**，服务启动时读取。校准后 ECE 降到 0.13，阈值才可用。

TF-IDF 后端没有现成制品，就在启动时用 10% 留出集现拟合一个 `T`。

### 6. 1 667 条数据上，LoRA 打赢全量微调

596 M 参数全量更新 + 1 667 条样本 = 过拟合。LoRA 只训 0.4% 参数（`q/k/v/o_proj`），显存约 3 GB，adapter 只有 8.8 MB。全量微调靠 `lr 2e-5 + 梯度检查点` 才勉强跟上，但仍低 4 个点。

另一个必须写死的点：**训练和推理的 chat 模板必须逐字一致**，所以 `SYSTEM_PROMPT` 只放在 `common/labels.py` 一处，三个 LLM 脚本都 import 它。

---

## 目录结构

```
.
├── common/                  # 全项目共用口径（改口径只改这里）
│   ├── labels.py            # 10 类标签、别名表、分流规则、SYSTEM_PROMPT（单一来源）
│   ├── data.py              # 读取 / 去重 / 固定 seed 抽样
│   ├── metrics.py           # 统一指标 + ECE + 温度校准 + Markdown 渲染
│   └── model_paths.py       # 向上逐级查找本地模型，缺权重就报错、不偷偷联网下载
├── data/
│   ├── build_dataset.py     # 原始 CSV → train/val + meta.json（含被丢弃样本清单）
│   ├── train.json / val.json
│   └── meta.json            # 数据指纹、标签分布、去重与冲突明细
├── rule/classify_rule.py    # L0：关键词规则，顺序即优先级
├── tfidf/classify_tfidf.py  # L1：字符级 1-2gram + 线性模型
├── bert/train_bert.py       # L2：bert-base-chinese 微调（cls/mean/max 池化、类别加权、FGM 可选）
├── bert/predict_bert.py
├── llm_zero_shot/           # L3：Qwen3-0.6B 零样本
├── llm_lora/train_lora.py   # L4：LoRA / --full_ft 全量微调（Loss Masking 手写 chat 模板）
├── llm_lora/eval_lora.py
├── llm_api_zero_shot/       # L4': OpenAI 兼容 API 零样本（只改环境变量就能换服务商）
├── eval/eval_all.py         # 汇总成一张表 → outputs/summary.md
├── service/
│   ├── classify_news.py     # 分流服务（库）：L0 → 模型 → 置信度闸门 → 队列路由
│   ├── classify_cli.py      # 同一套契约的 CLI 形态，bert→tfidf→rule 三级降级，可被管道 / Agent 调用
│   └── demo_early_report.py # 喂 20 条模拟提问，看三队列长什么样（走 CLI 的 auto 降级）
```

---

## 快速开始

### 0. 安装

```bash
pip install -r requirements.txt
```

### 1. 不用任何模型权重（30 秒，验证链路通不通）

```bash
python rule/classify_rule.py            # 规则基线，416 条毫秒级
python service/demo_early_report.py     # 看 auto / review / drop 三队列
```

`demo_early_report.py` 在没有 BERT 权重时会**逐级降级**：`bert → tfidf → rule`，所以第一步就能看到真实分类结果（TF-IDF 现场训练约 2 秒），而不是满屏「无法确定」。

注意默认阈值 0.85 下 `auto` 通常是 **0 条** —— 校准后的概率被压平过，10 分类很少过线。这不是链路坏了，是阈值还没用 risk–coverage 曲线标定（见「关键设计决策 5」）。想看三队列被填满：

```bash
python service/demo_early_report.py --conf 0.6
# 合计 20 条：auto 4 / review 14 / drop 2
```

### 2. 有 CPU 就够（TF-IDF）

```bash
python tfidf/classify_tfidf.py
python service/classify_cli.py --model tfidf --text "布洛芬一天最多吃几次？"
```

### 3. 用 BERT 后端（需要权重）

```bash
python bert/train_bert.py               # 8 epoch，CPU 上较慢，建议 GPU
python service/classify_cli.py --text "布洛芬怎么吃" --json
```

### 4. 跑大模型两路（需要 GPU + 本地权重）

```bash
python llm_lora/train_lora.py                          # LoRA，1667 条 × 4 epoch
python llm_lora/eval_lora.py
python llm_zero_shot/classify_zero_shot.py
python llm_lora/train_lora.py --full_ft --gradient_checkpointing   # 8G 卡必开
```

### 5. 汇总

```bash
python eval/eval_all.py --run-light     # 先跑规则 + TF-IDF，再汇总到 outputs/summary.md
```

### 服务接口

```bash
python service/classify_cli.py --text "尖锐湿疣能治好吗？潜伏期多久？"
```

```json
{
  "title": "尖锐湿疣能治好吗？潜伏期多久？",
  "topic": "需拆解",
  "sub_topic": "多问",
  "relevant": false,
  "conf": 0.7123,
  "queue": "review",
  "model": "bert-cls-v1",
  "version": "2026-12"
}
```

下游按这个契约消费：有 `conf` 才能做低置信转人工，只有 `label` 下游没法决策。

---

## 模型与数据说明

**权重不随仓库分发**（GitHub 单文件上限 100 MB）。需要时按 `common/model_paths.py` 约定的位置放置：

```
pretrain_models/
├── bert-base-chinese/    # https://huggingface.co/google-bert/bert-base-chinese
└── Qwen3-0.6B/           # https://huggingface.co/Qwen/Qwen3-0.6B
```

脚本从自身目录向上逐级查找 `pretrain_models/<name>`，**并且校验目录里真的有权重文件** —— 否则会拿到半下载的空目录（离线环境最常见的坑）。找不到就抛错并列出所有尝试过的路径，不会静默回退到联网下载。

`data/train.json`、`data/val.json` 已附（约 270 KB）。原始 CSV 不随仓库提供，`data/build_dataset.py --csv <path>` 可重新生成，`data/meta.json` 里记了源文件 sha256。

---

## 已知问题与没做的事

诚实记一笔，这些比"已完成"列表更有用：

- **`llm_api_zero_shot/classify_api_zero_shot.py::SYSTEM` 还是上一版 TNEWS 的措辞**（"新闻标题分类助手"），医疗版本忘了改。API 那一路也没跑出结果，表里是空的。
- `service/` 下有两个同名 `ShuntService`：`classify_news.py` 只支持 BERT（无权重就退化成"无法确定"），`classify_cli.py` 支持 bert→tfidf→rule 三级降级。职责重叠，应该合并成一个、`classify_news.py` 改名。
- `service/classify_news.py` 文件名沿用自新闻分类版本，语义已经是医疗意图，建议改名 `classify_service.py`（会影响两处 import）。
- **没有做消融**。Label Smoothing / Focal Loss / FGM 三个开关都实现了但默认关，本数据上没量过效果 —— 不能把别的数据集上的结论直接搬过来。
- 没有单元测试。`common/labels.py::_self_check()` 覆盖了口径一致性（10 类、别名最长优先、`auto ⇒ relevant` 不变量），但仅此而已。
- 数据只有 2 083 条、10 类，`定义 / 临床表现 / 治疗方法 / 方法` 四类语言线索互相重叠，占 45% 的样本，是当前 F1 的主要瓶颈。
- 没有 `test` 标签集，所以没有真正的泛化数字，表里全部来自 416 条 val。

## Roadmap

- [ ] 修掉 API 脚本的 prompt 残留，补齐这条对照
- [ ] `pytest` 覆盖 `labels` / `metrics` / `data` 三个纯函数模块
- [ ] FastAPI 包一层 `/classify`，返回同款契约
- [ ] 风险–覆盖率曲线：标定 `precision ≥ 0.9` 时的最大自动放行比例（现在 0.85 是拍的）
- [ ] 近似重复检测（SimHash），当前只处理了整句完全相同

## 关于这份代码

课程项目（**八斗AI · 第六周 数据接入与分流**），在课程提供的框架与数据上完成实现、调参与评测。仓库里保留了课程讲解时的行文风格（文件头部的"教学重点"），因为那些注释记录的是**为什么这么写**，删掉反而丢信息。

如果你在作业/面试场景看到这份代码，请以"完成课程项目的实现与实验"来理解，不是从零设计的独立作品。

## License

代码 MIT。数据集与预训练模型的授权遵循其原始来源，不在此仓库的许可范围内。

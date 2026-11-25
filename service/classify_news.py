"""
分流服务：把单点模型拼成可被业务调用的接口（L5 系统组合）

教学重点：
  1. 真实系统从来不是「一个模型」，而是四段串起来：
        L0 规则兜底 → 模型分类 → 置信度闸门 → 队列路由
     每一层只负责自己最擅长的那一段。
  2. 接口契约比准确率更重要：下游要的是 {topic, sub_topic, relevant, conf, queue}，
     有了 conf 才能做「低置信转人工」；只有 label 的话下游没法决策。
  3. 降级要显式：BERT 权重不存在时退回规则法并打标记，
     工作台不会崩，但调用方能从 model 字段看出「这次是降级结果」。
  4. **置信度必须先校准再用**。本数据上 BERT 的 ECE 是 0.22，
     原始 softmax 和真实命中率差得很远；服务启动时读 outputs/calibration.json
     里的温度 T 把概率压平（ECE → 0.13），阈值才有意义。

本版本（医疗意图版）与 TNEWS 版的三处差别：
  - 队列名 research → auto（"投研"在医疗语义下名不副实）；
  - drop 只由 L0 产生（垃圾输入），模型永远不做丢弃决策；
  - 规则快筛**默认关闭**：TNEWS 里公告标题格式固定、规则很准；
    这里规则准确率只有 0.26（BERT 0.48），拿它抢答会拉低整体质量，
    所以规则只作为 --use-rules 的对照实验保留。

被谁调用：
  第 9 周会把这个函数包成 Agent 的第一个工具（Function Calling / MCP / CLI 三接法）。

用法：
  python service/classify_news.py --text "布洛芬一天最多吃几次？"
  python service/classify_news.py --file inputs.txt --use-rules
"""

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "rule"))

from common.labels import (DEFAULT_CONF_THRESHOLD, LABELS, LABEL2ID, is_garbage,
                           is_relevant, queue_name, to_group)

MODEL_NAME = "bert-cls-v1"
MODEL_VERSION = "2026-12"
FALLBACK_NAME = "fallback-v0"


class ShuntService:
    """分流服务：加载一次模型，之后每次调用复用"""

    def __init__(self, ckpt_path=None, conf_threshold=DEFAULT_CONF_THRESHOLD,
                 use_rules=False):
        self.conf_threshold = conf_threshold
        self.use_rules = use_rules
        self.ckpt_path = Path(ckpt_path) if ckpt_path else CODE_ROOT / "outputs" / "bert_ckpt.pt"
        self.model, self.tok = None, None
        self.max_length = 96
        self.temperature = 1.0
        self._try_load()
        self._load_calibration()

    def _try_load(self):
        """加载 BERT；权重缺失时**显式降级**，而不是静默出错"""
        if not self.ckpt_path.exists():
            print(f"[warn] 未找到 {self.ckpt_path}，降级为规则法（先跑 bert/train_bert.py）", flush=True)
            return
        try:
            import torch
            from transformers import AutoTokenizer
            sys.path.insert(0, str(CODE_ROOT / "bert"))
            from train_bert import DEFAULT_MAX_LENGTH, BertClassifier
            from common.model_paths import bert_path

            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            self.tok = AutoTokenizer.from_pretrained(str(bert_path()))
            self.model = BertClassifier(bert_path(), num_labels=len(LABELS),
                                        pool=ckpt.get("pool", "cls"))
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            # 训练/推理的 max_length 必须一致，否则句尾疑问词被截掉、指标凭空掉一截
            self.max_length = (ckpt.get("args") or {}).get("max_length", DEFAULT_MAX_LENGTH)
        except Exception as e:                      # 模型坏了也不能让服务整体挂掉
            print(f"[warn] BERT 加载失败（{e}），降级为规则法", flush=True)
            self.model = None

    def _load_calibration(self):
        """读温度 T。服务端不自己去读评测集 —— 校准参数和权重一样属于「制品」。"""
        p = CODE_ROOT / "outputs" / "calibration.json"
        if not p.exists():
            return
        try:
            cal = json.loads(p.read_text(encoding="utf-8"))
            self.temperature = float(cal.get("temperature") or 1.0)
            self.max_length = int(cal.get("max_length") or self.max_length)
        except Exception as e:
            print(f"[warn] calibration.json 读取失败（{e}），按 T=1.0 处理", flush=True)

    def _calibrate(self, prob):
        """单一概率向量按 T 重新归一化"""
        import numpy as np
        p = np.asarray(prob, dtype=np.float64)
        if abs(self.temperature - 1.0) < 1e-6:
            return p
        logp = np.log(np.clip(p, 1e-12, 1.0)) / self.temperature
        logp -= logp.max()
        q = np.exp(logp)
        return q / q.sum()

    def _model_predict(self, text):
        import torch
        enc = self.tok(text, truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"])
        prob = torch.softmax(logits.float(), -1)[0].numpy()
        prob = self._calibrate(prob)              # 原始 softmax 过度自信，先压平再用
        idx = int(prob.argmax())
        return idx, float(prob[idx])

    def classify(self, title: str) -> dict:
        """
        返回字段说明（下游按这个契约消费）：
          topic     粗粒度意图组（是什么/怎么办/需拆解…），给人看的
          sub_topic 10 类细标签，给统计用的
          relevant  能否自动路由（意图唯一且可定型）
          conf      校准后的置信度；< threshold 时 queue=review
          queue     auto / review / drop
          model     用了哪条路径（rule / bert-cls-v1 / fallback-v0）
        """
        text = (title or "").strip()

        # ---- L0：垃圾输入直接 drop，不送模型 ----
        # 训练集里没有长度 <3 的样本，模型没见过这种形态，会给随机答案（且可能很自信）
        if is_garbage(text):
            return self._pack(title, LABEL2ID["无法确定"], 0.0, "rule-l0", queue="drop")

        # ---- 规则快筛（默认关闭，见文件头说明） ----
        if self.use_rules:
            from classify_rule import rule_match
            hit = rule_match(text)
            if hit is not None:
                # 规则命中的置信度是人为设定的，不是模型输出：这里给 0.99 会直接放行，
                # 而规则准确率只有 0.26，所以只给一个「刚够 review」的分，交给下游看 model 字段
                return self._pack(title, hit, 0.60, "rule-v1")

        # ---- 模型 ----
        if self.model is not None:
            idx, conf = self._model_predict(text)
            return self._pack(title, idx, conf, MODEL_NAME)

        # ---- 降级：没有模型也没有规则命中 → 无法定型，全部进 review ----
        return self._pack(title, LABEL2ID["无法确定"], 0.0, FALLBACK_NAME)

    def _pack(self, title, idx, conf, model_used, queue=None):
        return {
            "title": title,
            "topic": to_group(idx),
            "sub_topic": LABELS[idx],
            "relevant": is_relevant(idx),
            "conf": round(float(conf), 4),
            "queue": queue or queue_name(idx, conf, self.conf_threshold),
            "model": model_used,
            "version": MODEL_VERSION,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="单条问题")
    ap.add_argument("--file", help="每行一条的文件")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--conf_threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    ap.add_argument("--use-rules", action="store_true", help="打开规则快筛（对照用）")
    args = ap.parse_args()

    svc = ShuntService(args.ckpt, args.conf_threshold, args.use_rules)
    if svc.model is not None:
        print(f"[info] 后端={MODEL_NAME} 温度 T={svc.temperature} max_length={svc.max_length} "
              f"阈值={svc.conf_threshold}", flush=True)
    texts = [args.text] if args.text else Path(args.file).read_text(encoding="utf-8").splitlines()
    for t in [x for x in texts if x.strip()]:
        print(json.dumps(svc.classify(t.strip()), ensure_ascii=False))


if __name__ == "__main__":
    main()

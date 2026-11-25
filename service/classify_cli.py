"""
分流服务的命令行外壳（把服务做成人和 Agent 都能调的工具）

和 classify_news.py 的分工：
  classify_news.py  是「库」——别的脚本 import 它（demo 就是）；
  classify_cli.py   是「工具」——命令行直接喂一条/一批，输出一行一行 JSON。

教学重点：
  1. 同一套契约，两种使用形态。CLI 的价值在于「能被管道和 Agent 调」：
     `cat questions.txt | python service/classify_cli.py --json` 就是一个工具。
  2. 三种后端按可靠性降级，**失败要显式**：
        bert    精度最高（val Macro-F1 0.474），要 409 MB 权重
        tfidf   20 秒现训，val 0.377，无模型依赖（内网/合规场景的退路）
        rule    纯规则，val 0.222，只用于演示接口形态
     `--model auto` 依次尝试；`--model bert` 失败直接退出，不允许静默降级。
  3. 置信度必须校准后才敢当阈值用：BERT 的 T 从 outputs/calibration.json 读，
     TF-IDF 的 T 在启动时用 10% 留出集现拟合。

用法：
  python service/classify_cli.py                          # 交互模式
  python service/classify_cli.py --text "布洛芬怎么吃"      # 单条
  python service/classify_cli.py --text "..." --json       # 单行 JSON（给脚本用）
  python service/classify_cli.py --file questions.txt      # 批量
  python service/classify_cli.py --file -                  # 从 stdin 读
  python service/classify_cli.py --model tfidf             # 强制线性后端
  python service/classify_cli.py --use-rules               # 打开规则快筛
"""

import argparse
import json
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "rule"))

from common.labels import (DEFAULT_CONF_THRESHOLD, LABELS, LABEL2ID, is_garbage,
                           is_relevant, queue_name, to_group)

VERSION = "2026-12"
BACKEND_NAME = {
    "bert": "bert-cls-v1",
    "tfidf": "tfidf-sgd-v1",
    "rule": "rule-v1",
    "fallback": "fallback-v0",
}


class ShuntService:
    def __init__(self, model="auto", ckpt=None, conf_threshold=DEFAULT_CONF_THRESHOLD,
                 use_rules=False, verbose=False):
        self.conf_threshold = conf_threshold
        self.use_rules = use_rules
        self.verbose = verbose
        self.backend = "fallback"
        self.temperature = 1.0
        self.max_length = 96
        self._bert = self._tok = None
        self._vec = self._clf = None

        want = model
        if want in ("auto", "bert") and self._try_init_bert(ckpt):
            self.backend = "bert"
        elif want in ("auto", "tfidf") and self._init_tfidf():
            self.backend = "tfidf"
        elif want == "bert":
            raise SystemExit("[error] --model bert 但 BERT 后端不可用（检查 outputs/bert_ckpt.pt）")
        elif want == "tfidf":
            raise SystemExit("[error] --model tfidf 但 TF-IDF 后端初始化失败")
        else:
            self.backend = "rule"
            print("[info] 没有可用模型，退化到规则后端（结果仅供演示接口）", file=sys.stderr)

    # ---------------- 后端 1：BERT ----------------
    def _try_init_bert(self, ckpt_path):
        p = Path(ckpt_path) if ckpt_path else CODE_ROOT / "outputs" / "bert_ckpt.pt"
        if not p.exists():
            if self.verbose:
                print(f"[info] 无 {p.name}，跳过 BERT 后端", file=sys.stderr)
            return False
        try:
            import torch
            from transformers import AutoTokenizer
            sys.path.insert(0, str(CODE_ROOT / "bert"))
            from train_bert import DEFAULT_MAX_LENGTH, BertClassifier
            from common.model_paths import bert_path

            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            self._tok = AutoTokenizer.from_pretrained(str(bert_path()))
            self._bert = BertClassifier(bert_path(), num_labels=len(LABELS),
                                        pool=ckpt.get("pool", "cls"))
            self._bert.load_state_dict(ckpt["state_dict"])
            self._bert.eval()
            self.max_length = (ckpt.get("args") or {}).get("max_length", DEFAULT_MAX_LENGTH)
            self._load_temperature()
            return True
        except Exception as e:
            print(f"[warn] BERT 初始化失败（{e}）", file=sys.stderr)
            return False

    def _load_temperature(self):
        p = CODE_ROOT / "outputs" / "calibration.json"
        if not p.exists():
            return
        try:
            cal = json.loads(p.read_text(encoding="utf-8"))
            self.temperature = float(cal.get("temperature") or 1.0)
            self.max_length = int(cal.get("max_length") or self.max_length)
        except Exception:
            pass

    def _predict_bert(self, text):
        import numpy as np
        import torch
        enc = self._tok(text, truncation=True, max_length=self.max_length,
                        padding="max_length", return_tensors="pt")
        with torch.no_grad():
            logits = self._bert(enc["input_ids"], enc["attention_mask"])
        p = torch.softmax(logits.float(), -1)[0].numpy().astype(np.float64)
        if abs(self.temperature - 1.0) > 1e-6:
            logp = np.log(np.clip(p, 1e-12, 1.0)) / self.temperature
            logp -= logp.max()
            p = np.exp(logp) / np.exp(logp).sum()
        idx = int(p.argmax())
        return idx, float(p[idx])

    # ---------------- 后端 2：TF-IDF + SGD ----------------
    def _init_tfidf(self):
        """
        为什么这里用 SGDClassifier(log_loss) 而不是 SVM：
        LinearSVC 没有 predict_proba，拿不到「置信度」，也就没法做 queue 分流。
        log_loss 训练出的线性模型能直接给概率，代价是准确率略低一点。
        """
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import SGDClassifier
            from common.data import load_splits
            from common.metrics import apply_temperature, fit_temperature

            train, _ = load_splits()
            rng = np.random.RandomState(42)
            idx = rng.permutation(len(train))
            n_hold = max(50, int(len(train) * 0.1))
            hold, fit = idx[:n_hold], idx[n_hold:]

            self._vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2),
                                        max_features=300000, min_df=2, sublinear_tf=True)
            X = self._vec.fit_transform([train[i]["sentence"] for i in fit])
            self._clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30,
                                      class_weight="balanced", random_state=42)
            self._clf.fit(X, [train[i]["label"] for i in fit])

            # 用 10% 留出集拟合温度：不校准的 SGD 概率偏平，0.85 阈值几乎拦不下东西
            Xh = self._vec.transform([train[i]["sentence"] for i in hold])
            proba = self._clf.predict_proba(Xh)
            yh = [train[i]["label"] for i in hold]
            self.temperature = fit_temperature(proba, yh)[0]
            self._apply_t = apply_temperature
            return True
        except Exception as e:
            print(f"[warn] TF-IDF 初始化失败（{e}）", file=sys.stderr)
            return False

    def _predict_tfidf(self, text):
        import numpy as np
        proba = self._clf.predict_proba(self._vec.transform([text]))[0]
        proba = self._apply_t(proba, self.temperature)
        idx = int(np.argmax(proba))
        return idx, float(proba[idx])

    # ---------------- 统一入口 ----------------
    def classify(self, title: str) -> dict:
        text = (title or "").strip()
        t0 = time.time()

        # L0：垃圾输入直接 drop，不送模型（模型没见过这种形态，会给随机且自信的答案）
        if is_garbage(text):
            return self._pack(title, LABEL2ID["无法确定"], 0.0, "rule", queue="drop",
                              latency=time.time() - t0)

        if self.use_rules:
            from classify_rule import rule_match
            hit = rule_match(text)
            if hit is not None:
                return self._pack(title, hit, 0.60, "rule", latency=time.time() - t0)

        if self.backend == "bert":
            idx, conf = self._predict_bert(text)
            return self._pack(title, idx, conf, "bert", latency=time.time() - t0)
        if self.backend == "tfidf":
            idx, conf = self._predict_tfidf(text)
            return self._pack(title, idx, conf, "tfidf", latency=time.time() - t0)

        from classify_rule import rule_match
        idx = rule_match(text)
        if idx is None:
            idx, conf = LABEL2ID["无法确定"], 0.0
        else:
            conf = 0.60
        return self._pack(title, idx, conf, "rule" if rule_match(text) is not None else "fallback",
                          latency=time.time() - t0)

    def _pack(self, title, idx, conf, path, queue=None, latency=0.0):
        out = {
            "title": title,
            "topic": to_group(idx),
            "sub_topic": LABELS[idx],
            "relevant": is_relevant(idx),
            "conf": round(float(conf), 4),
            "queue": queue or queue_name(idx, conf, self.conf_threshold),
            "model": BACKEND_NAME[path],
            "version": VERSION,
        }
        if self.verbose:
            out["_debug"] = {
                "backend": self.backend,
                "label_id": idx,
                "latency_ms": round(latency * 1000, 2),
                "threshold": self.conf_threshold,
                "temperature": self.temperature,
            }
        return out


_DEFAULT = None


def classify(text, **kw):
    """函数式入口：单条调用，自动复用同一个 service 实例"""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ShuntService(**kw)
    return _DEFAULT.classify(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="单条文本")
    ap.add_argument("--file", help="每行一条的文件，- 表示 stdin")
    ap.add_argument("--model", default="auto", choices=["auto", "bert", "tfidf", "rule"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD)
    ap.add_argument("--use-rules", action="store_true")
    ap.add_argument("--json", action="store_true", help="单行紧凑 JSON（脚本/Agent 用）")
    ap.add_argument("--verbose", action="store_true", help="附带 _debug 字段")
    args = ap.parse_args()

    svc = ShuntService(args.model, args.ckpt, args.conf, args.use_rules, args.verbose)
    print(f"[info] 后端={BACKEND_NAME[svc.backend]} T={svc.temperature:.2f} "
          f"阈值={svc.conf_threshold}", file=sys.stderr)

    def emit(r):
        print(json.dumps(r, ensure_ascii=False, separators=(",", ":")) if args.json
              else json.dumps(r, ensure_ascii=False, indent=2))

    if args.text:
        emit(svc.classify(args.text))
        return

    if args.file:
        lines = (sys.stdin.read().splitlines() if args.file == "-"
                 else Path(args.file).read_text(encoding="utf-8").splitlines())
        for ln in lines:
            if ln.strip():
                emit(svc.classify(ln.strip()))
        return

    if not sys.stdin.isatty():
        for ln in sys.stdin.read().splitlines():
            if ln.strip():
                emit(svc.classify(ln.strip()))
        return

    print("交互模式：输入问题回车，q 退出", file=sys.stderr)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("q", "quit", "exit"):
            break
        if line:
            emit(svc.classify(line))


if __name__ == "__main__":
    main()

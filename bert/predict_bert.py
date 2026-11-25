"""
方法三（推理）：加载 BERT checkpoint，在干净验证集上评测

教学重点：
  1. 评测必须在**去重后的同一批样本**上跑，否则和 TF-IDF / LLM 不可比。
  2. 顺手测延迟：单条 vs batch=32 差 3 倍多（实测 394ms → 121ms/条）——
     「模型多大」不是成本，「能不能批处理」才是。
  3. 保存逐条预测，方便和规则/TF-IDF/LLM 做三方对照与错误分析。

用法：
  python bert/predict_bert.py                       # 默认 outputs/bert_ckpt.pt
  python bert/predict_bert.py --ckpt path/to/ckpt.pt --n 500
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val
from common.labels import LABELS, NUM_LABELS
from common.metrics import (apply_temperature, classification_report,
                            expected_calibration_error, fit_temperature,
                            to_markdown_table)
from common.model_paths import bert_path
from train_bert import BertClassifier, IntentDataset, DEFAULT_MAX_LENGTH

CODE_ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def predict(model, tok, texts, device, max_length=DEFAULT_MAX_LENGTH, batch_size=32):
    """返回 (preds, probs, 每样本毫秒数)"""
    preds, probs = [], []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                  padding="max_length", return_tensors="pt")
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        p = torch.softmax(logits.float(), dim=-1)
        preds.extend(p.argmax(-1).cpu().tolist())
        probs.extend(p.cpu().tolist())
    ms = (time.time() - t0) * 1000 / max(1, len(texts))
    return preds, probs, round(ms, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CODE_ROOT / "outputs" / "bert_ckpt.pt"))
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部干净样本")
    ap.add_argument("--out", default="outputs/bert_pred.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)   # torch 2.6 默认 weights_only=True，会拒载字典
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(bert_path()))
    model = BertClassifier(bert_path(), num_labels=NUM_LABELS, pool=ckpt.get("pool", "cls"))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # max_length 必须和训练时一致：训练用 96 反而推理用 64 会截掉句尾疑问词，指标凭空掉一截
    max_length = (ckpt.get("args") or {}).get("max_length", DEFAULT_MAX_LENGTH)

    samples, meta = load_clean_val(args.n or None)
    texts = [d["sentence"] for d in samples]
    preds, probs, ms = predict(model, tok, texts, device, max_length=max_length)

    y_true = [d["label"] for d in samples]

    # ---- 温度校准：在 val 上拟合 T，写进 calibration.json 供服务层读取 ----
    # 服务端不该自己去读评测集，否则「阈值」和「评测集」隐式耦合，
    # 换个数据集就得改服务代码。所以校准参数和权重一样，属于「制品」。
    temperature, nll = fit_temperature(probs, y_true)
    ece_raw = expected_calibration_error(y_true, probs)
    ece_cal = expected_calibration_error(y_true, apply_temperature(probs, temperature))

    report = classification_report(y_true, preds)
    report.update({
        "method": "bert_finetune",
        "pool": ckpt.get("pool"),
        "ckpt_val_f1": ckpt.get("val_macro_f1"),
        "ms_per_sample": ms,
        "ece": ece_cal,
        "ece_raw": ece_raw,
        "temperature": temperature,
    })

    out = CODE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report,
                               "preds": [{"text": t, "true": y, "pred": p}
                                         for t, y, p in zip(texts, y_true, preds)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    calib = CODE_ROOT / "outputs" / "calibration.json"

    # ---- 阈值标定（risk–coverage）：队列 auto 的准确率必须够高，否则宁可多转人工 ----
    # 衡量的是「按置信度从高到低放行时，放行部分的准确率」，
    # 目标 precision ≥ 0.90 下取最大覆盖率 —— 这就是 queue 阈值该取多少的依据。
    import numpy as np
    probs_cal = apply_temperature(probs, temperature)
    conf = probs_cal.max(axis=1)
    correct = (probs_cal.argmax(axis=1) == np.asarray(y_true))
    curve, best = [], None
    for t in [round(0.40 + 0.05 * i, 2) for i in range(12)]:
        mask = conf >= t
        if not mask.any():
            continue
        cov, prec = float(mask.mean()), float(correct[mask].mean())
        curve.append({"threshold": t, "coverage": round(cov, 4), "precision": round(prec, 4)})
        if prec >= 0.90 and (best is None or cov > best["coverage"]):
            best = curve[-1]

    calib.write_text(json.dumps({
        "model": "bert-cls-v1",
        "temperature": temperature,
        "n_calib": len(samples),
        "nll": nll,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "max_length": max_length,
        "pool": ckpt.get("pool"),
        "seed": meta.get("seed"),
        "risk_coverage": curve,
        "recommended_threshold": best,
        "note": "T 与阈值都在 val(416) 上拟合/标定，val 同时用于评测，指标略偏乐观",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {meta['n_eval']} 条（干净集）| {ms} ms/条 | "
          f"ECE {ece_raw} → {ece_cal}（T={temperature}）")
    print(to_markdown_table({"BERT 微调": report}))
    print(f"→ 结果写入 {out}")
    print(f"→ 校准参数写入 {calib}")


if __name__ == "__main__":
    main()

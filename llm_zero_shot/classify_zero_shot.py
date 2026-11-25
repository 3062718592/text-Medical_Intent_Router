"""
方法四：大模型零样本（L3，不训练）

教学重点：
  1. 零样本 = 把分类当成「生成」来做：写一句 prompt，让模型直接吐类别名。
     不用数据、不用训练，冷启动最快——这是它唯一的优势。
  2. 但它**不擅长细粒度意图分类**：10 个意图里有 4 组语义相邻
     （定义/临床表现、治疗方法/方法、作用/适用症、多问/其余），
     0.6B 的基座模型没有「按业务口径理解这几个词」的能力，容易塌缩到少数类。
     这是零样本的真实定位：**先跑通链路，不指望它能上线**。
  3. Qwen3 必须传 enable_thinking=False。不传的话模型会先吐一段
     <think>...</think> 推理链，把「只输出类别名」的格式彻底打乱。
     这是 Qwen2 → Qwen3 迁移时最容易踩的坑。
  4. 类别名必须连**定义**一起写进 prompt：只给 10 个词（尤其"方法"/"适用症"这种），
     模型只能猜；给了「适用症：这药我能不能用」，它才可能对上。

用法：
  python llm_zero_shot/classify_zero_shot.py              # 默认 200 条
  python llm_zero_shot/classify_zero_shot.py --n 416      # 全量 val
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val
from common.labels import LABELS, SYSTEM_PROMPT, match_label
from common.metrics import classification_report, pred_distribution, to_markdown_table
from common.model_paths import qwen3_path

CODE_ROOT = Path(__file__).resolve().parent.parent

# Prompt 从 common.labels 取单一来源：三个 LLM 脚本必须逐字一致，
# 否则 zero-shot / LoRA / API 的结果不可比（SFT 尤其敏感，训练与推理模板必须相同）
SYSTEM = SYSTEM_PROMPT


def parse_label(text: str) -> int:
    """从生成文本里抽类别名；抽不到返回 -1（记为解析失败）。

    解析统一走 common.labels.match_label（最长别名优先），
    这样 zero-shot / LoRA / API 三个脚本对同一段输出给出同样的标签。
    """
    return match_label(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--out", default="outputs/zero_shot_pred.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(qwen3_path()))
    model = AutoModelForCausalLM.from_pretrained(
        str(qwen3_path()), dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    samples, meta = load_clean_val(args.n)
    preds, raws, t0 = [], [], time.time()
    for i, d in enumerate(samples):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": d["sentence"]}]
        # enable_thinking=False 必须在模板阶段传：Qwen3 会在 assistant 前插入空 think 块
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        raws.append(raw)
        preds.append(parse_label(raw))
        if i % 50 == 0:
            print(f"  {i}/{len(samples)} {time.time()-t0:.0f}s", flush=True)

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, preds)
    report["method"] = "qwen3_zero_shot"
    report["sec_per_item"] = round((time.time() - t0) / len(samples), 2)
    report["pred_distribution"] = pred_distribution(preds)   # 塌缩的证据

    out = CODE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report,
                               "preds": [{"text": d["sentence"], "true": d["label"], "pred": p, "raw": r}
                                         for d, p, r in zip(samples, preds, raws)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {meta['n_eval']} 条 | {report['sec_per_item']}s/条")
    print(to_markdown_table({"Qwen3 zero-shot": report}))
    print(f"预测覆盖类别数：{report['pred_distribution']['used_classes']}/{len(LABELS)}"
          f" | 解析失败 {report['unparsed']} 条")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()

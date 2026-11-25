"""
方法五（推理）：加载 LoRA adapter，在干净验证集上评测

教学重点：
  1. 推理必须和训练用**同一个 chat 模板 + 同一个 enable_thinking=False**，
     否则格式错位，准确率会莫名掉一半。所以 SYSTEM 从 common.labels 取，不另写。
  2. 三方对比（同一批 416 条 val）：
       Qwen3 zero-shot  0.178 / 0.143
       LoRA SFT         见 outputs/lora_pred.json
       BERT 微调        0.478 / 0.474
     预期结论：SFT 修好了塌缩，但在「封闭标签 + 短文本」上仍不如判别式模型——
     这就是选型的答案：不是谁更强，是谁更合适。
  3. 解析失败率要单独统计（-1 的条数）。生成式模型总会有不听话的时候，
     这个数字直接决定你要不要加约束解码。

用法：
  python llm_lora/eval_lora.py                       # 默认 outputs/lora_adapter
  python llm_lora/eval_lora.py --adapter outputs/full_ft_ckpt --out outputs/full_ft_pred.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val
from common.labels import LABELS, SYSTEM_PROMPT, match_label
from common.metrics import classification_report, pred_distribution, to_markdown_table
from common.model_paths import qwen3_path

CODE_ROOT = Path(__file__).resolve().parent.parent

SYSTEM = SYSTEM_PROMPT


def parse_label(text: str) -> int:
    """统一走 common.labels.match_label（最长别名优先），与训练/其它 LLM 脚本同源"""
    return match_label(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(CODE_ROOT / "outputs" / "lora_adapter"))
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部 val")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--out", default="outputs/lora_pred.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(qwen3_path()))
    ckpt_dir = Path(args.adapter)
    if (ckpt_dir / "adapter_config.json").exists():
        # LoRA：先载基座，再挂 adapter
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            str(qwen3_path()), dtype=torch.float16 if device == "cuda" else torch.float32)
        model = PeftModel.from_pretrained(base, str(ckpt_dir)).to(device).eval()
        kind = "lora"
    else:
        # 全量微调：目录里就是完整权重，直接 from_pretrained
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir), dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        kind = "full_ft"
    print(f"[eval] 加载方式：{kind}（{ckpt_dir}）", flush=True)

    samples, meta = load_clean_val(args.n or None)
    preds, raws = [], []
    for i, d in enumerate(samples):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["sentence"]}]
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
            print(f"  {i}/{len(samples)}", flush=True)

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, preds)
    report["method"] = "qwen3_lora_sft" if kind == "lora" else "qwen3_full_ft"
    report["pred_distribution"] = pred_distribution(preds)

    out = CODE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report,
                               "preds": [{"text": d["sentence"], "true": d["label"], "pred": p, "raw": r}
                                         for d, p, r in zip(samples, preds, raws)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {meta['n_eval']} 条（干净集）")
    print(to_markdown_table({f"Qwen3 + {kind}": report}))
    print(f"预测覆盖类别数：{report['pred_distribution']['used_classes']}/{len(LABELS)}"
          f" | 解析失败 {report['unparsed']} 条")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()

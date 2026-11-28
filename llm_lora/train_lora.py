"""
方法五：Qwen3-0.6B 指令微调（LoRA 与全量微调两种模式）

教学重点：
  1. SFT 的关键不是模型，而是三件事：**chat 模板、Loss Masking、数据格式**。
  2. Loss Masking：prompt 部分 labels 全设 -100（不算 loss），只在答案 token 上算；
     不 mask 的话模型学的是「复述模板」，不是「答标签」。
  3. 两种微调模式的分工：
        LoRA（默认）     只训低秩旁路（约 0.4%），显存约 3 GB，adapter 十几 MB
        全量微调 --full_ft  596M 参数全更新，显存 6~8 GB（8G 卡要开梯度检查点）
  4. Qwen3 必须 enable_thinking=False，且训练与推理的模板必须**逐字一致**——
     所以 SYSTEM 从 common.labels 取，不在这里另写一份。
  5. 本数据的排序坑：老版本用 `train[:num_train]` 取前 N 条，恰好把稀有类切掉
     （TNEWS 前 5000 条只有 23 条"证券"）。现在 1667 条全用，不存在这个偏差，
     但仍保留 --num_train 供"数据量 vs 效果"的对照实验（0=全用）。

用法：
  python llm_lora/train_lora.py                          # LoRA，1667 条 × 4 epoch
  python llm_lora/train_lora.py --num_train 500 --epochs 2   # 数据量对照
  python llm_lora/train_lora.py --full_ft --epochs 2 --gradient_checkpointing
  python llm_lora/train_lora.py --max_steps 20           # 冒烟测流程
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_splits
from common.labels import LABELS, SYSTEM_PROMPT
from common.model_paths import qwen3_path

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = CODE_ROOT / "outputs"

SYSTEM = SYSTEM_PROMPT


class SFTDataset(Dataset):
    """
    把分类样本转成 chat 格式的 SFT 样本。

    labels 的构造是本文件最核心的一行：
      prompt 部分 = -100（CrossEntropyLoss 会忽略 -100）
      答案部分   = 真实 token id（只有这里产生梯度）
    """

    def __init__(self, data, tok, max_length=512):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        answer = LABELS[d["label"]]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["sentence"]}]
        # tokenize=False 先拿文本：transformers 5.x 里 tokenize=True 返回 BatchEncoding，不好拼
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        input_ids = (p_ids + a_ids)[: self.max_length]
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_length]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


def collate(batch, pad_id):
    """手写 collate：input_ids 用 pad 补齐，labels 用 -100 补齐（PAD 不产生 loss）"""
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        lab[i, :n] = b["labels"]
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_train", type=int, default=0, help="0=全部训练样本")
    ap.add_argument("--data", default="train.json",
                    help="训练集文件名：train.json（原始 1667）或 train_aug.json（回译增强）；验证集恒为 val.json")
    ap.add_argument("--tag", default="",
                    help="产物后缀，如 aug → lora_adapter_aug / lora_train_log_aug.json，避免覆盖基线")
    ap.add_argument("--save_best", action="store_true",
                    help="每个 epoch 结束时若 val_loss 创新低就额外存一份 <产物>_best；配合早停使用")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4, help="全量微调建议 1~2")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="不填则 LoRA=2e-4、全量=2e-5")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--full_ft", action="store_true", help="全量微调；不加则 LoRA")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="省显存（全量微调在 8G 卡上必开）")
    ap.add_argument("--max_length", type=int, default=320,
                    help="要装得下 system prompt（约 245 token）；太短会把答案截掉，也没必要给太大余量")
    ap.add_argument("--max_steps", type=int, default=0, help=">0 时只跑这么多步（冒烟测试）")
    ap.add_argument("--val_n", type=int, default=200, help="训练中算 val_loss 用多少条，只影响耗时不影响结果")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 统一用 bf16：fp16 需要 GradScaler 防下溢，多一层状态、也多一处可能踩坑的地方；
    # 4060 支持 bf16，LoRA 在全量精度下反而更稳。CPU 上只能 fp32。
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(str(qwen3_path()))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train, val = load_splits(args.data)
    if args.num_train:
        train = train[: args.num_train]
    val = val[: min(args.val_n, len(val))]   # 训练中只看 val_loss，条数够用就行、省时间

    # 长度检查：prompt 占掉绝大部分 token，max_length 不够会把答案整段截掉，
    # 那时 loss 只来自 -100，训练会「静默失效」（loss 不降但不报错，甚至 NaN）
    probe = SFTDataset(train[:1], tok, args.max_length)
    n_prompt = sum(1 for x in probe[0]["labels"].tolist() if x == -100)
    n_answer = len(probe[0]["labels"]) - n_prompt
    _, _, label_raw = train[0]["sentence"], train[0]["label"], LABELS[train[0]["label"]]
    n_answer_need = len(tok(label_raw, add_special_tokens=False)["input_ids"]) + 1
    if n_answer < n_answer_need:
        raise SystemExit(
            f"max_length={args.max_length} 太小：prompt 就占 {n_prompt} token，"
            f"答案需要 {n_answer_need} 个，实际只剩 {n_answer} 个（答案会被截断→训练静默失效）")
    print(f"[data] train={len(train)} val={len(val)} "
          f"prompt≈{n_prompt} token 答案={n_answer} 总长={len(probe[0]['labels'])}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(str(qwen3_path()), dtype=dtype)

    if args.full_ft:
        model = base.to(device)                     # 所有参数默认 requires_grad=True
        if args.gradient_checkpointing:
            # 梯度检查点：用时间换显存，激活值不再常驻（全量微调在 8G 卡上基本必开）
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        lr = args.lr or 2e-5                        # 全量更新必须用更小的 lr，否则冲掉预训练知识
        print(f"[full-ft] 参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M"
              f"（全部可训练），lr={lr}", flush=True)
    else:
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(base, cfg).to(device)
        lr = args.lr or 2e-4
        # 一定要打印：target_modules 写错名字不会报错，只会「训了等于没训」
        model.print_trainable_parameters()

    train_dl = DataLoader(SFTDataset(train, tok, args.max_length), batch_size=args.batch_size,
                          shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
    val_dl = DataLoader(SFTDataset(val, tok, args.max_length), batch_size=args.batch_size,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    # bf16 不需要 GradScaler（它的动态范围与 fp32 同量级，不存在 fp16 的下溢问题）
    log, step = [], 0
    best_vloss, best_epoch = None, 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batch = 0.0, 0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum          # 梯度累积：等效放大 batch
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            running += loss.item() * args.grad_accum
            n_batch += 1
            if i % 50 == 0:
                print(f"  epoch {epoch} {i}/{len(train_dl)} loss={loss.item()*args.grad_accum:.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                vloss += model(**batch).loss.item()
                vn += 1
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n_batch), 4),
                    "val_loss": round(vloss / max(1, vn), 4), "steps": step,
                    "elapsed_s": round(time.time() - t_start, 1),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else None})
        print(f"epoch {epoch}: train_loss={log[-1]['train_loss']} val_loss={log[-1]['val_loss']} "
              f"steps={step} {log[-1]['elapsed_s']}s mem={log[-1]['peak_mem_gb']}GB", flush=True)

        # best-ckpt：val_loss 创新低就存一份，训练结束再从里面挑。
        # 为什么需要：本数据上 LoRA 到 epoch 3~4 就开始过拟合（val_loss 回升），
        # 而只存最后一轮等于把最差的那个交出去 —— 白跑几十分钟。
        if args.save_best and (best_vloss is None or log[-1]["val_loss"] < best_vloss):
            best_vloss, best_epoch = log[-1]["val_loss"], epoch
            _tag = ("_" + args.tag) if args.tag else ""
            _best_dir = OUT_DIR / (("full_ft_ckpt" if args.full_ft else "lora_adapter") + _tag + "_best")
            model.save_pretrained(str(_best_dir))
            print(f"  [best] epoch {epoch} val_loss={best_vloss} → {_best_dir.name}", flush=True)

        if args.max_steps:
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # tag 用于区分不同数据/超参下的产物，避免覆盖基线：
    #   基线（1667 条）    → outputs/lora_adapter             + lora_train_log.json
    #   增强（3196 条）    → outputs/lora_adapter_aug         + lora_train_log_aug.json
    tag = ("_" + args.tag) if args.tag else ""
    if args.full_ft:
        model.save_pretrained(str(OUT_DIR / ("full_ft_ckpt" + tag)))
        log_path = OUT_DIR / ("full_ft_train_log%s.json" % tag)
        print(f"全量权重 → {OUT_DIR / ('full_ft_ckpt' + tag)}")
    else:
        model.save_pretrained(str(OUT_DIR / ("lora_adapter" + tag)))   # 只存 adapter，十几 MB
        log_path = OUT_DIR / ("lora_train_log%s.json" % tag)
        print(f"adapter → {OUT_DIR / ('lora_adapter' + tag)}")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

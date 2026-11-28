# -*- coding: utf-8 -*-
"""
数据增强：中→英→中 回译（back-translation）

为什么用回译而不是同义词替换：
  同义词替换只改词面，句子结构不变，模型学到的还是同一种句法；
  回译经过一次「用另一种语言重新表达」，能带来真正的句式多样性（语序、主动被动、
  拆分合并）。代价是语义可能漂移，所以生成后必须做去重 + 长度/相似度过滤。

为什么只增强训练集：
  验证集必须保持人工标注的原始分布。哪怕只混进 1 条增强样本，
  所有方法在这 416 条上的对比数字就全部失效 —— 这是红线。

标注口径（必须写进 README，否则等于数据造假）：
  这批样本是「增强样本」，不是人工标注。最终数据集要能拆成：
      人工标注 1667  +  伪标签 N  +  回译增强 M  =  训练集总数
  同时把来源写进 data/meta.json 的 provenance 字段。

支持的引擎：
  deepseek   OpenAI 兼容接口（推荐，复用项目里的 LLM_API_KEY 范式）
  baidu      百度翻译开放平台（MD5 签名，免费 5 万字符/月）
  tencent    腾讯云机器翻译 TMT（TC3-HMAC-SHA256 签名，免费 500 万字符/月）
  mock       不联网，用固定规则伪造一遍流程（验证脚本用，--dry-run 也行）

环境变量：
  deepseek:  LLM_API_KEY / DEEPSEEK_API_KEY, LLM_BASE_URL(默认 https://api.deepseek.com),
             LLM_MODEL(默认 deepseek-chat)
  baidu:     BAIDU_APPID, BAIDU_KEY
  tencent:   TENCENT_SECRET_ID, TENCENT_SECRET_KEY, TENCENT_REGION(默认 ap-guangzhou)

用法：
  python data/augment_backtranslate.py --engine mock --limit 30        # 先看流程
  python data/augment_backtranslate.py --engine deepseek --target 2500 # 补到约 2500 条
  python data/augment_backtranslate.py --engine baidu --batch 10       # 10 条一次请求
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_ROOT / "data"
CACHE = DATA_DIR / "cache_backtranslate.jsonl"      # 断点续跑用，一行一条
OUT = DATA_DIR / "aug_backtranslated.json"


# ---------------------------------------------------------------- 缓存 / 断点
def load_cache():
    """返回 {原文: [回译1, 回译2, ...]}"""
    got = {}
    if not CACHE.exists():
        return got
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        got.setdefault(d["src"], []).append(d["dst"])
    return got


def append_cache(src, dst, engine):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"src": src, "dst": dst, "engine": engine},
                           ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 引擎：mock
def make_mock():
    def tr(texts, src_lang, tgt_lang):
        # 只是把句子反转 + 加标记，证明流程通；不产生任何可用的增强数据
        return ["<mock:%s>%s" % (tgt_lang, t[::-1]) for t in texts]
    return tr


# ---------------------------------------------------------------- 引擎：DeepSeek
def make_deepseek(batch_size):
    from openai import OpenAI

    key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("缺少 LLM_API_KEY（或 DEEPSEEK_API_KEY）")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    # 模型名跟着网关走：官方 DeepSeek 用 deepseek-chat；
    # 课程内网网关（model.mify.ai.srv）用 deepseek-v4-flash。
    # 不写死，是因为写死之后换网关会得到一个 404 model_not_found，排查起来很费劲。
    default_model = ("deepseek-v4-flash" if "mify" in base_url else "deepseek-chat")
    model = os.getenv("LLM_MODEL", default_model)
    client = OpenAI(api_key=key, base_url=base_url, timeout=60, max_retries=3)
    print("  引擎 deepseek | base_url=%s | model=%s" % (base_url, model))
    # 温度从外部注入：第 1 条变体用 T=0（最忠实的译文），
    # 第 2 条起必须升温，否则模型会给出和第一条逐字相同的结果（白花钱）。
    temp = [0.0]

    def tr(texts, src_lang, tgt_lang):
        lang = {"zh": "中文", "en": "英文"}[tgt_lang]
        numbered = "\n".join("%d. %s" % (i + 1, t) for i, t in enumerate(texts))
        sys_prompt = ("你是翻译。把下面的医疗问句逐条翻译成%s，"
                      "保持编号和行数不变，不要解释、不要合并、不要增删条目。"
                      "只输出译文列表。" % lang)
        r = client.chat.completions.create(
            model=model, temperature=temp[0],
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": numbered}],
        )
        raw = (r.choices[0].message.content or "").strip()
        # 解析 "1. xxx" / "1) xxx" / "1、xxx"
        out = []
        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            for sep in ("、", ")", "）", "."):
                head, dot, rest = s.partition(sep)
                if dot and head.strip().isdigit():
                    s = rest.strip()
                    break
            out.append(s)
        if len(out) != len(texts):       # 条数对不上就整批丢弃，绝不猜
            raise ValueError("返回 %d 条，期望 %d 条" % (len(out), len(texts)))
        return out

    tr._batch = batch_size

    def _set_temp(t):
        temp[0] = t
    tr._set_temp = _set_temp
    return tr


# ---------------------------------------------------------------- 引擎：百度
def make_baidu(batch_size):
    import requests

    appid = os.getenv("BAIDU_APPID")
    key = os.getenv("BAIDU_KEY")
    if not (appid and key):
        raise SystemExit("缺少 BAIDU_APPID / BAIDU_KEY")
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"

    def tr(texts, src_lang, tgt_lang):
        # 百度支持 q 里用 \n 分隔多条，返回多个 trans_result
        q = "\n".join(texts)
        salt = str(random.randint(32768, 65536))
        sign = hashlib.md5((appid + q + salt + key).encode("utf-8")).hexdigest()
        r = requests.post(url, data={"q": q, "from": src_lang, "to": tgt_lang,
                                     "appid": appid, "salt": salt, "sign": sign},
                          timeout=30)
        r.raise_for_status()
        d = r.json()
        if "error_code" in d and d["error_code"] != "52000":
            raise RuntimeError("百度返回错误 %s: %s" % (d.get("error_code"), d.get("error_msg")))
        res = [x["dst"] for x in d.get("trans_result", [])]
        if len(res) != len(texts):
            raise ValueError("返回 %d 条，期望 %d 条" % (len(res), len(texts)))
        return res

    tr._batch = batch_size
    return tr


# ---------------------------------------------------------------- 引擎：腾讯云
def make_tencent(batch_size):
    """TextTranslate 单条调用；腾讯云的批量接口叫 TextTranslateBatch，这里用单条最稳。"""
    import hashlib as hl
    import hmac
    import requests

    sid = os.getenv("TENCENT_SECRET_ID")
    skey = os.getenv("TENCENT_SECRET_KEY")
    region = os.getenv("TENCENT_REGION", "ap-guangzhou")
    if not (sid and skey):
        raise SystemExit("缺少 TENCENT_SECRET_ID / TENCENT_SECRET_KEY")

    host, service, action, version = "tmt.tencentcloudapi.com", "tmt", "TextTranslate", "2018-03-21"

    def _sign(payload_str, ts):
        date = time.strftime("%Y-%m-%d", time.gmtime(ts))
        creq = "\n".join(["POST", "/", "", "content-type:application/json; charset=utf-8",
                          "host:" + host, "", "content-type;host",
                          hl.sha256(payload_str.encode()).hexdigest()])
        scope = "%s/%s/tc3_request" % (date, service)
        sts = "\n".join(["TC3-HMAC-SHA256", str(ts), scope,
                         hl.sha256(creq.encode()).hexdigest()])
        k = ("TC3" + skey).encode()
        for m in (date, service, "tc3_request"):
            k = hmac.new(k, m.encode(), hashlib.sha256).digest()
        sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
        return ("TC3-HMAC-SHA256 Credential=%s/%s, SignedHeaders=content-type;host, Signature=%s"
                % (sid, scope, sig))

    def one(text, src_lang, tgt_lang):
        payload = {"SourceText": text, "Source": src_lang, "Target": tgt_lang, "ProjectId": 0}
        body = json.dumps(payload, ensure_ascii=False)
        ts = int(time.time())
        headers = {"Content-Type": "application/json; charset=utf-8", "Host": host,
                   "X-TC-Action": action, "X-TC-Version": version,
                   "X-TC-Timestamp": str(ts), "X-TC-Region": region,
                   "Authorization": _sign(body, ts)}
        r = requests.post("https://" + host, data=body.encode(), headers=headers, timeout=30)
        r.raise_for_status()
        d = r.json()["Response"]
        if "Error" in d:
            raise RuntimeError("腾讯云错误 %s: %s" % (d["Error"].get("Code"), d["Error"].get("Message")))
        return d["TargetText"]

    def tr(texts, src_lang, tgt_lang):
        return [one(t, src_lang, tgt_lang) for t in texts]

    tr._batch = 1
    return tr


ENGINES = {"mock": lambda bs: make_mock(), "deepseek": make_deepseek,
           "baidu": make_baidu, "tencent": make_tencent}


# ---------------------------------------------------------------- 主流程
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="deepseek", choices=sorted(ENGINES))
    ap.add_argument("--target", type=int, default=2500,
                    help="训练集目标条数（含原有 1667），0 表示不限")
    ap.add_argument("--batch", type=int, default=8, help="每次请求合并多少条")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（调试）")
    ap.add_argument("--sleep", type=float, default=0.3, help="批次间隔秒数，防限流")
    ap.add_argument("--variants", type=int, default=1,
                    help="每个原句最多生成几条变体。第 2 条起需要 --temperature>0，否则会得到相同译文")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="第 2 条及以后变体用的采样温度；第 1 条始终用 0 保证忠实")
    ap.add_argument("--dry-run", action="store_true", help="不写文件")
    args = ap.parse_args(argv)

    train = json.loads((DATA_DIR / "train.json").read_text(encoding="utf-8"))
    cache = load_cache()
    # have 要包含「原句 + 已生成的所有变体」，否则第二轮会产出与第一轮重复的句子
    have = {d["sentence"] for d in train}
    for vs in cache.values():
        have.update(vs)
    print("训练集 %d 条，目标 %s 条，每句最多 %d 条变体"
          % (len(train), args.target or "不限", args.variants))

    need = (args.target - len(train)) if args.target else 0
    if need <= 0 and args.target:
        print("已经达到目标条数，无需增强。")
        return 0
    if args.limit:
        need = args.limit
    if need <= 0:
        need = len(train) * args.variants

    tr = ENGINES[args.engine](args.batch)
    batch = getattr(tr, "_batch", args.batch)

    done, produced = 0, []
    # 只有「变体数还不够」的句子进队列
    queue = [d for d in train if len(cache.get(d["sentence"], [])) < args.variants]
    random.Random(42).shuffle(queue)
    print("待处理 %d 条，已有缓存 %d 句，批大小 %d"
          % (len(queue), len(cache), batch))

    t0 = time.time()
    for i in range(0, len(queue), batch):
        if len(produced) >= need:
            break
        chunk = queue[i:i + batch]
        texts = [d["sentence"] for d in chunk]
        # 这一批里已有变体的句数 > 0 就升温，制造与上一轮不同的译文
        has_prior = any(len(cache.get(t, [])) >= 1 for t in texts)
        if hasattr(tr, "_set_temp"):
            tr._set_temp(args.temperature if has_prior else 0.0)
        try:
            en = tr(texts, "zh", "en")
            time.sleep(args.sleep)
            back = tr(en, "en", "zh")
            time.sleep(args.sleep)
        except Exception as e:
            print("  [skip] 批次 %d 失败：%s" % (i, str(e)[:120]))
            continue

        for src_d, b in zip(chunk, back):
            b = (b or "").strip()
            if not b or b == src_d["sentence"] or b in have:
                continue                      # 原样返回 / 撞库 → 丢弃
            if len(b) < 6 or len(b) > 200:
                continue                      # 长度异常 → 丢弃
            have.add(b)
            append_cache(src_d["sentence"], b, args.engine)
            produced.append({"idx": len(produced),
                             "sentence": b,
                             "label": src_d["label"],
                             "src_idx": src_d.get("idx"),
                             "method": "backtranslation",
                             "engine": args.engine})
            if len(produced) >= need:
                break

        done += len(chunk)
        if (i // batch) % 20 == 0:
            el = time.time() - t0
            print("  %d/%d 条已请求，产出 %d 条（%.1fs）"
                  % (done, len(queue), len(produced), el), flush=True)

    print("\n产出增强样本 %d 条，用时 %.1fs" % (len(produced), time.time() - t0))
    if args.engine == "mock":
        print("[warn] mock 引擎产出的是乱码，只能用来验证流程，不要写进数据集")

    if args.dry_run or not produced:
        return 0

    if OUT.exists():                          # 合并而不是覆盖
        old = json.loads(OUT.read_text(encoding="utf-8"))
        produced = old + [{**p, "idx": len(old) + k} for k, p in enumerate(produced)]

    OUT.write_text(json.dumps(produced, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→ %s（%d 条）" % (OUT, len(produced)))
    print("\n下一步：python data/merge_augmented.py --report   # 合并进训练集并写 provenance")
    return 0


if __name__ == "__main__":
    sys.exit(main())

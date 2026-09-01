#!/usr/bin/env python3
"""ctx形式のアダプタで【語】行が実際に効いているかを測る（PLAN_20260801.md 手順2）。

**なぜ既存の eval_plot_following.py を使わないか**: あちらは `【あらすじ】/【本文】`
という旧書式で、ctx形式（【作品】【話】【語】【ここまで】…【続き】）で学習した
アダプタには噛み合わない。旧書式を食わせると分布外の入力になり、
「効いていない」のか「書式が違う」のか切り分けられない。

**なぜ gen_server.py を使わないか**: gen_server は viewpoint/genre/body しか
受け取らず、work_title / terms / context を渡す口が無いので ctx形式の
プロンプトを作れない。共有サーバを改造せずに済むよう単独で読み込む。

測り方（同じサンプルに対して条件だけ変える対照実験）:
  real    … 【語】が実物
  none    … 【語】none
  foreign … 【語】を**別作品の語**に差し替える

指標は「生成文に現れた自作品の語のうち、**与えた文脈には出てこないもの**の数」。
文脈に出ている語は写せば当たるので、それを除かないと【語】の効果を測れない
（7/31 の生成比較では、文脈からの写しと【語】の利用を区別できなかった）。

  real > none なら【語】は読まれている。
  foreign で他作品の語が漏れ出すなら、読まれている裏付けになる（強すぎる場合は害）。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TERMS_LINE = re.compile(r"^【語】.*$", re.MULTILINE)
CTX_HEAD = "【ここまで】"
CONT_MARK = "【続き】"


def context_body(prompt: str) -> str:
    """**見出しを除いた**文脈本体だけを返す。

    見出しには【語】行そのものが入っているので、prompt全体を「文脈」として渡すと
    語は必ず「文脈にある」と判定され、指標が常に0になる（8/1未明に実測して発覚）。
    """
    body = prompt.rsplit(CONT_MARK, 1)[0]
    return body.split(CTX_HEAD, 1)[1] if CTX_HEAD in body else body


def load_unseen(data_path: str, sources_path: str, min_terms: int):
    """学習で一度も使われていない作品のサンプルだけを返す。"""
    seen = set()
    if Path(sources_path).exists():
        for line in open(sources_path, encoding="utf-8"):
            seen.update(json.loads(line).get("source_ids") or [])

    out = []
    for line in open(data_path, encoding="utf-8"):
        r = json.loads(line)
        ncode = (r.get("meta") or {}).get("ncode")
        if ncode in seen:
            continue
        terms = terms_of(r["prompt"])
        if len(terms) >= min_terms:
            out.append((ncode, terms, r))
    return out, len(seen)


def terms_of(prompt: str) -> list[str]:
    m = TERMS_LINE.search(prompt)
    if not m:
        return []
    body = m.group(0)[3:].strip()
    if body in ("", "none"):
        return []
    return [t.strip() for t in re.split(r"[、,]", body) if t.strip()]


def with_terms(prompt: str, terms: list[str]) -> str:
    """【語】行だけを差し替える。他は1文字も変えない。"""
    return TERMS_LINE.sub("【語】" + ("、".join(terms) if terms else "none"), prompt, count=1)


def novel_hits(text: str, terms: list[str], context: str) -> list[str]:
    """生成文に現れた語のうち、文脈に出てこないものだけを返す。"""
    return [t for t in terms if t in text and t not in context]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--data", default="./data/train_ctx_subset/balanced-00000.jsonl")
    ap.add_argument("--sources", default="./runs/batch_sources.jsonl")
    ap.add_argument("--works", type=int, default=10, help="使う作品数")
    ap.add_argument("--samples", type=int, default=2, help="条件あたりの生成本数")
    ap.add_argument("--min-terms", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--out", help="生成文を保存するJSONL")
    args = ap.parse_args()

    random.seed(args.seed)
    pool, n_seen = load_unseen(args.data, args.sources, args.min_terms)
    print(f"[eval] 学習で消費済みの作品 {n_seen}件 / 未使用かつ語{args.min_terms}個以上の"
          f"サンプル {len(pool)}件", file=sys.stderr)
    if len(pool) < 2:
        print("[eval] サンプルが足りません", file=sys.stderr)
        return 1

    # 作品ごとに1件だけ拾う（同じ作品を何度も測っても独立な観測にならない）
    by_work: dict[str, tuple] = {}
    for ncode, terms, r in pool:
        by_work.setdefault(ncode, (ncode, terms, r))
    picked = list(by_work.values())
    random.shuffle(picked)
    picked = picked[: args.works]
    print(f"[eval] 対象 {len(picked)}作品", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16)
    print("[eval] モデル読み込み中...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=qc, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print("[eval] 読み込み完了", file=sys.stderr)

    def gen(prompt: str, seed: int) -> str:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=args.temperature,
                                 top_p=0.95, repetition_penalty=1.05,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    writer = open(args.out, "w", encoding="utf-8") if args.out else None
    score = {"real": [], "none": [], "foreign": []}
    raw = {"real": [], "none": [], "foreign": []}   # 文脈フィルタを外した素の出現数
    leak: list[int] = []

    for idx, (ncode, terms, r) in enumerate(picked):
        # 差し替え用に別作品の語を借りる
        others = [t for n, t, _ in picked if n != ncode]
        foreign = random.choice(others) if others else []
        variants = {
            "real": with_terms(r["prompt"], terms),
            "none": with_terms(r["prompt"], []),
            "foreign": with_terms(r["prompt"], foreign),
        }
        for cond, prompt in variants.items():
            for s in range(args.samples):
                text = gen(prompt, args.seed + idx * 100 + s)
                ctx = context_body(prompt)
                hits = novel_hits(text, terms, ctx)
                score[cond].append(len(hits))
                raw[cond].append(sum(1 for t in terms if t in text))
                rec = {"ncode": ncode, "cond": cond, "sample": s,
                       "own_terms": terms, "hits": hits, "text": text}
                if cond == "foreign":
                    fl = novel_hits(text, foreign, ctx)
                    leak.append(len(fl))
                    rec["foreign_terms"] = foreign
                    rec["foreign_hits"] = fl
                if writer:
                    writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[eval] {idx + 1}/{len(picked)} {ncode} 完了", file=sys.stderr)

    if writer:
        writer.close()

    print("\n=== 【語】の効果（自作品の語のうち、文脈に無いものの出現数）===")
    print(f"{'条件':<10}{'平均':>8}{'中央値':>8}{'1語以上出た率':>16}{'n':>6}")
    for cond in ("real", "none", "foreign"):
        v = score[cond]
        if not v:
            continue
        rate = 100 * sum(1 for x in v if x > 0) / len(v)
        print(f"{cond:<10}{st.mean(v):>8.2f}{st.median(v):>8.1f}{rate:>15.1f}%{len(v):>6}")

    if leak:
        rate = 100 * sum(1 for x in leak if x > 0) / len(leak)
        print(f"\n他作品の語の漏れ出し（foreign条件）: 平均 {st.mean(leak):.2f}語 / "
              f"1語以上 {rate:.1f}%")

    # **全条件ゼロは「効いていない」ではなく「測れていない」**（2026-08-01 実測で発覚）。
    # 生成長200トークン(≒300字)では、文脈に無い固有名詞が登場する事自体が稀で、
    # 学習データの出現率から見た期待値そのものが1本あたり0.1〜0.2語しかない。
    # 差を検出できない設定で「効果なし」と結論すると誤った判断に繋がる。
    raw_all = [x for v in raw.values() for x in v]
    print(f"\n参考: 文脈フィルタを外した素の出現数 平均 {st.mean(raw_all):.2f}語 "
          f"（フィルタ後が0でこれが>0なら、出ている語は全て文脈由来）")

    real, none = st.mean(score["real"]), st.mean(score["none"])
    print("\n--- 読み方 ---")
    if max(st.mean(v) for v in score.values()) == 0:
        print("**測定が成立していない（検出力不足）。** 全条件で0語のため差を検出できない。")
        print(f"  生成長 {args.max_new_tokens}トークン(≒{int(args.max_new_tokens * 1.5)}字)は短すぎる。")
        print("  → --max-new-tokens 600 以上、--works/--samples を増やして再測定すること。")
        if st.mean(raw_all) > 0:
            print("  なお素の出現は平均 %.2f語あり、**出た語は全て文脈由来**だった。"
                  % st.mean(raw_all))
            print("  「文脈に無い名前を新たに出すことは無かった」という所見は残るが、"
                  "この生成長では期待値自体が低く、確証にはならない。")
    elif real > none * 1.5 and real > 0.2:
        print("【語】は読まれている。real が none を明確に上回った。")
    elif real <= none:
        print("【語】は効いていない。real が none を上回らない。"
              "\n  → 行を置いてあるだけで信号になっていない。書式か抽出方法の見直しが要る。")
    else:
        print("差が小さく判定不能。--works / --samples を増やして再測定すること。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
